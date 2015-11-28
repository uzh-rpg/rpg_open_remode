// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <rmd/seed_matrix.cuh>

#include <rmd/texture_memory.cuh>
#include <rmd/helper_vector_types.cuh>

#include "seed_init.cu"
#include "seed_check.cu"
#include "epipolar_match.cu"
#include "seed_update.cu"

rmd::SeedMatrix::SeedMatrix(
    const size_t &width,
    const size_t &height,
    const PinholeCamera &cam)
  : width_(width)
  , height_(height)
  , ref_img_(width, height)
  , curr_img_(width, height)
  , sum_templ_(width, height)
  , const_templ_denom_(width, height)
  , mu_(width, height)
  , sigma_(width, height)
  , a_(width, height)
  , b_(width, height)
  , convergence_(width, height)
  , epipolar_matches_(width, height)
{
  // Save image details to be uploaded to device memory
  dev_data_.ref_img = ref_img_.dev_ptr;
  dev_data_.curr_img = curr_img_.dev_ptr;
  dev_data_.sum_templ = sum_templ_.dev_ptr;
  dev_data_.const_templ_denom = const_templ_denom_.dev_ptr;
  dev_data_.mu = mu_.dev_ptr;
  dev_data_.sigma = sigma_.dev_ptr;
  dev_data_.a = a_.dev_ptr;
  dev_data_.b = b_.dev_ptr;
  dev_data_.convergence = convergence_.dev_ptr;
  dev_data_.epipolar_matches = epipolar_matches_.dev_ptr;

  // Save camera parameters
  dev_data_.cam = cam;
  dev_data_.one_pix_angle = cam.getOnePixAngle();
  dev_data_.width  = width;
  dev_data_.height = height;
  // Device image size
  host_img_size_.width  = width_;
  host_img_size_.height = height_;

  // Kernel configuration for depth estimation
  dim_block_.x = 16;
  dim_block_.y = 16;
  dim_grid_.x = (width  + dim_block_.x - 1) / dim_block_.x;
  dim_grid_.y = (height + dim_block_.y - 1) / dim_block_.y;

  // Image reducer to compute statistics on seeds
  dim3 num_threads_per_block;
  dim3 num_blocks_per_grid;
  num_threads_per_block.x = 16;
  num_threads_per_block.y = 16;
  num_blocks_per_grid.x = 4;
  num_blocks_per_grid.y = 4;
  img_reducer_ = new ImageReducer<int>(num_threads_per_block, num_blocks_per_grid);
}

rmd::SeedMatrix::~SeedMatrix()
{
  delete img_reducer_;
}

bool rmd::SeedMatrix::setReferenceImage(
    float *host_ref_img_align_row_maj,
    const rmd::SE3<float> &T_curr_world,
    const float &min_depth,
    const float &max_depth)
{
  // Upload reference image to device memory
  ref_img_.setDevData(host_ref_img_align_row_maj);
  // Set scene parameters
  dev_data_.scene.min_depth    = min_depth;
  dev_data_.scene.max_depth    = max_depth;
  dev_data_.scene.avg_depth    = (min_depth+max_depth)/2.0f;
  dev_data_.scene.depth_range  = max_depth - min_depth;
  dev_data_.scene.sigma_sq_max = dev_data_.scene.depth_range * dev_data_.scene.depth_range / 36.0f;
  // Algorithm parameters
  dev_data_.eta_inlier  = 0.7f;
  dev_data_.eta_outlier = 0.05f;
  dev_data_.epsilon     = dev_data_.scene.depth_range / 1000.0f;
  // Copy data to device memory
  dev_data_.setDevData();

  T_world_ref_ = T_curr_world.inv();

  rmd::bindTexture(ref_img_tex, ref_img_);

  rmd::seedInitKernel<<<dim_grid_, dim_block_>>>(dev_data_.dev_ptr);
  cudaDeviceSynchronize();

  rmd::bindTexture(sum_templ_tex, sum_templ_, cudaFilterModePoint);
  rmd::bindTexture(const_templ_denom_tex, const_templ_denom_, cudaFilterModePoint);
  return true;
}

bool rmd::SeedMatrix::update(
    float *host_curr_img_align_row_maj,
    const SE3<float> &T_curr_world)
{
  const rmd::SE3<float> T_curr_ref = T_curr_world * T_world_ref_;
  dist_from_ref_ = norm(T_curr_ref.getTranslation());

  // Upload current image to device memory
  curr_img_.setDevData(host_curr_img_align_row_maj);
  // Bind texture memory for the current image
  rmd::bindTexture(curr_img_tex, curr_img_);

  // ... and model parameters
  rmd::bindTexture(mu_tex, mu_);
  rmd::bindTexture(sigma_tex, sigma_);
  rmd::bindTexture(a_tex, a_);
  rmd::bindTexture(b_tex, b_);

  // Assest current convergence status
  rmd::seedCheckKernel<<<dim_grid_, dim_block_>>>(dev_data_.dev_ptr);
  cudaError err = cudaDeviceSynchronize();
  if(cudaSuccess != err)
    throw CudaException("SeedMatrix: unable to synchronize device", err);
  rmd::bindTexture(convergence_tex, convergence_, cudaFilterModePoint);

  // Establish epipolar correspondences
  // call epipolar matching kernel
  rmd::copyImgSzToConst(&host_img_size_);

  rmd::seedEpipolarMatchKernel<<<dim_grid_, dim_block_>>>(dev_data_.dev_ptr, T_curr_ref);
  err = cudaDeviceSynchronize();
  if(cudaSuccess != err)
    throw CudaException("SeedMatrix: unable to synchronize device", err);
  rmd::bindTexture(epipolar_matches_tex, epipolar_matches_);

  rmd::seedUpdateKernel<<<dim_grid_, dim_block_>>>(dev_data_.dev_ptr, T_curr_ref.inv());

  return true;
}

void rmd::SeedMatrix::downloadDepthmap(float *host_depthmap_align_row_maj) const
{
  mu_.getDevData(host_depthmap_align_row_maj);
}

void rmd::SeedMatrix::downloadConvergence(int *host_align_row_maj) const
{
  convergence_.getDevData(host_align_row_maj);
}

const rmd::DeviceImage<float> & rmd::SeedMatrix::getMu() const
{
  return mu_;
}

const rmd::DeviceImage<float> & rmd::SeedMatrix::getSigmaSq() const
{
  return sigma_;
}

const rmd::DeviceImage<float> & rmd::SeedMatrix::getA() const
{
  return a_;
}

const rmd::DeviceImage<float> & rmd::SeedMatrix::getB() const
{
  return b_;
}

const rmd::DeviceImage<int> & rmd::SeedMatrix::getConvergence() const
{
  return convergence_;
}

size_t rmd::SeedMatrix::getConvergedCount() const
{
  return img_reducer_->countEqual(convergence_, ConvergenceStates::CONVERGED);
}

float rmd::SeedMatrix::getDistFromRef() const
{
  return dist_from_ref_;
}

#if RMD_BUILD_TESTS
void rmd::SeedMatrix::downloadSigmaSq(float *host_align_row_maj) const
{
  sigma_.getDevData(host_align_row_maj);
}
void rmd::SeedMatrix::downloadA(float *host_align_row_maj) const
{
  a_.getDevData(host_align_row_maj);
}
void rmd::SeedMatrix::downloadB(float *host_align_row_maj) const
{
  b_.getDevData(host_align_row_maj);
}
void rmd::SeedMatrix::downloadSumTempl(float *host_align_row_maj) const
{
  sum_templ_.getDevData(host_align_row_maj);
}
void rmd::SeedMatrix::downloadConstTemplDenom(float *host_align_row_maj) const
{
  const_templ_denom_.getDevData(host_align_row_maj);
}
void rmd::SeedMatrix::downloadEpipolarMatches(float2 *host_align_row_maj) const
{
  epipolar_matches_.getDevData(host_align_row_maj);
}
#endif
