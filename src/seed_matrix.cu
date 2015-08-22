#include <rmd/seed_matrix.cuh>
#include <rmd/texture_memory.cuh>
#include <rmd/helper_vector_types.cuh>

#include "seed_init.cu"
#include "seed_update.cu"
#include "seed_check.cu"
#include "epipolar_match.cu"

#include <stdio.h>

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
  dev_data_.ref_img.set(ref_img_);
  dev_data_.curr_img.set(curr_img_);
  dev_data_.sum_templ.set(sum_templ_);
  dev_data_.const_templ_denom.set(const_templ_denom_);
  dev_data_.mu.set(mu_);
  dev_data_.sigma.set(sigma_);
  dev_data_.a.set(a_);
  dev_data_.b.set(b_);
  dev_data_.convergence.set(convergence_);
  dev_data_.epipolar_matches.set(epipolar_matches_);
  // Save camera parameters
  dev_data_.cam    = cam;
  dev_data_.one_pix_angle = cam.getOnePixAngle();
  dev_data_.width  = width;
  dev_data_.height = height;
  // Kernel configuration
  dim_block_.x = 16;
  dim_block_.y = 16;
  dim_grid_.x = (width  + dim_block_.x - 1) / dim_block_.x;
  dim_grid_.y = (height + dim_block_.y - 1) / dim_block_.y;
}

bool rmd::SeedMatrix::setReferenceImage(
    float *host_ref_img_align_row_maj,
    const SE3<float> &T_curr_world,
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
  dev_data_.epsilon     = dev_data_.scene.depth_range / 10000.0f;
  // Copy data to device memory
  dev_data_.setDevData();

  T_world_ref_ = T_curr_world.inv();

  rmd::bindTexture(ref_img_tex, ref_img_);

  rmd::seedInitKernel<<<dim_grid_, dim_block_>>>(dev_data_.dev_ptr);
  cudaDeviceSynchronize();
  return true;
}

bool rmd::SeedMatrix::update(
    float *host_curr_img_align_row_maj,
    const SE3<float> &T_curr_world)
{
  const rmd::SE3<float> T_curr_ref = T_curr_world * T_world_ref_;

  // Upload current image to device memory
  curr_img_.setDevData(host_curr_img_align_row_maj);
  // Bind texture memory for the current image
  rmd::bindTexture(curr_img_tex, curr_img_);

  // ... and model parameters
  rmd::bindTexture(mu_tex, mu_);
  rmd::bindTexture(sigma_tex, sigma_);
  rmd::bindTexture(a_tex, a_);
  rmd::bindTexture(b_tex, b_);

  printf("zieta\n");

  // Assest current convergence status
  rmd::seedCheckKernel<<<dim_grid_, dim_block_>>>(dev_data_.dev_ptr);
  rmd::bindTexture(convergence_tex, convergence_);

  cudaDeviceSynchronize();
  // Establish epipolar correspondences
  // call epipolar matching kernel
  rmd::seedEpipolarMatch<<<dim_grid_, dim_block_>>>(
                                                    dev_data_.dev_ptr,
                                                    T_curr_ref);
  rmd::bindTexture(epipolar_matches_tex, epipolar_matches_);

  rmd::seedUpdateKernel<<<dim_grid_, dim_block_>>>(
                                                   dev_data_.dev_ptr,
                                                   T_curr_ref.inv());
  return true;
}

void rmd::SeedMatrix::downloadDepthmap(float *host_depthmap_align_row_maj)
{
  mu_.getDevData(host_depthmap_align_row_maj);
}

#ifdef RMD_DEBUG
void rmd::SeedMatrix::downloadDisparity(
    float *host_disp_x_align_row_maj,
    float *host_disp_y_align_row_maj)
{
  float2 disp[640*480];
  epipolar_matches_.getDevData(disp);

  for(size_t y = 0; y<480; ++y)
  {
    for(size_t x = 0; x<640; ++x)
    {
      const size_t i = y*640+x;
      host_disp_x_align_row_maj[i] = disp[i].x;
      host_disp_y_align_row_maj[i] = disp[i].y;
    }
  }
}

void rmd::SeedMatrix::downloadConvergence(
    unsigned char *host_conv_align_row_maj)
{
  unsigned char conv[640*480];
  convergence_.getDevData(conv);
  for(size_t y = 0; y<480; ++y)
  {
    for(size_t x = 0; x<640; ++x)
    {
      const size_t i = y*640+x;
      host_conv_align_row_maj[i] = conv[i];
    }
  }
}
#endif
