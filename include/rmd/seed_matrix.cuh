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

#ifndef SEED_MATRIX_CUH
#define SEED_MATRIX_CUH

#include <cuda_runtime.h>
#include <rmd/device_image.cuh>
#include <rmd/pinhole_camera.cuh>
#include <rmd/mvs_device_data.cuh>
#include <rmd/se3.cuh>
#include <rmd/reduction.cuh>

namespace rmd
{

namespace ConvergenceStates
{
enum ConvergenceState
{
  UPDATE = 0,
  CONVERGED,
  BORDER,
  DIVERGED,
  NO_MATCH,
  NOT_VISIBLE
};
}
typedef ConvergenceStates::ConvergenceState ConvergenceState;

class SeedMatrix
{
public:
  SeedMatrix(
      const size_t &width,
      const size_t &height,
      const PinholeCamera &cam);
  ~SeedMatrix();
  bool setReferenceImage(
      float *host_ref_img_align_row_maj,
      const SE3<float> &T_curr_world,
      const float &min_depth,
      const float &max_depth);
  bool update(
      float *host_curr_img_align_row_maj,
      const SE3<float> &T_curr_world);

  void downloadDepthmap(float *host_depthmap_align_row_maj) const;
  void downloadConvergence(int *host_align_row_maj) const;


  const DeviceImage<float> & getMu() const;
  const DeviceImage<float> & getSigmaSq() const;
  const DeviceImage<float> & getA() const;
  const DeviceImage<float> & getB() const;

  const DeviceImage<int> & getConvergence() const;

  size_t getConvergedCount() const;

  float getDistFromRef() const;

#if RMD_BUILD_TESTS
  void downloadSigmaSq(float *host_align_row_maj) const;
  void downloadA(float *host_align_row_maj) const;
  void downloadB(float *host_align_row_maj) const;
  void downloadSumTempl(float *host_align_row_maj) const;
  void downloadConstTemplDenom(float *host_align_row_maj) const;
  void downloadEpipolarMatches(float2 *host_align_row_maj) const;
#endif

private:
  size_t width_;
  size_t height_;
  DeviceImage<float> ref_img_, curr_img_;
  // Template statistics for NCC (pre)computation
  DeviceImage<float> sum_templ_, const_templ_denom_;
  // Measurement parameters
  DeviceImage<float> mu_, sigma_, a_, b_;
  // Convergence state
  DeviceImage<int> convergence_;
  // Epipolar matches
  DeviceImage<float2> epipolar_matches_;
  mvs::DeviceData dev_data_;
  SE3<float> T_world_ref_;
  float dist_from_ref_;
  // kernel config
  dim3 dim_block_;
  dim3 dim_grid_;
  // Image reduction to compute seed statistics
  ImageReducer<int> *img_reducer_;

  // Image size to be copied to constant memory
  Size host_img_size_;
};

} // rmd namespace

#endif // SEED_MATRIX_CUH
