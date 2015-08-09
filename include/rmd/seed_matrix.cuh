#ifndef SEED_MATRIX_CUH
#define SEED_MATRIX_CUH

#include <cuda_runtime.h>
#include <rmd/image.cuh>
#include <rmd/pinhole_camera.cuh>
#include <rmd/device_data.cuh>
#include <rmd/se3.cuh>

namespace rmd
{

class SeedMatrix
{
public:
  SeedMatrix(
      const size_t &width,
      const size_t &height,
      const PinholeCamera &cam);
  bool setReferenceImage(
      float *host_ref_img_align_row_maj,
      const SE3<float> &T_curr_world,
      const float &min_depth,
      const float &max_depth);
  bool update(
      float *host_curr_img_align_row_maj,
      const SE3<float> &T_curr_world);
private:
  size_t width_;
  size_t height_;
  Image<float> ref_img_, curr_img_;
  // Template statistics for NCC (pre)computation
  Image<float> sum_templ_, const_templ_denom_;
  // measurement parameters
  Image<float> mu_, sigma_, a_, b_;
  DeviceData dev_data_;
  SE3<float> T_world_ref_;
  // kernel config
  dim3 dim_block_;
  dim3 dim_grid_;
};

} // rmd namespace

#endif // SEED_MATRIX_CUH
