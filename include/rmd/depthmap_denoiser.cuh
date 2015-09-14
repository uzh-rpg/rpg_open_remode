#ifndef RMD_DEPTHMAP_DENOISER_CUH
#define RMD_DEPTHMAP_DENOISER_CUH

#include <rmd/device_image.cuh>

namespace rmd
{

class DepthmapDenoiser
{

public:
  DepthmapDenoiser(size_t width, size_t height, float depth_range);
  void denoise(
      const rmd::DeviceImage<float> &mu,
      const rmd::DeviceImage<float> &sigma_sq,
      const rmd::DeviceImage<float> &a,
      const rmd::DeviceImage<float> &b,
      float *host_denoised);
private:
  DeviceImage<float> u_;
  DeviceImage<float> u_head_;
  DeviceImage<float> p_;
  DeviceImage<float> g_;

  dim3 dim_block_;
  dim3 dim_grid_;

  const float large_sigma_sq_;
};

}

#endif // RMD_DEPTHMAP_DENOISER_CUH
