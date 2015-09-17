#ifndef RMD_DEPTHMAP_DENOISER_CUH
#define RMD_DEPTHMAP_DENOISER_CUH

#include <rmd/device_image.cuh>

namespace rmd
{

namespace denoise
{

struct DeviceData
{
  DeviceData(
      DeviceImage<float> * const u_dev_ptr,
      DeviceImage<float> * const u_head_dev_ptr,
      DeviceImage<float> * const p_dev_ptr,
      DeviceImage<float> * const g_dev_ptr,
      const size_t &w,
      const size_t &h);
  const float L;
  const float tau;
  const float sigma;
  const float theta;

  DeviceImage<float> * const u;
  DeviceImage<float> * const u_head;
  DeviceImage<float> * const p;
  DeviceImage<float> * const g;

  const size_t width;
  const size_t height;

  float large_sigma_sq;
};

} // denoise namespace

class DepthmapDenoiser
{

public:
  DepthmapDenoiser(size_t width, size_t height);
  ~DepthmapDenoiser();
  void denoise(
      const rmd::DeviceImage<float> &mu,
      const rmd::DeviceImage<float> &sigma_sq,
      const rmd::DeviceImage<float> &a,
      const rmd::DeviceImage<float> &b,
      float *host_denoised);
  void setLargeSigmaSq(float depth_range);
private:
  DeviceImage<float> u_;
  DeviceImage<float> u_head_;
  DeviceImage<float> p_;
  DeviceImage<float> g_;

  denoise::DeviceData * host_ptr;
  denoise::DeviceData * dev_ptr;

  dim3 dim_block_;
  dim3 dim_grid_;
};

} // rmd namespace

#endif // RMD_DEPTHMAP_DENOISER_CUH
