#include <rmd/depthmap_denoiser.cuh>
#include <rmd/texture_memory.cuh>

rmd::DepthmapDenoiser::DepthmapDenoiser(size_t width, size_t height)
  : u_(width, height)
  , u_head_(width, height)
  , p_(width, height)
  , g_(width, height)
{
  dim_block_.x = 16;
  dim_block_.y = 16;
  dim_grid_.x = (width  + dim_block_.x - 1) / dim_block_.x;
  dim_grid_.y = (height + dim_block_.y - 1) / dim_block_.y;
}

void rmd::DepthmapDenoiser::denoise(
    const rmd::DeviceImage<float> &mu,
    const rmd::DeviceImage<float> &sigma_sq,
    const rmd::DeviceImage<float> &a,
    const rmd::DeviceImage<float> &b,
    float *host_denoised)
{
  rmd::bindTexture(mu_tex, mu);
  rmd::bindTexture(sigma_tex, sigma_sq);
  rmd::bindTexture(a_tex, a);
  rmd::bindTexture(b_tex, b);

  const float L = sqrtf(8.0f);
  const float tau = 0.02f;
  const float sigma = (1.0f / (L * L)) / tau;
  const float theta = 0.5f;

  u_ = mu;
  u_head_ = u_;
  p_.zero();

  for (int i = 0; i < 200; ++i)
  {

  }
}

void rmd::DepthmapDenoiser::setLargeSigmaSq(float depth_range)
{
  large_sigma_sq_=  depth_range * depth_range / 72.0f;
}
