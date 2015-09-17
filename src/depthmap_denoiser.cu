#include <iostream>
#include <rmd/depthmap_denoiser.cuh>
#include <rmd/texture_memory.cuh>

namespace rmd
{

template<typename T>
inline __device__
T max(T a, T b)
{
  return a > b ? a : b;
}

template<typename T>
inline __device__
T min(T a, T b)
{
  return a < b ? a : b;
}

__global__
void computeWeightsKernel(denoise::DeviceData *dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if (x < dev_ptr->width && y < dev_ptr->height)
  {
    const float E_pi = tex2D(a_tex, xx, yy) / (tex2D(a_tex, xx, yy) + tex2D(b_tex, xx, yy));
    dev_ptr->g->atXY(x, y) = rmd::max<float>((E_pi*tex2D(sigma_tex, xx, yy)+(1.0f-E_pi)*dev_ptr->large_sigma_sq)/dev_ptr->large_sigma_sq, 1.0f);
  }
}

} // rmd namespace

rmd::denoise::DeviceData::DeviceData(DeviceImage<float> * const u_dev_ptr,
                                     DeviceImage<float> * const u_head_dev_ptr,
                                     DeviceImage<float> * const p_dev_ptr,
                                     DeviceImage<float> * const g_dev_ptr,
                                     const size_t &w,
                                     const size_t &h)
  : L(sqrt(8.0f))
  , tau(0.02f)
  , sigma((1 / (L*L)) / tau)
  , theta(0.5f)
  , u(u_dev_ptr)
  , u_head(u_head_dev_ptr)
  , p(p_dev_ptr)
  , g(g_dev_ptr)
  , width(w)
  , height(h)
{ }

rmd::DepthmapDenoiser::DepthmapDenoiser(size_t width, size_t height)
  : u_(width, height)
  , u_head_(width, height)
  , p_(width, height)
  , g_(width, height)
{
  host_ptr = new rmd::denoise::DeviceData(
        u_.dev_ptr,
        u_head_.dev_ptr,
        p_.dev_ptr,
        g_.dev_ptr,
        width,
        height);
  const cudaError err = cudaMalloc(
        &dev_ptr,
        sizeof(*host_ptr));
  if(cudaSuccess != err)
    throw CudaException("DeviceData, cannot allocate device memory.", err);

  dim_block_.x = 16;
  dim_block_.y = 16;
  dim_grid_.x = (width  + dim_block_.x - 1) / dim_block_.x;
  dim_grid_.y = (height + dim_block_.y - 1) / dim_block_.y;
}

rmd::DepthmapDenoiser::~DepthmapDenoiser()
{
  delete host_ptr;
  const cudaError err = cudaFree(dev_ptr);
  if(cudaSuccess != err)
    throw CudaException("DeviceData, unable to free device memory.", err);
}

void rmd::DepthmapDenoiser::denoise(
    const rmd::DeviceImage<float> &mu,
    const rmd::DeviceImage<float> &sigma_sq,
    const rmd::DeviceImage<float> &a,
    const rmd::DeviceImage<float> &b,
    float *host_denoised)
{
  // large_sigma_sq must be set before calling this method
  if(host_ptr->large_sigma_sq < 0.0f)
  {
    std::cerr << "ERROR: setLargeSigmaSq must be called before this method" << std::endl;
    return;
  }
  const cudaError err = cudaMemcpy(
        dev_ptr,
        host_ptr,
        sizeof(*host_ptr),
        cudaMemcpyHostToDevice);
  if(cudaSuccess != err)
    throw CudaException("DeviceData, cannot copy to device memory.", err);

  rmd::bindTexture(mu_tex, mu);
  rmd::bindTexture(sigma_tex, sigma_sq);
  rmd::bindTexture(a_tex, a);
  rmd::bindTexture(b_tex, b);

  rmd::computeWeightsKernel<<<dim_grid_, dim_block_>>>(dev_ptr);
  rmd::bindTexture(g_tex, g_);

  u_ = mu;
  u_head_ = u_;
  p_.zero();

  for (int i = 0; i < 200; ++i)
  {

  }
  u_.getDevData(host_denoised);
}

void rmd::DepthmapDenoiser::setLargeSigmaSq(float depth_range)
{
  host_ptr->large_sigma_sq =  depth_range * depth_range / 72.0f;
}
