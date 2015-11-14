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

#include <iostream>
#include <rmd/depthmap_denoiser.cuh>
#include <rmd/texture_memory.cuh>
#include <cuda_toolkit/helper_math.h>

namespace rmd
{

namespace denoise
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

__constant__ struct Size c_img_size;

__global__
void computeWeightsKernel(denoise::DeviceData *dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if (x < c_img_size.width && y < c_img_size.height)
  {
    const float E_pi = tex2D(a_tex, xx, yy) / (tex2D(a_tex, xx, yy) + tex2D(b_tex, xx, yy));
    dev_ptr->g->atXY(x, y) = max<float>((E_pi*tex2D(sigma_tex, xx, yy)+(1.0f-E_pi)*dev_ptr->large_sigma_sq)/dev_ptr->large_sigma_sq, 1.0f);
  }
}

__global__
void updateTVL1PrimalDualKernel(denoise::DeviceData *dev_ptr)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const float xx = x+0.5f;
  const float yy = y+0.5f;

  if (x < c_img_size.width && y < c_img_size.height)
  {
    const float noisy_depth = tex2D(mu_tex, xx, yy );
    const float old_u = dev_ptr->u->atXY(x, y);
    const float g = tex2D(g_tex, xx, yy);
    // update dual
    const float2 p = dev_ptr->p->atXY(x, y);
    float2 grad_uhead = make_float2(0.0f, 0.0f);
    const float current_u = dev_ptr->u->atXY(x, y);
    grad_uhead.x = dev_ptr->u_head->atXY(min<int>(c_img_size.width-1, x+1), y)  - current_u;
    grad_uhead.y = dev_ptr->u_head->atXY(x, min<int>(c_img_size.height-1, y+1)) - current_u;
    const float2 temp_p = g* grad_uhead * dev_ptr->sigma + p;
    const float sqrt_p = sqrt(temp_p.x * temp_p.x + temp_p.y * temp_p.y);
    dev_ptr->p->atXY(x, y) = temp_p / max<float>(1.0f, sqrt_p);

    __syncthreads();

    // update primal
    float2 current_p = dev_ptr->p->atXY(x, y);
    float2 w_p = dev_ptr->p->atXY(max<int>(0, x-1), y);
    float2 n_p = dev_ptr->p->atXY(x, max<int>(0, y-1));
    if (x == 0)
      w_p.x = 0.0f;
    else if (x >= c_img_size.width - 1)
      current_p.x = 0.0f;
    if (y == 0)
      n_p.y = 0.0f;
    else if (y >= c_img_size.height - 1)
      current_p.y = 0.0f;
    const float divergence = current_p.x - w_p.x + current_p.y - n_p.y;

    float temp_u = old_u + dev_ptr->tau * g * divergence;
    if ((temp_u - noisy_depth) > (dev_ptr->tau * dev_ptr->lambda))
    {
      dev_ptr->u->atXY(x, y) = temp_u - dev_ptr->tau * dev_ptr->lambda;
    }
    else if ((temp_u - noisy_depth) < (-dev_ptr->tau * dev_ptr->lambda))
    {
      dev_ptr->u->atXY(x, y) = temp_u + dev_ptr->tau * dev_ptr->lambda;
    }
    else
    {
      dev_ptr->u->atXY(x, y) = noisy_depth;
    }
    dev_ptr->u_head->atXY(x, y) = dev_ptr->u->atXY(x, y)
        + dev_ptr->theta * (dev_ptr->u->atXY(x, y) - old_u);
  }
  __syncthreads();
}

} // denoise namespace

} // rmd namespace

rmd::denoise::DeviceData::DeviceData(DeviceImage<float>  * const u_dev_ptr,
                                     DeviceImage<float>  * const u_head_dev_ptr,
                                     DeviceImage<float2> * const p_dev_ptr,
                                     DeviceImage<float>  * const g_dev_ptr,
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
  , lambda(0.2f)
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

  host_img_size_.width  = width;
  host_img_size_.height = height;
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
    float *host_denoised,
    float lambda,
    int iterations)
{
  // large_sigma_sq must be set before calling this method
  if(host_ptr->large_sigma_sq < 0.0f)
  {
    std::cerr << "ERROR: setLargeSigmaSq must be called before this method" << std::endl;
    return;
  }
  host_ptr->lambda = lambda;
  cudaError err = cudaMemcpy(
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

  err = cudaMemcpyToSymbol(rmd::denoise::c_img_size, &host_img_size_, sizeof(rmd::Size));
  if(cudaSuccess != err)
    throw CudaException("DepthmapDenoiser: unable to copy to const memory", err);

  rmd::denoise::computeWeightsKernel<<<dim_grid_, dim_block_>>>(dev_ptr);
  rmd::bindTexture(g_tex, g_);

  u_ = mu;
  u_head_ = u_;
  p_.zero();

  for(int i = 0; i < iterations; ++i)
  {
    rmd::denoise::updateTVL1PrimalDualKernel<<<dim_grid_, dim_block_>>>(dev_ptr);
  }
  u_.getDevData(host_denoised);
}

void rmd::DepthmapDenoiser::setLargeSigmaSq(float depth_range)
{
  host_ptr->large_sigma_sq =  depth_range * depth_range / 72.0f;
}
