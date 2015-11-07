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

#include <rmd/reduction.cuh>
#include <cuda_toolkit/helper_timer.h>

namespace rmd
{

__global__
void reductionSumKernel(int *out_dev_ptr,
                        size_t out_stride,
                        const int *in_dev_ptr,
                        size_t in_stride,
                        size_t n,
                        size_t m)
{
  extern __shared__ int s_partial[];
  int count = 0;

  // Sum over 2D thread grid, use (x,y) indices
  for(int x = blockIdx.x * blockDim.x + threadIdx.x;
      x < n;
      x += blockDim.x*gridDim.x)
  {
    for(int y = blockIdx.y * blockDim.y + threadIdx.y;
        y < m;
        y += blockDim.y*gridDim.y)
    {
      count += in_dev_ptr[y*in_stride+x];
    }
  }
  // Sums are written to shared memory, single index
  s_partial[threadIdx.y*blockDim.x+threadIdx.x] = count;
  __syncthreads();

  // Reduce over block sums stored in shared memory
  // Start using half the block threads,
  // halve the active threads at each iteration
  const int tid = threadIdx.y*blockDim.x+threadIdx.x;
  for (int num_active_threads = (blockDim.x*blockDim.y)>>1;
       num_active_threads;
       num_active_threads >>= 1 ) {
    if ( tid < num_active_threads)
    {
      s_partial[tid] += s_partial[tid+num_active_threads];
    }
    __syncthreads();
  }
  // Thread 0 writes the result for the block
  if(0 == tid)
  {
    out_dev_ptr[blockIdx.y*out_stride+blockIdx.x] = s_partial[0];
  }
}

__global__
void maskKernel(int *out_dev_ptr,
                size_t out_stride,
                const int *in_dev_ptr,
                size_t in_stride,
                size_t n,
                size_t m,
                int value)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= n || y >= m)
    return;

  out_dev_ptr[y*out_stride+x] = in_dev_ptr[y*in_stride+x] == value ? 1 : 0;
}

} // rmd namespace

// Sum image by reduction
// Cfr. listing 12.1 by N. Wilt, "The CUDA Handbook"
int rmd::sum(const int *in_img_data,
             size_t in_img_stride,
             size_t in_img_width,
             size_t in_img_height)
{
  // Kernel configuration
  dim3 dim_block;
  dim3 dim_grid;
  dim_block.x = 16; // Num threads
  dim_block.y = 16;
  dim_grid.x = 4;   // Num blocks
  dim_grid.y = 4;

  // Compute required amount of shared memory
  unsigned int sh_mem_size = dim_block.x * dim_block.y * sizeof(int);

  // Allocate intermediate result (TODO: this should be pre-allocated)
  int *d_partial;
  size_t d_partial_pitch;
  cudaError err = cudaMallocPitch(
        &d_partial,
        &d_partial_pitch,
        dim_grid.x*sizeof(int),
        dim_grid.y);
  if(cudaSuccess != err)
    throw CudaException("countEqual: unable to allocate device memory", err);
  const size_t d_partial_stride = d_partial_pitch / sizeof(int);

  // Allocate final result
  int *d_count;
  err = cudaMalloc(&d_count, sizeof(int));
  if(cudaSuccess != err)
    throw CudaException("countEqual: unable to allocate device memory", err);

  reductionSumKernel<<<dim_grid, dim_block, sh_mem_size>>>(d_partial,
                                                           d_partial_stride,
                                                           in_img_data,
                                                           in_img_stride,
                                                           in_img_width,
                                                           in_img_height);

  reductionSumKernel<<<1, dim_block, sh_mem_size>>>(d_count,
                                                    0,
                                                    d_partial,
                                                    d_partial_stride,
                                                    dim_grid.x,
                                                    dim_grid.y);

  // download sum
  int h_count;
  err = cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  if(cudaSuccess != err)
    throw CudaException("sum: unable to copy result from device to host", err);

  // Free device memory
  err = cudaFree(d_count);
  if(cudaSuccess != err)
    throw CudaException("sum: unable to free device memory", err);
  err = cudaFree(d_partial);
  if(cudaSuccess != err)
    throw CudaException("sum: unable to free device memory", err);

  return h_count;
}

int rmd::sum(const rmd::DeviceImage<int> &in_img)
{
  return rmd::sum(in_img.data,
                  in_img.stride,
                  in_img.width,
                  in_img.height);
}

size_t rmd::countEqual(const DeviceImage<int> &in_img,
                       int value)
{
  // Allocate mask
  int *d_mask;
  size_t d_mask_pitch;
  cudaError err = cudaMallocPitch(
        &d_mask,
        &d_mask_pitch,
        in_img.width*sizeof(int),
        in_img.height);
  const size_t d_mask_stride = d_mask_pitch / sizeof(int);

  // Kernel configuration
  dim3 dim_block;
  dim3 dim_grid;
  dim_block.x = 16;
  dim_block.y = 16;
  dim_grid.x = (in_img.width  + dim_block.x - 1) / dim_block.x;
  dim_grid.y = (in_img.height + dim_block.y - 1) / dim_block.y;

  maskKernel<<<dim_grid, dim_block>>>(d_mask,
                                      d_mask_stride,
                                      in_img.data,
                                      in_img.stride,
                                      in_img.width,
                                      in_img.height,
                                      value);

  // Sum over mask
  int mask_sum = rmd::sum(d_mask,
                          d_mask_stride,
                          in_img.width,
                          in_img.height);

  // Free device memory
  err = cudaFree(d_mask);
  if(cudaSuccess != err)
    throw CudaException("countEqual: unable to free device memory", err);

  return static_cast<size_t>(mask_sum);
}
