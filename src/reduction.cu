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

namespace rmd
{

__global__
void countEqualKernel(
    int *out_dev_ptr,
    const int *in_dev_ptr,
    int value)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  extern __shared__ int s_partial[];
  int count = 0;


  //in_dev_ptr->atXY(x, y)
}

} // rmd namespace

size_t rmd::countEqual(
    const rmd::DeviceImage<int> &in_img,
    int value)
{
  // Kernel configuration
  dim3 dim_block;
  dim3 dim_grid;
  dim_block.x = 16; // Num threads
  dim_block.y = 16;
  dim_grid.x = 4;   // Num blocks
  dim_grid.y = 4;
  // Compute required shared memory
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

  // countEqualKernel<<<dim_grid, dim_block, sh_mem_size>>>(d_partial, in_img.dev_ptr, value);

  int h_count;
  err = cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  if(cudaSuccess != err)
    throw CudaException("countEqual: unable to copy result from device to host", err);
  err = cudaFree(d_count);
  if(cudaSuccess != err)
    throw CudaException("countEqual: unable to free device memory", err);
  err = cudaFree(d_partial);
  if(cudaSuccess != err)
    throw CudaException("countEqual: unable to free device memory", err);
  return static_cast<size_t>(h_count);
}
