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

namespace rmd
{

// From Reduction SDK sample:
// Prevent instantiation of the generic struct using an undefined symbol
// in the function body (so it won't compile)
template<typename T>
struct SharedMemory
{
  __device__ T *getPointer()
  {
    extern __device__ void error(void);
    error();
    return NULL;
  }
};

// Required specializations
template<>
struct SharedMemory<int>
{
  __device__ int *getPointer()
  {
    extern __shared__ int s_int[];
    return s_int;
  }
};

template<>
struct SharedMemory<float>
{
  __device__ float *getPointer()
  {
    extern __shared__ float s_float[];
    return s_float;
  }
};

// Templated kernels
template<typename T>
__global__
void reductionSumKernel(T *out_dev_ptr,
                        size_t out_stride,
                        const T *in_dev_ptr,
                        size_t in_stride,
                        size_t n,
                        size_t m)
{
  SharedMemory<T> smem;
  T *s_partial = smem.getPointer();

  T sum = 0;

  // Sum over 2D thread grid, use (x,y) indices
  for(int x = blockIdx.x * blockDim.x + threadIdx.x;
      x < n;
      x += blockDim.x*gridDim.x)
  {
    for(int y = blockIdx.y * blockDim.y + threadIdx.y;
        y < m;
        y += blockDim.y*gridDim.y)
    {
      sum += in_dev_ptr[y*in_stride+x];
    }
  }
  // Sums are written to shared memory, single index
  s_partial[threadIdx.y*blockDim.x+threadIdx.x] = sum;
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

template<typename T>
__global__
void reductionCountEqKernel(int *out_dev_ptr,
                            size_t out_stride,
                            const T *in_dev_ptr,
                            size_t in_stride,
                            size_t n,
                            size_t m,
                            T value)
{
  SharedMemory<int> smem;
  int *s_partial = smem.getPointer();

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
      if(value == in_dev_ptr[y*in_stride+x])
      {
        count += 1;
      }
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
    if (tid < num_active_threads)
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

} // rmd namespace
