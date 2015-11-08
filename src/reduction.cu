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
#include "reduction_kernels.cu"

template<typename T>
rmd::ImageReducer<T>::ImageReducer(dim3 num_threads_per_block,
                                   dim3 num_blocks_per_grid)
  : block_dim_(num_threads_per_block)
  , grid_dim_(num_blocks_per_grid)
  , is_dev_part_alloc_(false)
  , is_dev_fin_alloc_(false)
{
  // Compute required amount of shared memory
  sh_mem_size_ = block_dim_.x * block_dim_.y * sizeof(T);

  // Allocate intermediate result
  const cudaError part_alloc_err = cudaMallocPitch(
        &dev_partial_,
        &dev_partial_pitch_,
        grid_dim_.x*sizeof(T),
        grid_dim_.y);
  if(cudaSuccess != part_alloc_err)
  {
    throw CudaException("ImageReducer: unable to allocate pitched device memory for partial results", part_alloc_err);
  }
  else
  {
    dev_partial_stride_ = dev_partial_pitch_ / sizeof(T);
    is_dev_part_alloc_ = true;
  }

  // Allocate final result
  const cudaError fin_alloc_err = cudaMalloc(&dev_final_, sizeof(T));
  if(cudaSuccess != fin_alloc_err)
  {
    throw CudaException("ImageReducer: unable to allocate device memory for final result", fin_alloc_err);
  }
  else
  {
    is_dev_fin_alloc_ = true;
  }
}

template<typename T>
rmd::ImageReducer<T>::~ImageReducer()
{
  // Free device memory
  if(is_dev_fin_alloc_)
  {
    const cudaError err = cudaFree(dev_final_);
    if(cudaSuccess != err)
      throw CudaException("ImageReducer: unable to free device memory", err);
  }
  if(is_dev_part_alloc_)
  {
    const cudaError err = cudaFree(dev_partial_);
    if(cudaSuccess != err)
      throw CudaException("ImageReducer: unable to free device memory", err);
  }
}

// Sum image by reduction
// Cfr. listing 12.1 by N. Wilt, "The CUDA Handbook"
template<typename T>
T rmd::ImageReducer<T>::sum(const T *in_img_data,
                            size_t in_img_stride,
                            size_t in_img_width,
                            size_t in_img_height)
{
  if(is_dev_fin_alloc_ && is_dev_part_alloc_)
  {
    reductionSumKernel<T>
        <<<grid_dim_, block_dim_, sh_mem_size_>>>
                                                (dev_partial_,
                                                 dev_partial_stride_,
                                                 in_img_data,
                                                 in_img_stride,
                                                 in_img_width,
                                                 in_img_height);

    reductionSumKernel<T>
        <<<1, block_dim_, sh_mem_size_>>>
                                        (dev_final_,
                                         0,
                                         dev_partial_,
                                         dev_partial_stride_,
                                         grid_dim_.x,
                                         grid_dim_.y);

    // download sum
    T h_count;
    const cudaError err = cudaMemcpy(&h_count, dev_final_, sizeof(T), cudaMemcpyDeviceToHost);
    if(cudaSuccess != err)
      throw CudaException("sum: unable to copy result from device to host", err);

    return h_count;
  }
  else
  {
    return 0;
  }
}

template<typename T>
T rmd::ImageReducer<T>::sum(const DeviceImage<T> &in_img)
{
  return rmd::ImageReducer<T>::sum(in_img.data,
                                   in_img.stride,
                                   in_img.width,
                                   in_img.height);
}


// Count elements equal to 'value'
// First count over the thread grid,
// then perform a reduction sum on a single thread block
template<>
size_t rmd::ImageReducer<int>::countEqual(const int *in_img_data,
                                          size_t in_img_stride,
                                          size_t in_img_width,
                                          size_t in_img_height,
                                          int value)
{
  if(is_dev_fin_alloc_ && is_dev_part_alloc_)
  {
    reductionCountEqKernel<int>
        <<<grid_dim_, block_dim_, sh_mem_size_>>>
                                                (dev_partial_,
                                                 dev_partial_stride_,
                                                 in_img_data,
                                                 in_img_stride,
                                                 in_img_width,
                                                 in_img_height,
                                                 value);

    reductionSumKernel<int>
        <<<1, block_dim_, sh_mem_size_>>>
                                        (dev_final_,
                                         0,
                                         dev_partial_,
                                         dev_partial_stride_,
                                         grid_dim_.x,
                                         grid_dim_.y);

    // download sum
    int h_count;
    const cudaError err = cudaMemcpy(&h_count, dev_final_, sizeof(int), cudaMemcpyDeviceToHost);
    if(cudaSuccess != err)
      throw CudaException("countEqual: unable to copy result from device to host", err);

    return static_cast<size_t>(h_count);
  }
  else
  {
    return 0;
  }
}

template<>
size_t rmd::ImageReducer<int>::countEqual(const DeviceImage<int> &in_img,
                                          int value)
{
  return rmd::ImageReducer<int>::countEqual(in_img.data,
                                            in_img.stride,
                                            in_img.width,
                                            in_img.height,
                                            value);
}

template class rmd::ImageReducer<int>;
template class rmd::ImageReducer<float>;
