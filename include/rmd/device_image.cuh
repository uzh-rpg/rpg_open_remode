#ifndef DEVICE_IMAGE_CUH
#define DEVICE_IMAGE_CUH

#include <cuda_runtime.h>
#include <rmd/cuda_exception.cuh>

namespace rmd
{

template<typename ElementType>
struct DeviceImage
{
  __host__
  DeviceImage(size_t width, size_t height)
    : width(width),
      height(height)
  {
    cudaError err = cudaMallocPitch(
          &data,
          &pitch,
          width*sizeof(ElementType),
          height);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to allocate pitched memory.", err);
    stride = pitch / sizeof(ElementType);

    err = cudaMalloc(
          &dev_ptr,
          sizeof(*this));
    if(err != cudaSuccess)
      throw CudaException("DeviceData, cannot allocate device memory to store image parameters.", err);

    err = cudaMemcpy(
          dev_ptr,
          this,
          sizeof(*this),
          cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
      throw CudaException("DeviceData, cannot copy image parameters to device memory.", err);
  }

  __device__
  ElementType & operator()(size_t x, size_t y)
  {
    return data[y*stride+x];
  }

  __device__
  const ElementType & operator()(size_t x, size_t y) const
  {
    return data[y*stride+x];
  }

  /// Upload aligned_data_row_major to device memory
  __host__
  void setDevData(const ElementType * aligned_data_row_major)
  {
    const cudaError err = cudaMemcpy2D(
          data,
          pitch,
          aligned_data_row_major,
          width*sizeof(ElementType),
          width*sizeof(ElementType),
          height,
          cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to copy data from host to device.", err);
  }

  /// Download the data from the device memory to aligned_data_row_major, a preallocated array in host memory
  __host__
  void getDevData(ElementType* aligned_data_row_major) const
  {
    const cudaError err = cudaMemcpy2D(
          aligned_data_row_major,      // destination memory address
          width*sizeof(ElementType),   // pitch of destination memory
          data,                        // source memory address
          pitch,                       // pitch of source memory
          width*sizeof(ElementType),   // width of matrix transfor (columns in bytes)
          height,                      // height of matrix transfer
          cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to copy data from device to host.", err);
  }

  __host__
  ~DeviceImage()
  {
    const cudaError err = cudaFree(data);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to free allocated memory.", err);
  }

  // fields
  size_t width;
  size_t height;
  size_t pitch;
  size_t stride;
  ElementType * data;
  DeviceImage<ElementType> *dev_ptr;
};

} // namespace rmd

#endif // DEVICE_IMAGE_CUH
