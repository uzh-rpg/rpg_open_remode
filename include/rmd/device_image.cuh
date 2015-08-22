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
    const cudaError err = cudaMallocPitch(
          &data,
          &pitch,
          width*sizeof(ElementType),
          height);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to allocate pitched memory.", err);
    stride = pitch / sizeof(ElementType);
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
