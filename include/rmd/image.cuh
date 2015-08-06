#ifndef RMD_IMAGE_CUH_
#define RMD_IMAGE_CUH_

#include <cassert>
#include <cuda_runtime.h>
#include <rmd/cuda_exception.cuh>

namespace rmd
{

template<typename ElementType>
struct Image
{

  __host__ Image(size_t width, size_t height)
    : width(width),
      height(height),
      is_externally_allocated(false)
  {
    const cudaError err = cudaMallocPitch(&dev_data,
                                          &pitch,
                                          width*sizeof(ElementType),
                                          height);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to allocate pitched memory.", err);
    stride = pitch / sizeof(ElementType);
    channel_format_desc = cudaCreateChannelDesc<ElementType>();
  }

  __host__ __device__ Image(const Image<ElementType>& src_img)
    : width(src_img.getWidth())
    , height(src_img.getHeight())
    , pitch(src_img.getPitch())
    , stride(src_img.getStride())
    , dev_data(src_img.getDevDataPtr())
    , channel_format_desc(src_img.getChannelFormatDesc())
    , is_externally_allocated(true)
  {
  }

  __host__ ~Image()
  {
    if(!is_externally_allocated)
    {
      const cudaError err = cudaFree(dev_data);
      if(err != cudaSuccess)
        throw CudaException("Image: unable to free allocated memory.", err);
    }
  }

  /// Set all the device data to zero
  __host__ void zero()
  {
    const cudaError err = cudaMemset2D(dev_data, pitch, 0,
                                       width*sizeof(ElementType), height);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to zero.", err);
  }

  /// Set the device data to the values in the array
  __host__ void setDevData(const ElementType * aligned_data_row_major)
  {
    const cudaError err = cudaMemcpy2D(dev_data, pitch,
                                       aligned_data_row_major,
                                       width*sizeof(ElementType),
                                       width*sizeof(ElementType),
                                       height,
                                       cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to copy data from host to device.", err);
  }

  /// Download the data from the device data to a preallocated array on host
  __host__ void getDevData(ElementType* aligned_data_row_major)
  {
    const cudaError err = cudaMemcpy2D(aligned_data_row_major,       // destination memory address
                                       width*sizeof(ElementType),   // pitch of destination memory
                                       dev_data,                    // source memory address
                                       pitch,                       // pitch of source memory
                                       width*sizeof(ElementType),   // width of matrix transfor (columns in bytes)
                                       height,                      // height of matrix transfer
                                       cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to copy data from device to host.", err);
  }

  /// Copy assignment operator
  /// fill this image with content from another image
  __host__ Image<ElementType>& operator=(const Image<ElementType>& other_image)
  {
    if(&other_image != this)
    {
      assert(this->getWidth()  == other_image.getWidth() &&
             this->getHeight() == other_image.getHeight());
      const cudaError err = cudaMemcpy2D(getDevDataPtr(),
                                         getPitch(),
                                         other_image.getDevDataPtr(),
                                         other_image.getPitch(),
                                         width*sizeof(ElementType),
                                         height,
                                         cudaMemcpyDeviceToDevice);
      if(err != cudaSuccess)
        throw CudaException("Image, operator '=': unable to copy data from another image.", err);
    }
    return *this;
  }

  __host__ __device__ __forceinline__ size_t getWidth()  const { return width;  }

  __host__ __device__ __forceinline__ size_t getHeight() const { return height; }

  __host__ __device__ __forceinline__ size_t getPitch()  const { return pitch;  }

  __host__ __device__ __forceinline__ size_t getStride() const { return stride; }

  __host__ __device__ __forceinline__ cudaChannelFormatDesc getChannelFormatDesc() const { return channel_format_desc; }

  /// return ptr to dev data
  __host__ __device__ __forceinline__ ElementType * getDevDataPtr() const { return dev_data; }

  __device__ __forceinline__ ElementType & operator()(int x, int y)
  {
    return dev_data[stride*y+x];
  }

  // fields
  size_t width;
  size_t height;
  size_t pitch;
  size_t stride;
  ElementType * dev_data;
  bool is_externally_allocated;
  cudaChannelFormatDesc channel_format_desc;
};

} // namespace rmd

#endif /* RMD_IMAGE_CUH_ */
