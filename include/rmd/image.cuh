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
    : width_(width),
      height_(height),
      is_externally_allocated_(false)
  {
    const cudaError err = cudaMallocPitch(&dev_data_,
                                          &pitch_,
                                          width_*sizeof(ElementType),
                                          height_);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to allocate pitched memory.", err);
    stride_ = pitch_ / sizeof(ElementType);
    channel_format_desc_ = cudaCreateChannelDesc<ElementType>();
  }

  __host__ __device__ Image(const Image<ElementType>& src_img)
    : width_(src_img.getWidth())
    , height_(src_img.getHeight())
    , pitch_(src_img.getPitch())
    , stride_(src_img.getStride())
    , dev_data_(src_img.getDevDataPtr())
    , channel_format_desc_(src_img.getChannelFormatDesc())
    , is_externally_allocated_(true)
  {
  }

  __host__ ~Image()
  {
    if(!is_externally_allocated_)
    {
      const cudaError err = cudaFree(dev_data_);
      if(err != cudaSuccess)
        throw CudaException("Image: unable to free allocated memory.", err);
    }
  }

  /// Set all the device data to zero
  __host__ void zero()
  {
    const cudaError err = cudaMemset2D(dev_data_, pitch_, 0,
                                       width_*sizeof(ElementType), height_);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to zero.", err);
  }

  /// Set the device data to the values in the array
  __host__ void setDevData(const ElementType * aligned_data_row_major)
  {
    const cudaError err = cudaMemcpy2D(dev_data_, pitch_,
                                       aligned_data_row_major,
                                       width_*sizeof(ElementType),
                                       width_*sizeof(ElementType),
                                       height_,
                                       cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to copy data from host to device.", err);
  }

  /// Download the data from the device data to a preallocated array on host
  __host__ void getDevData(ElementType* aligned_data_row_major)
  {
    const cudaError err = cudaMemcpy2D(aligned_data_row_major,       // destination memory address
                                       width_*sizeof(ElementType),   // pitch of destination memory
                                       dev_data_,                    // source memory address
                                       pitch_,                       // pitch of source memory
                                       width_*sizeof(ElementType),   // width of matrix transfor (columns in bytes)
                                       height_,                      // height of matrix transfer
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
      const cudaError err = cudaMemcpy2D(this->getDevDataPtr(),
                                         this->getPitch(),
                                         other_image.getDevDataPtr(),
                                         other_image.getPitch(),
                                         width_*sizeof(ElementType),
                                         height_,
                                         cudaMemcpyDeviceToDevice);
      if(err != cudaSuccess)
        throw CudaException("Image, operator '=': unable to copy data from another image.", err);
    }
    return *this;
  }

  __host__ __device__ __forceinline__ size_t getWidth()  const { return width_;  }

  __host__ __device__ __forceinline__ size_t getHeight() const { return height_; }

  __host__ __device__ __forceinline__ size_t getPitch()  const { return pitch_;  }

  __host__ __device__ __forceinline__ size_t getStride() const { return stride_; }

  __host__ __device__ __forceinline__ cudaChannelFormatDesc getChannelFormatDesc() const { return channel_format_desc_; }

  /// return ptr to dev data
  __host__ __device__ __forceinline__ ElementType * getDevDataPtr() const { return dev_data_; }

  __device__ __forceinline__ ElementType & operator()(int x, int y)
  {
    return dev_data_[stride_*y+x];
  }

  // fields
  size_t width_;
  size_t height_;
  size_t pitch_;
  size_t stride_;
  ElementType * dev_data_;
  bool is_externally_allocated_;
  cudaChannelFormatDesc channel_format_desc_;
};

} // namespace rmd

#endif /* RMD_IMAGE_CUH_ */
