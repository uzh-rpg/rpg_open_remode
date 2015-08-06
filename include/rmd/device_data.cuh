#ifndef RMD_DEVICE_DATA_CUH
#define RMD_DEVICE_DATA_CUH

#include <rmd/pinhole_camera.cuh>
#include <rmd/image.cuh>

namespace rmd
{

struct DeviceSceneData
{
  float min_depth;
  float max_depth;
  float avg_depth;
  float depth_range;
  float sigma_sq_max;
};

template<typename ElementType>
struct Device2DData
{
  void set(
      ElementType *data,
      const size_t &pitch,
      const size_t &stride)
  {
    this->data   = data;
    this->pitch  = pitch;
    this->stride = stride;
  }
  void set(const Image<ElementType> &img)
  {
    set(
          img.getDevDataPtr(),
          img.getPitch(),
          img.getStride()
          );
  }
  ElementType *data;
  size_t pitch;
  size_t stride;
};

struct DeviceData
{
  __host__
  DeviceData()
  {
    // Allocate device memory
    const cudaError err = cudaMalloc(&dev_ptr, sizeof(*this));
    if(err != cudaSuccess)
      throw CudaException("DeviceData, cannot allocate device memory to store image parameters.", err);
  }
  __host__
  ~DeviceData()
  {
    cudaFree(dev_ptr);
  }
  void setDevData()
  {
    // Copy data to device memory
    const cudaError err = cudaMemcpy(dev_ptr, this, sizeof(*this), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
      throw CudaException("DeviceData, cannot copy image parameters to device memory.", err);
  }

  Device2DData<float> ref_img;
  Device2DData<float> curr_img;
  Device2DData<float> mu;
  Device2DData<float> sigma;
  Device2DData<float> a;
  Device2DData<float> b;

  PinholeCamera cam;
  float one_pix_angle;
  size_t width;
  size_t height;

  DeviceSceneData scene;

  DeviceData *dev_ptr;
};

} // rmd namespace

#endif // RMD_DEVICE_DATA_CUH
