#ifndef RMD_DEVICE_DATA_CUH
#define RMD_DEVICE_DATA_CUH

#include <rmd/pinhole_camera.cuh>
#include <rmd/device_image.cuh>

namespace rmd
{

struct SceneData
{
  float min_depth;
  float max_depth;
  float avg_depth;
  float depth_range;
  float sigma_sq_max;
};

struct TemplatePatch
{
#ifndef RMD_TEMPLATE_PATCH_SIDE
#define RMD_TEMPLATE_PATCH_SIDE 5
#endif
  TemplatePatch()
    : side(RMD_TEMPLATE_PATCH_SIDE)
    , offset(make_int2(-side/2, -side/2))
  { }
  const int  side;
  const int2 offset;
};

// DeviceData struct stores pointers to dev memory.
// It is allocated and set from host.
struct DeviceData
{
  __host__
  DeviceData()
    : is_dev_allocated(false)
  {
  }
  __host__
  ~DeviceData()
  {
    if(is_dev_allocated)
      cudaFree(dev_ptr);
  }
  __host__
  void setDevData()
  {
    if(!is_dev_allocated)
    {
      // Allocate device memory
      const cudaError err = cudaMalloc(&dev_ptr, sizeof(*this));
      if(err != cudaSuccess)
        throw CudaException("DeviceData, cannot allocate device memory to store image parameters.", err);
      else
      {
        is_dev_allocated = true;
      }
    }
    // Copy data to device memory
    const cudaError err = cudaMemcpy(dev_ptr, this, sizeof(*this), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
      throw CudaException("DeviceData, cannot copy image parameters to device memory.", err);
  }


  DeviceImage<float> *ref_img;
  DeviceImage<float> *curr_img;
  DeviceImage<float> *sum_templ;
  DeviceImage<float> *const_templ_denom;
  DeviceImage<float> *mu;
  DeviceImage<float> *sigma;
  DeviceImage<float> *a;
  DeviceImage<float> *b;
  DeviceImage<unsigned char> *convergence;
  DeviceImage<float2> *epipolar_matches;

  PinholeCamera cam;
  float one_pix_angle;
  size_t width;
  size_t height;

  SceneData scene;
  TemplatePatch patch;

  // Algorithm parameters
  float eta_inlier;
  float eta_outlier;
  float epsilon;

  DeviceData *dev_ptr;
  bool is_dev_allocated;
};

} // rmd namespace

#endif // RMD_DEVICE_DATA_CUH
