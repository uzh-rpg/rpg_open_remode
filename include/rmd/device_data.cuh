#ifndef RMD_DEVICE_DATA_CUH
#define RMD_DEVICE_DATA_CUH

#include <rmd/pinhole_camera.cuh>

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
  ElementType  *data;
  size_t pitch;
  size_t stride;
};

struct DeviceData
{
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
};

} // rmd namespace

#endif // RMD_DEVICE_DATA_CUH
