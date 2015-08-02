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

struct Device2DData
{
  float  *data;
  size_t pitch;
  size_t stride;
};

struct DeviceData
{
  Device2DData ref_img;
  Device2DData curr_img;
  Device2DData mu;
  Device2DData sigma;
  Device2DData a;
  Device2DData b;

  PinholeCamera cam;
  size_t width;
  size_t height;

  DeviceSceneData scene;
};

} // rmd namespace

#endif // RMD_DEVICE_DATA_CUH
