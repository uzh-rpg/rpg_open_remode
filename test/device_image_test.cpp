#include <gtest/gtest.h>
#include <rmd/device_image.cuh>
namespace rmd
{
void sobel(DeviceImage<float> &in_img, DeviceImage<float> &out_img);
}

TEST(RMDCuTests, deviceImageSobelTest)
{
  const size_t w = 640;
  const size_t h = 480;

  rmd::DeviceImage<float> in_img(w, h);
  rmd::DeviceImage<float> out_img(w, h);

  rmd::sobel(in_img, out_img);
}
