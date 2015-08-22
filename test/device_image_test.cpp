#include <gtest/gtest.h>
#include <rmd/device_image.cuh>

TEST(RMDCuTests, deviceImageTest)
{
  const size_t w = 640;
  const size_t h = 480;

  rmd::DeviceImage<float> img(w, h);
}
