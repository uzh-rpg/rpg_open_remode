#ifndef RMD_TEST_SOBEL_CUH
#define RMD_TEST_SOBEL_CUH

#include <rmd/device_image.cuh>

namespace rmd
{

void sobel(
    const DeviceImage<float> &in_img,
    DeviceImage<float2> &out_grad);

} // rmd namespace

#endif // RMD_TEST_SOBEL_CUH
