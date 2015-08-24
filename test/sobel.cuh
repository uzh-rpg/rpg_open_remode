#ifndef SOBEL_CUH
#define SOBEL_CUH

#include <rmd/device_image.cuh>

namespace rmd
{

void sobel(
    const DeviceImage<float> &in_img,
    DeviceImage<float2> &out_grad);

} // rmd namespace

#endif // SOBEL_CUH
