#ifndef RMD_TEST_COPY_CUH
#define RMD_TEST_COPY_CUH

#include <rmd/device_image.cuh>

namespace rmd
{

void copy(
    const DeviceImage<float> &img,
    DeviceImage<float> &copy);

} // rmd namespace

#endif // RMD_TEST_COPY_CUH
