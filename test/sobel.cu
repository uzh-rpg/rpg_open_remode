#include <rmd/device_image.cuh>

namespace rmd
{

__global__
void initKernel(
    DeviceImage<float> *in_dev_ptr,
    DeviceImage<float> *out_dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= in_dev_ptr->width || y >= in_dev_ptr->height)
    return;

  DeviceImage<float> &in_img  = *in_dev_ptr;
  DeviceImage<float> &out_img = *out_dev_ptr;
  out_img(x, y) = in_img(x, y);
}

void sobel(
    DeviceImage<float> &in_img,
    DeviceImage<float> &out_img)
{
  initKernel<<<16, 16>>>(in_img.dev_ptr, out_img.dev_ptr);
}

} // rmd namespace
