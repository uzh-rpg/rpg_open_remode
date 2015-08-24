#include <rmd/device_image.cuh>

namespace rmd
{

__global__
void sobelKernel(
    const DeviceImage<float> *in_dev_ptr,
    DeviceImage<float2> *out_dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= in_dev_ptr->width-1
     || y >= in_dev_ptr->height-1
     || x < 1
     || y < 1)
    return;

  const DeviceImage<float>  &in_img = *in_dev_ptr;
  const float ul = in_img(x-1, y-1);
  const float um = in_img(x,   y-1);
  const float ur = in_img(x+1, y-1);
  const float ml = in_img(x-1, y);
  const float mr = in_img(x+1, y);
  const float ll = in_img(x-1, y+1);
  const float lm = in_img(x,   y+1);
  const float lr = in_img(x+1, y+1);

  DeviceImage<float2> &out_grad = *out_dev_ptr;
  // Sobel operator
  //dx_out(x, y) = 1.0f*ul + 2.0f*ml + 1.0f*ll - 1.0f*ur - 2.0f*mr -1.0f*lr;
  //dy_out(x, y) = 1.0f*ul + 2.0f*um + 1.0f*ur -1.0f*ll - 2.0f*lm -1.0f*lr;
  // Use the Scharr operator for comparison with OpenCV
  out_grad(x, y).x = -3.0f*ul - 10.0f*ml - 3.0f*ll + 3.0f*ur + 10.0f*mr + 3.0f*lr;
  out_grad(x, y).y = -3.0f*ul - 10.0f*um - 3.0f*ur + 3.0f*ll + 10.0f*lm + 3.0f*lr;
}

void sobel(
    const DeviceImage<float> &in_img,
    DeviceImage<float2> &out_grad)
{
  sobelKernel<<<16, 16>>>(in_img.dev_ptr, out_grad.dev_ptr);
}

} // rmd namespace
