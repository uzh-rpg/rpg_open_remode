#include <rmd/device_image.cuh>

namespace rmd
{

__global__
void copyKernel(
    const DeviceImage<float> *in_dev_ptr,
    DeviceImage<float> *out_dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= in_dev_ptr->width || y >= in_dev_ptr->height)
    return;

  const DeviceImage<float>  &img = *in_dev_ptr;
  DeviceImage<float> &copy = *out_dev_ptr;
  copy(x, y) = img(x, y);
}

void copy(
    const DeviceImage<float> &img,
    DeviceImage<float> &copy)
{
  // CUDA fields
  dim3 dim_block;
  dim3 dim_grid;
  dim_block.x = 16;
  dim_block.y = 16;
  dim_grid.x = (img.width  + dim_block.x - 1) / dim_block.x;
  dim_grid.y = (img.height + dim_block.y - 1) / dim_block.y;
  copyKernel<<<dim_grid, dim_block>>>(img.dev_ptr, copy.dev_ptr);
}

} // rmd namespace

