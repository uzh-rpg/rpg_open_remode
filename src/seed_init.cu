#ifndef RMD_SEED_INIT_CU
#define RMD_SEED_INIT_CU

#include <rmd/device_data.cuh>

namespace rmd
{

__global__
void seedInitKernel(DeviceData *dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x > dev_ptr->width || y > dev_ptr->height)
    return;

  dev_ptr->mu.data[y*dev_ptr->mu.stride+x] = dev_ptr->scene.avg_depth;
  dev_ptr->sigma.data[y*dev_ptr->sigma.stride+x] = dev_ptr->scene.sigma_sq_max;
  dev_ptr->a.data[y*dev_ptr->a.stride+x] = 10.0f;
  dev_ptr->b.data[y*dev_ptr->b.stride+x] = 10.0f;
}

} // rmd namespace

#endif
