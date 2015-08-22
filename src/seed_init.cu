#ifndef RMD_SEED_INIT_CU
#define RMD_SEED_INIT_CU

#include <rmd/device_data.cuh>
#include <rmd/texture_memory.cuh>

namespace rmd
{

__global__
void seedInitKernel(DeviceData *dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= dev_ptr->width || y >= dev_ptr->height)
    return;

  // Compute template statistics for NCC
  float sum_templ    = 0.0f;
  float sum_templ_sq = 0.0f;
  const int  &side   = dev_ptr->patch.side;
  const int2 &offset = dev_ptr->patch.offset;
  for(int patch_y=0; patch_y<side; ++patch_y)
  {
    for(int patch_x=0; patch_x<side; ++patch_x)
    {
      const float templ = tex2D(
            ref_img_tex,
            (float)(x+offset.x+patch_x)+0.5f,
            (float)(y+offset.y+patch_y)+0.5f
            );
      sum_templ += templ;
      sum_templ_sq += templ*templ;
    }
  }
  dev_ptr->sum_templ.data[y*dev_ptr->sum_templ.stride+x] = sum_templ;
  dev_ptr->const_templ_denom.data[y*dev_ptr->const_templ_denom.stride+x] =
      (float) ( (double) side*side*sum_templ_sq - (double) sum_templ*sum_templ );

  // Init measurement parameters
  dev_ptr->mu.data[y*dev_ptr->mu.stride+x] = dev_ptr->scene.avg_depth;
  dev_ptr->sigma.data[y*dev_ptr->sigma.stride+x] = dev_ptr->scene.sigma_sq_max;
  dev_ptr->a.data[y*dev_ptr->a.stride+x] = 10.0f;
  dev_ptr->b.data[y*dev_ptr->b.stride+x] = 10.0f;
}

} // rmd namespace

#endif
