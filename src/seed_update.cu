#ifndef RMD_SEED_UPDATE_CU
#define RMD_SEED_UPDATE_CU

#include <rmd/device_data.cuh>

namespace rmd
{

__global__
void seedUpdateKernel(DeviceData *dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= dev_ptr->width || y >= dev_ptr->height)
    return;

  const float xx = x+0.5f;
  const float yy = y+0.5f;

  // Check convergence status of seed
  if(ConvergenceStates::UPDATING != tex2D(convergence_tex, xx, yy))
    return;

  // Retrieve current estimations of parameters
  const float mu = tex2D(mu_tex, xx, yy);
  const float sigma_sq = tex2D(sigma_tex, xx, yy);
  const float a = tex2D(a_tex, xx, yy);
  const float b = tex2D(b_tex, xx, yy);

  // Bayesian update
  const float2 px_ref = make_float2((float)x, (float)y);
  const float3 f_ref = normalize(dev_ptr->cam.cam2world(px_ref));
#if 0
  float2 px;
  bool projection_within_image = true;
  if (findEpipolarMatch<ZMSSD>(
        px,
        projection_within_image,
        px_ref,
        cam_curr.world2cam(T_curr_ref * (f_ref*mu)), /* = px_curr_mean */
        cam_curr.world2cam(T_curr_ref * (f_ref*fmaxf(mu - 3.0f*sqrtf(sigma_sq), 0.01f))), /* = px_curr_min_d */
        cam_curr.world2cam(T_curr_ref * (f_ref*(mu + (3.0f*sqrtf(sigma_sq))))), /* = px_curr_max_d  */
        patch_edge_size,
        width, height
        ))
  {
    // Seed to be updated x/y with epi match px
    out_epipolar_matches[y*stride_32f2 + x] = px;
    out_converged[c32s] = 0;
  }
  else if (projection_within_image)
  {
    out_converged[c32s] = -1;
  }
  else
  {
    out_converged[c32s] = -2;
  }
#endif
}

} // rmd namespace

#endif
