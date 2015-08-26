#ifndef RMD_SEED_CHECK_CU
#define RMD_SEED_CHECK_CU

#include <rmd/device_data.cuh>
#include <rmd/texture_memory.cuh>
#include <rmd/seed_matrix.cuh>

namespace rmd
{

__global__
void seedCheckKernel(DeviceData *dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= dev_ptr->width || y >= dev_ptr->height)
    return;

  if(x > dev_ptr->width-dev_ptr->patch.side-1 || y > dev_ptr->height-dev_ptr->patch.side-1 ||
     x < dev_ptr->patch.side || y < dev_ptr->patch.side)
  {
    dev_ptr->convergence->at(x, y) = ConvergenceStates::BORDER;
    return;
  }

  const float xx = x+0.5f;
  const float yy = y+0.5f;

  // Retrieve current estimations of parameters
  const float mu = tex2D(mu_tex, xx, yy);
  const float sigma_sq = tex2D(sigma_tex, xx, yy);
  const float a = tex2D(a_tex, xx, yy);
  const float b = tex2D(b_tex, xx, yy);

  // if E(inlier_ratio) > eta_inlier && sigma_sq < epsilon
  if( ((a / (a + b)) > dev_ptr->eta_inlier)
      && (sigma_sq < dev_ptr->epsilon) )
  { // The seed converged
    dev_ptr->convergence->at(x, y) = ConvergenceStates::CONVERGED;
  }
  else if((a-1) / (a + b - 2) < dev_ptr->eta_outlier)
  { // The seed failed to converge
    dev_ptr->convergence->at(x, y) = ConvergenceStates::DIVERGED;
  }
  else
  {
    dev_ptr->convergence->at(x, y) = ConvergenceStates::UPDATE;
  }
}

} // rmd namespace

#endif
