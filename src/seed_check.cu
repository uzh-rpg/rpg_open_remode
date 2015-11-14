// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RMD_SEED_CHECK_CU
#define RMD_SEED_CHECK_CU

#include <rmd/mvs_device_data.cuh>
#include <rmd/texture_memory.cuh>
#include <rmd/seed_matrix.cuh>

namespace rmd
{

__global__
void seedCheckKernel(mvs::DeviceData *dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= dev_ptr->width || y >= dev_ptr->height)
    return;

  if(x > dev_ptr->width-RMD_CORR_PATCH_SIDE-1 || y > dev_ptr->height-RMD_CORR_PATCH_SIDE-1 ||
     x < RMD_CORR_PATCH_SIDE || y < RMD_CORR_PATCH_SIDE)
  {
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::BORDER;
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
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::CONVERGED;
  }
  else if((a-1) / (a + b - 2) < dev_ptr->eta_outlier)
  { // The seed failed to converge
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::DIVERGED;
  }
  else
  {
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::UPDATE;
  }
}

} // rmd namespace

#endif
