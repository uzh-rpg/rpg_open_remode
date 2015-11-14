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

#ifndef RMD_EPIPOLAR_MATCH_CU
#define RMD_EPIPOLAR_MATCH_CU

#include <float.h>
#include <rmd/se3.cuh>
#include <rmd/seed_matrix.cuh>
#include <rmd/mvs_device_data.cuh>
#include <rmd/texture_memory.cuh>
#include <rmd/helper_vector_types.cuh>

namespace rmd
{

__constant__ Size c_img_size;
extern "C" void copyImgSzToConst(rmd::Size *h_img_size)
{
  cudaMemcpyToSymbol(c_img_size, h_img_size, sizeof(rmd::Size));
}

__global__
void seedEpipolarMatchKernel(
    mvs::DeviceData *dev_ptr,
    SE3<float> T_curr_ref)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= c_img_size.width || y >= c_img_size.height)
    return;

  const float xx = x+0.5f;
  const float yy = y+0.5f;

  const int seed_state = tex2D(convergence_tex, xx, yy);
  if( (ConvergenceStates::BORDER    == seed_state) ||
      (ConvergenceStates::CONVERGED == seed_state) ||
      (ConvergenceStates::DIVERGED  == seed_state) )
  {
    return;
  }

  // Retrieve current estimations of depth
  const float mu = tex2D(mu_tex, xx, yy);
  const float sigma = sqrtf(tex2D(sigma_tex, xx, yy));

  const float2 px_ref = make_float2((float)x, (float)y);
  const float3 f_ref = normalize(dev_ptr->cam.cam2world(px_ref));
  const float2 px_mean_curr =
      dev_ptr->cam.world2cam( T_curr_ref * (f_ref * mu) );

  const float2 px_min_curr =
      dev_ptr->cam.world2cam( T_curr_ref * (f_ref * fmaxf( mu - 3.0f*sigma, 0.01f)) );
  const float2 px_max_curr =
      dev_ptr->cam.world2cam( T_curr_ref * (f_ref * ( mu + (3.0f*sigma) ) ) );

  const float2 epi_line = px_max_curr - px_min_curr;
  const float2 epi_dir  = normalize(epi_line);
  const float  half_length = 0.5f * fminf(norm(epi_line), RMD_MAX_EXTENT_EPIPOLAR_SEARCH);
  float2 px_curr, best_px_curr;

  // Retrieve template statistics for NCC matching;
  const float sum_templ = tex2D(sum_templ_tex, xx, yy);
  const float const_templ_denom = tex2D(const_templ_denom_tex, xx, yy);

  // init best match score
  float best_ncc = -1.0f;

  float sum_img;
  float sum_img_sq;
  float sum_img_templ;
  for(float l = -half_length; l <= half_length; l += 0.7f)
  {
    px_curr = px_mean_curr + l*epi_dir;
    if( (px_curr.x >= c_img_size.width - RMD_CORR_PATCH_SIDE)  ||
        (px_curr.y >= c_img_size.height - RMD_CORR_PATCH_SIDE) ||
        (px_curr.x < RMD_CORR_PATCH_SIDE)                    ||
        (px_curr.y < RMD_CORR_PATCH_SIDE) )
    {
      continue;
    }

    sum_img       = 0.0f;
    sum_img_sq    = 0.0f;
    sum_img_templ = 0.0f;

    for(int patch_y=0; patch_y<RMD_CORR_PATCH_SIDE; ++patch_y)
    {
      for(int patch_x=0; patch_x<RMD_CORR_PATCH_SIDE; ++patch_x)
      {
        const float templ = tex2D(
              ref_img_tex,
              px_ref.x+(float)(RMD_CORR_PATCH_OFFSET+patch_x)+0.5f,
              px_ref.y+(float)(RMD_CORR_PATCH_OFFSET+patch_y)+0.5f);
        const float img = tex2D(
              curr_img_tex,
              px_curr.x+(float)(RMD_CORR_PATCH_OFFSET+patch_x)+0.5f,
              px_curr.y+(float)(RMD_CORR_PATCH_OFFSET+patch_y)+0.5f);
        sum_img    += img;
        sum_img_sq += img*img;
        sum_img_templ += img*templ;
      }
    }
    const float ncc_numerator = RMD_CORR_PATCH_AREA*sum_img_templ - sum_img*sum_templ;
    const float ncc_denominator = (RMD_CORR_PATCH_AREA*sum_img_sq - sum_img*sum_img)*const_templ_denom;

    const float ncc = ncc_numerator * rsqrtf(ncc_denominator + FLT_MIN);

    if(ncc > best_ncc)
    {
      best_px_curr = px_curr;
      best_ncc = ncc;
    }
  }
  if(best_ncc < 0.5f)
  {
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::NO_MATCH;
  }
  else
  {
    dev_ptr->epipolar_matches->atXY(x, y) = best_px_curr;
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::UPDATE;
  }
}

} // rmd namespace

#endif
