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

#ifndef RMD_SEED_INIT_CU
#define RMD_SEED_INIT_CU

#include <rmd/mvs_device_data.cuh>
#include <rmd/texture_memory.cuh>

namespace rmd
{

__global__
void seedInitKernel(mvs::DeviceData *dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= dev_ptr->width || y >= dev_ptr->height)
    return;

  // Compute template statistics for NCC
  float sum_templ    = 0.0f;
  float sum_templ_sq = 0.0f;
  for(int patch_y=0; patch_y<RMD_CORR_PATCH_SIDE; ++patch_y)
  {
    for(int patch_x=0; patch_x<RMD_CORR_PATCH_SIDE; ++patch_x)
    {
      const float templ = tex2D(
            ref_img_tex,
            (float)(x+RMD_CORR_PATCH_OFFSET+patch_x)+0.5f,
            (float)(y+RMD_CORR_PATCH_OFFSET+patch_y)+0.5f);
      sum_templ += templ;
      sum_templ_sq += templ*templ;
    }
  }
  dev_ptr->sum_templ->atXY(x, y) = sum_templ;

  dev_ptr->const_templ_denom->atXY(x, y) =
      (float) ( (double) RMD_CORR_PATCH_AREA*sum_templ_sq - (double) sum_templ*sum_templ );

  // Init measurement parameters
  dev_ptr->mu->atXY(x, y) = dev_ptr->scene.avg_depth;
  dev_ptr->sigma->atXY(x, y) = dev_ptr->scene.sigma_sq_max;
  dev_ptr->a->atXY(x, y) = 10.0f;
  dev_ptr->b->atXY(x, y) = 10.0f;
}

} // rmd namespace

#endif
