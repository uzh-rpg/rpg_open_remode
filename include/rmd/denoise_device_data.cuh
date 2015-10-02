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

#ifndef RMD_DENOISE_DEVICE_DATA_CUH
#define RMD_DENOISE_DEVICE_DATA_CUH

#include <rmd/device_image.cuh>

namespace rmd
{

namespace denoise
{

struct DeviceData
{
  DeviceData(
      DeviceImage<float>  * const u_dev_ptr,
      DeviceImage<float>  * const u_head_dev_ptr,
      DeviceImage<float2> * const p_dev_ptr,
      DeviceImage<float>  * const g_dev_ptr,
      const size_t &w,
      const size_t &h);
  const float L;
  const float tau;
  const float sigma;
  const float theta;

  DeviceImage<float>  * const u;
  DeviceImage<float>  * const u_head;
  DeviceImage<float2> * const p;
  DeviceImage<float>  * const g;

  const size_t width;
  const size_t height;

  float large_sigma_sq;
  float lambda;
};

} // denoise namespace

} // rmd namespace

#endif // RMD_DENOISE_DEVICE_DATA_CUH
