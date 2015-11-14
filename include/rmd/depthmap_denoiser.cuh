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

#ifndef RMD_DEPTHMAP_DENOISER_CUH
#define RMD_DEPTHMAP_DENOISER_CUH

#include <rmd/device_image.cuh>
#include <rmd/denoise_device_data.cuh>

namespace rmd
{

class DepthmapDenoiser
{
public:
  DepthmapDenoiser(size_t width, size_t height);
  ~DepthmapDenoiser();
  void denoise(const rmd::DeviceImage<float> &mu,
               const rmd::DeviceImage<float> &sigma_sq,
               const rmd::DeviceImage<float> &a,
               const rmd::DeviceImage<float> &b,
               float *host_denoised,
               float lambda,
               int iterations);
  void setLargeSigmaSq(float depth_range);
private:
  DeviceImage<float> u_;
  DeviceImage<float> u_head_;
  DeviceImage<float2> p_;
  DeviceImage<float> g_;

  denoise::DeviceData * host_ptr;
  denoise::DeviceData * dev_ptr;

  dim3 dim_block_;
  dim3 dim_grid_;

  // Image size to be copied to constant memory
  Size host_img_size_;
};

} // rmd namespace

#endif // RMD_DEPTHMAP_DENOISER_CUH
