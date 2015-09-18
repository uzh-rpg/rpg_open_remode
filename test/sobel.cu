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

#include <rmd/device_image.cuh>
#include "test_texture_memory.cuh"

namespace rmd
{

__global__
void sobelKernel(
    const DeviceImage<float> *in_dev_ptr,
    DeviceImage<float2> *out_dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= in_dev_ptr->width-1
     || y >= in_dev_ptr->height-1
     || x < 1
     || y < 1)
    return;

  const DeviceImage<float>  &in_img = *in_dev_ptr;
  const float ul = in_img(x-1, y-1);
  const float um = in_img(x,   y-1);
  const float ur = in_img(x+1, y-1);
  const float ml = in_img(x-1, y);
  const float mr = in_img(x+1, y);
  const float ll = in_img(x-1, y+1);
  const float lm = in_img(x,   y+1);
  const float lr = in_img(x+1, y+1);

  DeviceImage<float2> &out_grad = *out_dev_ptr;
  // Sobel operator
  //dx_out(x, y) = 1.0f*ul + 2.0f*ml + 1.0f*ll - 1.0f*ur - 2.0f*mr -1.0f*lr;
  //dy_out(x, y) = 1.0f*ul + 2.0f*um + 1.0f*ur -1.0f*ll - 2.0f*lm -1.0f*lr;
  // Use the Scharr operator for comparison with OpenCV
  out_grad(x, y).x = -3.0f*ul - 10.0f*ml - 3.0f*ll + 3.0f*ur + 10.0f*mr + 3.0f*lr;
  out_grad(x, y).y = -3.0f*ul - 10.0f*um - 3.0f*ur + 3.0f*ll + 10.0f*lm + 3.0f*lr;
}

__global__
void sobelTexKernel(DeviceImage<float2> *out_dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= out_dev_ptr->width-1
     || y >= out_dev_ptr->height-1
     || x < 1
     || y < 1)
    return;

  const float xx = x+0.5f;
  const float yy = y+0.5f;

  const float ul = tex2D(img_tex, xx-1.0f, yy-1.0f);
  const float um = tex2D(img_tex, xx,      yy-1.0f);
  const float ur = tex2D(img_tex, xx+1.0f, yy-1.0f);
  const float ml = tex2D(img_tex, xx-1.0f, yy);
  const float mr = tex2D(img_tex, xx+1.0f, yy);
  const float ll = tex2D(img_tex, xx-1.0f, yy+1.0f);
  const float lm = tex2D(img_tex, xx,      yy+1.0f);
  const float lr = tex2D(img_tex, xx+1.0f, yy+1.0f);

  DeviceImage<float2> &out_grad = *out_dev_ptr;
  // Sobel operator
  //dx_out(x, y) = 1.0f*ul + 2.0f*ml + 1.0f*ll - 1.0f*ur - 2.0f*mr -1.0f*lr;
  //dy_out(x, y) = 1.0f*ul + 2.0f*um + 1.0f*ur -1.0f*ll - 2.0f*lm -1.0f*lr;
  // Use the Scharr operator for comparison with OpenCV
  out_grad(x, y).x = -3.0f*ul - 10.0f*ml - 3.0f*ll + 3.0f*ur + 10.0f*mr + 3.0f*lr;
  out_grad(x, y).y = -3.0f*ul - 10.0f*um - 3.0f*ur + 3.0f*ll + 10.0f*lm + 3.0f*lr;
}

void sobel(
    const DeviceImage<float> &in_img,
    DeviceImage<float2> &out_grad)
{
  // CUDA fields
  dim3 dim_block;
  dim3 dim_grid;
  dim_block.x = 16;
  dim_block.y = 16;
  dim_grid.x = (in_img.width  + dim_block.x - 1) / dim_block.x;
  dim_grid.y = (in_img.height + dim_block.y - 1) / dim_block.y;
  sobelKernel<<<dim_grid, dim_block>>>(in_img.dev_ptr, out_grad.dev_ptr);
  cudaDeviceSynchronize();
}

void sobelTex(
    const DeviceImage<float> &in_img,
    DeviceImage<float2> &out_grad)
{
  rmd::bindTexture(img_tex, in_img);

  // CUDA fields
  dim3 dim_block;
  dim3 dim_grid;
  dim_block.x = 16;
  dim_block.y = 16;
  dim_grid.x = (in_img.width  + dim_block.x - 1) / dim_block.x;
  dim_grid.y = (in_img.height + dim_block.y - 1) / dim_block.y;
  sobelTexKernel<<<dim_grid, dim_block>>>(out_grad.dev_ptr);
  cudaDeviceSynchronize();
}

} // rmd namespace
