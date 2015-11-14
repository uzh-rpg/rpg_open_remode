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

#ifndef RMD_MVS_DEVICE_DATA_CUH
#define RMD_MVS_DEVICE_DATA_CUH

#include <rmd/pinhole_camera.cuh>
#include <rmd/device_image.cuh>

namespace rmd
{

namespace mvs
{

struct SceneData
{
  float min_depth;
  float max_depth;
  float avg_depth;
  float depth_range;
  float sigma_sq_max;
};

#ifndef RMD_CORR_PATCH_SIDE
#define RMD_CORR_PATCH_SIDE    5
#endif
#define RMD_CORR_PATCH_OFFSET -RMD_CORR_PATCH_SIDE/2
#define RMD_CORR_PATCH_AREA    RMD_CORR_PATCH_SIDE*RMD_CORR_PATCH_SIDE

// DeviceData struct stores pointers to dev memory.
// It is allocated and set from host.
struct DeviceData
{
  __host__
  DeviceData()
    : is_dev_allocated(false)
  {
  }
  __host__
  ~DeviceData()
  {
    if(is_dev_allocated)
      cudaFree(dev_ptr);
  }
  __host__
  void setDevData()
  {
    if(!is_dev_allocated)
    {
      // Allocate device memory
      const cudaError err = cudaMalloc(&dev_ptr, sizeof(*this));
      if(err != cudaSuccess)
        throw CudaException("DeviceData, cannot allocate device memory to store image parameters.", err);
      else
      {
        is_dev_allocated = true;
      }
    }
    // Copy data to device memory
    const cudaError err = cudaMemcpy(dev_ptr, this, sizeof(*this), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
      throw CudaException("DeviceData, cannot copy image parameters to device memory.", err);
  }


  DeviceImage<float> *ref_img;
  DeviceImage<float> *curr_img;
  DeviceImage<float> *sum_templ;
  DeviceImage<float> *const_templ_denom;
  DeviceImage<float> *mu;
  DeviceImage<float> *sigma;
  DeviceImage<float> *a;
  DeviceImage<float> *b;
  DeviceImage<int> *convergence;
  DeviceImage<float2> *epipolar_matches;

  PinholeCamera cam;
  float one_pix_angle;
  size_t width;
  size_t height;

  SceneData scene;

  // Algorithm parameters
  float eta_inlier;
  float eta_outlier;
  float epsilon;

  DeviceData *dev_ptr;
  bool is_dev_allocated;
};

} // mvs namespace

} // rmd namespace

#endif // RMD_MVS_DEVICE_DATA_CUH
