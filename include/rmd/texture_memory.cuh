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

#ifndef RMD_TEXTURE_MEMORY_CUH
#define RMD_TEXTURE_MEMORY_CUH

#include <cuda_runtime.h>
#include <rmd/device_image.cuh>

namespace rmd
{

texture<float, cudaTextureType2D, cudaReadModeElementType> ref_img_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> curr_img_tex;

texture<float, cudaTextureType2D, cudaReadModeElementType> mu_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> sigma_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> a_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> b_tex;

texture<int, cudaTextureType2D, cudaReadModeElementType> convergence_tex;
texture<float2, cudaTextureType2D, cudaReadModeElementType> epipolar_matches_tex;

texture<float, cudaTextureType2D, cudaReadModeElementType> g_tex;

// Pre-computed template statistics
texture<float, cudaTextureType2D, cudaReadModeElementType> sum_templ_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> const_templ_denom_tex;

template<typename ElementType>
inline void bindTexture(
    texture<ElementType, cudaTextureType2D> &tex,
    const DeviceImage<ElementType> &mem,
    cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;

  const cudaError bindStatus = cudaBindTexture2D(
        0,
        tex,
        mem.data,
        mem.getCudaChannelFormatDesc(),
        mem.width,
        mem.height,
        mem.pitch
        );
  if(bindStatus != cudaSuccess)
    throw CudaException("Unable to bind texture: ", bindStatus);
}

} // rmd namespace

#endif // RMD_TEXTURE_MEMORY_CUH
