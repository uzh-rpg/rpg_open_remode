#ifndef RMD_TEXTURE_MEMORY_CUH
#define RMD_TEXTURE_MEMORY_CUH

#include <cuda_runtime.h>
#include <rmd/padded_memory.cuh>

namespace rmd
{

texture<float, cudaTextureType2D, cudaReadModeElementType> ref_img_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> curr_img_tex;

texture<float, cudaTextureType2D, cudaReadModeElementType> mu_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> sigma_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> a_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> b_tex;

template<typename ElementType>
inline bool bindTexture(
    texture<ElementType, cudaTextureType2D> &tex,
    PaddedMemory<ElementType> &mem,
    cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;

  Device2DData<ElementType> dev_data;
  mem.getDevData(dev_data);

  const cudaError bindStatus =
      cudaBindTexture2D(
        0,
        tex,
        dev_data.data,
        mem.getChannelFormatDesc(),
        mem.getWidth(),
        mem.getHeight(),
        dev_data.pitch
        );
  return bindStatus == cudaSuccess;
}

} // rmd namespace

#endif // RMD_TEXTURE_MEMORY_CUH
