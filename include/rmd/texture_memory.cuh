#ifndef RMD_TEXTURE_MEMORY_CUH
#define RMD_TEXTURE_MEMORY_CUH

#include <cuda_runtime.h>
#include <rmd/padded_memory.cuh>

namespace rmd
{

texture<float, cudaTextureType2D, cudaReadModeElementType> img_ref_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> img_curr_tex;

texture<float, cudaTextureType2D, cudaReadModeElementType> mu_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> sigma_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> a_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> b_tex;

template<typename ElementType>
inline bool bindTexture(
    texture<ElementType, cudaTextureType2D> &tex,
    PaddedMemory &mem,
    cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
  const cudaError bindStatus =
      cudaBindTexture2D(
        0,
        tex,
        mem.getDevDataPtr(),
        mem.getChannelFormatDesc(),
        mem.getWidth(),
        mem.getHeight(),
        mem.getPitch());
  return bindStatus == cudaSuccess;
}

} // rmd namespace

#endif // RMD_TEXTURE_MEMORY_CUH
