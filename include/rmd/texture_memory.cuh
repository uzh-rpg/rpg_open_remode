#ifndef RMD_TEXTURE_MEMORY_CUH
#define RMD_TEXTURE_MEMORY_CUH

#include <cuda_runtime.h>
#include <rmd/image.cuh>

namespace rmd
{

texture<float, cudaTextureType2D, cudaReadModeElementType> ref_img_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> curr_img_tex;

texture<float, cudaTextureType2D, cudaReadModeElementType> mu_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> sigma_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> a_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> b_tex;

template<typename ElementType>
inline void bindTexture(
    texture<ElementType, cudaTextureType2D> &tex,
    Image<ElementType> &mem,
    cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;

  const cudaError bindStatus = cudaBindTexture2D(
        0,
        tex,
        mem.getDevDataPtr(),
        mem.getChannelFormatDesc(),
        mem.getWidth(),
        mem.getHeight(),
        mem.getPitch()
        );
  if(bindStatus != cudaSuccess)
    throw CudaException("Unable to bind texture: ", bindStatus);
}

} // rmd namespace

#endif // RMD_TEXTURE_MEMORY_CUH
