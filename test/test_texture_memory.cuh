#ifndef RMD_TEST_TEXTURE_MEMORY_CUH
#define RMD_TEST_TEXTURE_MEMORY_CUH

#include <cuda_runtime.h>
#include <rmd/device_image.cuh>

namespace rmd
{

texture<float, cudaTextureType2D, cudaReadModeElementType> img_tex;

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

#endif // RMD_TEST_TEXTURE_MEMORY_CUH
