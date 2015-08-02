#include <rmd/padded_memory.cuh>

rmd::PaddedMemory::PaddedMemory(
    const size_t &width,
    const size_t &height,
    float **dev_mem,
    size_t *pitch)
  : m_is_dev_alloc(false)
  , m_dev_mem(*dev_mem)
{
  m_is_dev_alloc = (cudaSuccess == cudaMallocPitch(
                      dev_mem,
                      pitch,
                      width*sizeof(float),
                      height));
}

rmd::PaddedMemory::~PaddedMemory()
{
  if(m_is_dev_alloc)
  {
    cudaFree(m_dev_mem);
  }
}
