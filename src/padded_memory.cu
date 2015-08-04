#include <rmd/padded_memory.cuh>

rmd::PaddedMemory::PaddedMemory(
    const size_t &width,
    const size_t &height)
  : m_width(width)
  , m_height(height)
  , m_is_dev_alloc(false)
{
  m_is_dev_alloc = (cudaSuccess == cudaMallocPitch(
                      &m_dev_mem,
                      &m_pitch,
                      width*sizeof(float),
                      height));

  m_channel_format_desc = cudaCreateChannelDesc<float>();
}

rmd::PaddedMemory::~PaddedMemory()
{
  if(m_is_dev_alloc)
  {
    cudaFree(m_dev_mem);
  }
}

void rmd::PaddedMemory::getDevData(rmd::Device2DData &dev_data) const
{
  dev_data.data   = m_dev_mem;
  dev_data.pitch  = m_pitch;
  dev_data.stride = m_pitch / sizeof(float);
}
