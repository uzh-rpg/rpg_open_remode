#include <rmd/padded_memory.cuh>

rmd::PaddedMemory::PaddedMemory(
    const size_t &width,
    const size_t &height,
    rmd::Device2DData &dev_data)
  : m_width(width)
  , m_height(height)
  , m_is_dev_alloc(false)
  , m_dev_mem(dev_data.data)
{
  m_is_dev_alloc = (cudaSuccess == cudaMallocPitch(
                      &dev_data.data,
                      &dev_data.pitch,
                      width*sizeof(float),
                      height));
  m_pitch = dev_data.pitch;
  dev_data.stride = dev_data.pitch / sizeof(float);
  m_channel_format_desc = cudaCreateChannelDesc<float>();
}

rmd::PaddedMemory::~PaddedMemory()
{
  if(m_is_dev_alloc)
  {
    cudaFree(m_dev_mem);
  }
}
