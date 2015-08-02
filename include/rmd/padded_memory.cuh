#ifndef PADDED_MEMORY_CUH
#define PADDED_MEMORY_CUH

#include <rmd/device_data.cuh>

namespace rmd
{

class PaddedMemory
{
public:
  PaddedMemory(
      const size_t &width,
      const size_t &height,
      rmd::Device2DData &dev_data);

  ~PaddedMemory();

// TODO: define copy constructor and assignment operator

  float * getDevDataPtr() { return m_dev_mem; }
  cudaChannelFormatDesc getChannelFormatDesc() const { return m_channel_format_desc; }
  size_t getWidth()  const { return m_width;  }
  size_t getHeight() const { return m_height; }
  size_t getPitch()  const { return m_pitch;  }

private:
  size_t m_width;
  size_t m_height;
  bool m_is_dev_alloc;
  float *m_dev_mem;
  size_t m_pitch;
  cudaChannelFormatDesc m_channel_format_desc;
};

} // rmd namespace

#endif // PADDED_MEMORY_CUH
