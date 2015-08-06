#ifndef PADDED_MEMORY_CUH
#define PADDED_MEMORY_CUH

#include <rmd/device_data.cuh>

namespace rmd
{

template<typename ElementType>
class PaddedMemory
{
public:
  PaddedMemory(
      const size_t &width,
      const size_t &height)
    : m_width(width)
    , m_height(height)
    , m_is_dev_alloc(false)
  {
    m_is_dev_alloc = (cudaSuccess == cudaMallocPitch(
                        &m_dev_mem,
                        &m_pitch,
                        width*sizeof(ElementType),
                        height));

    m_channel_format_desc = cudaCreateChannelDesc<ElementType>();
  }

  ~PaddedMemory()
  {
    if(m_is_dev_alloc)
    {
      cudaFree(m_dev_mem);
    }
  }

  void getDevData(Device2DData<ElementType> &dev_data) const
  {
    dev_data.data   = m_dev_mem;
    dev_data.pitch  = m_pitch;
    dev_data.stride = m_pitch / sizeof(ElementType);
  }

  bool setDevData(const ElementType *host_img_align_row_maj)
  {
    if(cudaSuccess != cudaMemcpy2D(
          m_dev_mem,
          m_pitch,
          host_img_align_row_maj,
          m_width*sizeof(ElementType),
          m_width*sizeof(ElementType),
          m_height,
          cudaMemcpyHostToDevice))
      return false;
    else
    {
      return true;
    }
  }

  // TODO: define copy constructor and assignment operator

  cudaChannelFormatDesc getChannelFormatDesc() const { return m_channel_format_desc; }
  size_t getWidth()  const { return m_width;  }
  size_t getHeight() const { return m_height; }

private:
  size_t m_width;
  size_t m_height;
  bool m_is_dev_alloc;
  ElementType *m_dev_mem;
  size_t m_pitch;
  cudaChannelFormatDesc m_channel_format_desc;
};

} // rmd namespace

#endif // PADDED_MEMORY_CUH
