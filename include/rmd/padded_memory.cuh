#ifndef PADDED_MEMORY_CUH
#define PADDED_MEMORY_CUH

namespace rmd
{

class PaddedMemory
{
public:
  PaddedMemory(
      const size_t &width,
      const size_t &height,
      float **dev_mem,
      size_t *pitch);

  ~PaddedMemory();

// TODO: define copy constructor and assignment operator

private:
  bool m_is_dev_alloc;
  float *m_dev_mem;
};

} // rmd namespace

#endif // PADDED_MEMORY_CUH
