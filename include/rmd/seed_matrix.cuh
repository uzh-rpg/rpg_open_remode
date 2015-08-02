#ifndef SEED_MATRIX_CUH
#define SEED_MATRIX_CUH

#include <cuda_runtime.h>
#include <rmd/padded_memory.cuh>
#include <rmd/pinhole_camera.cuh>
#include <rmd/se3.cuh>

namespace rmd
{

struct DeviceData
{
  float *mu;
  size_t mu_pitch;
  float *sigma;
  size_t sigma_pitch;
  float *a;
  size_t a_pitch;
  float *b;
  size_t b_pitch;
  PinholeCamera cam;
};

class SeedMatrix
{
public:
  SeedMatrix(
      const size_t &width,
      const size_t &height,
      const PinholeCamera &cam);
  ~SeedMatrix();
private:
  PaddedMemory *m_mu, *m_sigma, *m_a, *m_b;
  DeviceData m_host_data, *m_dev_ptr;
};

} // rmd namespace

#endif // SEED_MATRIX_CUH
