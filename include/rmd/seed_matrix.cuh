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
  // Image data
  float *ref_img;
  size_t ref_img_pitch;
  float *curr_img;
  size_t curr_img_pitch;
  // Seed data
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
  bool setReferenceImage(
      float *host_ref_img_align_row_maj,
      const SE3<float> &T_curr_world);
private:
  size_t m_width;
  size_t m_height;
  PaddedMemory *m_ref_img, *m_curr_img;
  PaddedMemory *m_mu, *m_sigma, *m_a, *m_b;
  DeviceData m_host_data, *m_dev_ptr;
  SE3<float> m_T_world_ref;
};

} // rmd namespace

#endif // SEED_MATRIX_CUH
