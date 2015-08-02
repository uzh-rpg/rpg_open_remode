#include <rmd/seed_matrix.cuh>

rmd::SeedMatrix::SeedMatrix(
    const size_t &width,
    const size_t &height,
    const PinholeCamera &cam)
  : m_width(width)
  , m_height(height)
{
  m_ref_img  = new PaddedMemory(width, height, &m_host_data.ref_img,  &m_host_data.ref_img_pitch);
  m_curr_img = new PaddedMemory(width, height, &m_host_data.curr_img, &m_host_data.curr_img_pitch);

  m_mu    = new PaddedMemory(width, height, &m_host_data.mu,    &m_host_data.mu_pitch);
  m_sigma = new PaddedMemory(width, height, &m_host_data.sigma, &m_host_data.sigma_pitch);
  m_a     = new PaddedMemory(width, height, &m_host_data.a,     &m_host_data.a_pitch);
  m_b     = new PaddedMemory(width, height, &m_host_data.b,     &m_host_data.b_pitch);

  m_host_data.cam = cam;

  cudaMalloc(&m_dev_ptr, sizeof(m_host_data));
  cudaMemcpy(m_dev_ptr, &m_host_data, sizeof(m_host_data), cudaMemcpyHostToDevice);
}

rmd::SeedMatrix::~SeedMatrix()
{
  delete m_ref_img;
  delete m_curr_img;
  delete m_mu;
  delete m_sigma;
  delete m_a;
  delete m_b;
  cudaFree(m_dev_ptr);
}

bool rmd::SeedMatrix::setReferenceImage(
    float *host_ref_img_align_row_maj,
    const SE3<float> &T_curr_world)
{
  const cudaError err = cudaMemcpy2D(
        m_host_data.ref_img,
        m_host_data.ref_img_pitch,
        host_ref_img_align_row_maj,
        m_width*sizeof(float),
        m_width*sizeof(float),
        m_height,
        cudaMemcpyHostToDevice);

  m_T_world_ref = T_curr_world.inv();

  return err == cudaSuccess;
}
