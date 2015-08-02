#include <rmd/seed_matrix.cuh>

rmd::SeedMatrix::SeedMatrix(
    const size_t &width,
    const size_t &height,
    const PinholeCamera &cam)
{
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
  delete m_mu;
  delete m_sigma;
  delete m_a;
  delete m_b;
  cudaFree(m_dev_ptr);
}
