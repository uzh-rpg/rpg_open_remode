#include <rmd/seed_matrix.cuh>
#include <rmd/texture_memory.cuh>

#include "seed_init.cu"

rmd::SeedMatrix::SeedMatrix(
    const size_t &width,
    const size_t &height,
    const PinholeCamera &cam)
  : m_width(width)
  , m_height(height)
{
  m_ref_img  = new PaddedMemory(width, height, m_host_data.ref_img);
  m_curr_img = new PaddedMemory(width, height, m_host_data.curr_img);

  m_mu    = new PaddedMemory(width, height, m_host_data.mu);
  m_sigma = new PaddedMemory(width, height, m_host_data.sigma);
  m_a     = new PaddedMemory(width, height, m_host_data.a);
  m_b     = new PaddedMemory(width, height, m_host_data.b);

  m_host_data.cam    = cam;
  m_host_data.width  = width;
  m_host_data.height = height;

  cudaMalloc(&m_dev_ptr, sizeof(m_host_data));
  cudaMemcpy(m_dev_ptr, &m_host_data, sizeof(m_host_data), cudaMemcpyHostToDevice);

  m_dim_block.x = 16;
  m_dim_block.y = 16;
  m_dim_grid.x = (width  + m_dim_block.x - 1) / m_dim_block.x;
  m_dim_grid.y = (height + m_dim_block.y - 1) / m_dim_block.y;
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
        m_host_data.ref_img.data,
        m_host_data.ref_img.pitch,
        host_ref_img_align_row_maj,
        m_width*sizeof(float),
        m_width*sizeof(float),
        m_height,
        cudaMemcpyHostToDevice);
  if(err != cudaSuccess)
    return false;

  m_T_world_ref = T_curr_world.inv();

  rmd::bindTexture(img_ref_tex, *m_ref_img);

  rmd::seedInitKernel<<<m_dim_grid, m_dim_block>>>(m_dev_ptr);

  return true;
}
