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
  m_ref_img  = new PaddedMemory(width, height);
  m_ref_img->getDevData(m_host_data.ref_img);
  m_curr_img = new PaddedMemory(width, height);
  m_curr_img->getDevData(m_host_data.curr_img);

  m_mu = new PaddedMemory(width, height);
  m_mu->getDevData(m_host_data.mu);
  m_sigma = new PaddedMemory(width, height);
  m_sigma->getDevData(m_host_data.sigma);
  m_a = new PaddedMemory(width, height);
  m_a->getDevData(m_host_data.a);
  m_b = new PaddedMemory(width, height);
  m_b->getDevData(m_host_data.b);

  m_host_data.cam    = cam;
  m_host_data.one_pix_angle = cam.getOnePixAngle();
  m_host_data.width  = width;
  m_host_data.height = height;

  cudaMalloc(&m_dev_ptr, sizeof(m_host_data));

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
    const SE3<float> &T_curr_world,
    const float &min_depth,
    const float &max_depth)
{
  if(cudaSuccess != cudaMemcpy2D(
        m_host_data.ref_img.data,
        m_host_data.ref_img.pitch,
        host_ref_img_align_row_maj,
        m_width*sizeof(float),
        m_width*sizeof(float),
        m_height,
        cudaMemcpyHostToDevice))
    return false;

  m_host_data.scene.min_depth = min_depth;
  m_host_data.scene.max_depth = max_depth;
  m_host_data.scene.avg_depth = (min_depth+max_depth)/2.0f;
  m_host_data.scene.depth_range = max_depth - min_depth;
  m_host_data.scene.sigma_sq_max = m_host_data.scene.depth_range * m_host_data.scene.depth_range / 36.0f;

  cudaMemcpy(m_dev_ptr, &m_host_data, sizeof(m_host_data), cudaMemcpyHostToDevice);

  m_T_world_ref = T_curr_world.inv();

  if(cudaSuccess != rmd::bindTexture(ref_img_tex, *m_ref_img))
    return false;

  rmd::seedInitKernel<<<m_dim_grid, m_dim_block>>>(m_dev_ptr);

  return true;
}

bool rmd::SeedMatrix::update(
    float *host_curr_img_align_row_maj,
    const SE3<float> &T_curr_world)
{
  if(cudaSuccess != cudaMemcpy2D(
       m_host_data.curr_img.data,
       m_host_data.curr_img.pitch,
       host_curr_img_align_row_maj,
       m_width*sizeof(float),
       m_width*sizeof(float),
       m_height,
       cudaMemcpyHostToDevice))
    return false;
  const rmd::SE3<float> T_curr_ref = T_curr_world * m_T_world_ref;

  if(cudaSuccess != rmd::bindTexture(curr_img_tex, *m_curr_img))
    return false;
  if(cudaSuccess != rmd::bindTexture(mu_tex, *m_mu))
    return false;
  if(cudaSuccess != rmd::bindTexture(sigma_tex, *m_sigma))
    return false;
  if(cudaSuccess != rmd::bindTexture(a_tex, *m_a))
    return false;
  if(cudaSuccess != rmd::bindTexture(b_tex, *m_b))
    return false;

  return true;
}
