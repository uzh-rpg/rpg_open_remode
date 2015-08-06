#include <rmd/seed_matrix.cuh>
#include <rmd/texture_memory.cuh>

#include "seed_init.cu"

rmd::SeedMatrix::SeedMatrix(
    const size_t &width,
    const size_t &height,
    const PinholeCamera &cam)
  : m_width(width)
  , m_height(height)
  , ref_img_(width, height)
  , curr_img_(width, height)
  , mu_(width, height)
  , sigma_(width, height)
  , a_(width, height)
  , b_(width, height)
{

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
  cudaFree(m_dev_ptr);
}

bool rmd::SeedMatrix::setReferenceImage(
    float *host_ref_img_align_row_maj,
    const SE3<float> &T_curr_world,
    const float &min_depth,
    const float &max_depth)
{
  ref_img_.setDevData(host_ref_img_align_row_maj);

  m_host_data.ref_img.data   = ref_img_.getDevDataPtr();
  m_host_data.ref_img.pitch  = ref_img_.getPitch();
  m_host_data.ref_img.stride = ref_img_.getStride();

  m_host_data.curr_img.data   = curr_img_.getDevDataPtr();
  m_host_data.curr_img.pitch  = curr_img_.getPitch();
  m_host_data.curr_img.stride = curr_img_.getStride();

  m_host_data.mu.data   = mu_.getDevDataPtr();
  m_host_data.mu.pitch  = mu_.getPitch();
  m_host_data.mu.stride = mu_.getStride();

  m_host_data.sigma.data   = sigma_.getDevDataPtr();
  m_host_data.sigma.pitch  = sigma_.getPitch();
  m_host_data.sigma.stride = sigma_.getStride();

  m_host_data.a.data   = a_.getDevDataPtr();
  m_host_data.a.pitch  = a_.getPitch();
  m_host_data.a.stride = a_.getStride();

  m_host_data.b.data   = b_.getDevDataPtr();
  m_host_data.b.pitch  = b_.getPitch();
  m_host_data.b.stride = b_.getStride();

  m_host_data.scene.min_depth = min_depth;
  m_host_data.scene.max_depth = max_depth;
  m_host_data.scene.avg_depth = (min_depth+max_depth)/2.0f;
  m_host_data.scene.depth_range = max_depth - min_depth;
  m_host_data.scene.sigma_sq_max = m_host_data.scene.depth_range * m_host_data.scene.depth_range / 36.0f;

  cudaMemcpy(m_dev_ptr, &m_host_data, sizeof(m_host_data), cudaMemcpyHostToDevice);

  m_T_world_ref = T_curr_world.inv();

  rmd::bindTexture(ref_img_tex, ref_img_);

  rmd::seedInitKernel<<<m_dim_grid, m_dim_block>>>(m_dev_ptr);

  return true;
}

bool rmd::SeedMatrix::update(
    float *host_curr_img_align_row_maj,
    const SE3<float> &T_curr_world)
{
  curr_img_.setDevData(host_curr_img_align_row_maj);

  const rmd::SE3<float> T_curr_ref = T_curr_world * m_T_world_ref;

  rmd::bindTexture(curr_img_tex, curr_img_);

  rmd::bindTexture(mu_tex, mu_);

  rmd::bindTexture(sigma_tex, sigma_);

  rmd::bindTexture(a_tex, a_);

  rmd::bindTexture(b_tex, b_);


  return true;
}
