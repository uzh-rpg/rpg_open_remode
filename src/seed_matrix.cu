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
  // Save image details to be uploaded to device memory
  dev_data.ref_img.set(ref_img_);
  dev_data.curr_img.set(curr_img_);
  dev_data.mu.set(mu_);
  dev_data.sigma.set(sigma_);
  dev_data.a.set(a_);
  dev_data.b.set(b_);
  // Save camera parameters
  dev_data.cam    = cam;
  dev_data.one_pix_angle = cam.getOnePixAngle();
  dev_data.width  = width;
  dev_data.height = height;
  // Kernel configuration
  m_dim_block.x = 16;
  m_dim_block.y = 16;
  m_dim_grid.x = (width  + m_dim_block.x - 1) / m_dim_block.x;
  m_dim_grid.y = (height + m_dim_block.y - 1) / m_dim_block.y;
}

bool rmd::SeedMatrix::setReferenceImage(
    float *host_ref_img_align_row_maj,
    const SE3<float> &T_curr_world,
    const float &min_depth,
    const float &max_depth)
{
  // Upload reference image to device memory
  ref_img_.setDevData(host_ref_img_align_row_maj);
  // Set scene parameters
  dev_data.scene.min_depth = min_depth;
  dev_data.scene.max_depth = max_depth;
  dev_data.scene.avg_depth = (min_depth+max_depth)/2.0f;
  dev_data.scene.depth_range = max_depth - min_depth;
  dev_data.scene.sigma_sq_max = dev_data.scene.depth_range * dev_data.scene.depth_range / 36.0f;
  // Copy data to device memory
  dev_data.setDevData();

  m_T_world_ref = T_curr_world.inv();

  rmd::bindTexture(ref_img_tex, ref_img_);

  rmd::seedInitKernel<<<m_dim_grid, m_dim_block>>>(dev_data.dev_ptr);

  return true;
}

bool rmd::SeedMatrix::update(
    float *host_curr_img_align_row_maj,
    const SE3<float> &T_curr_world)
{
  // Upload current image to device memory
  curr_img_.setDevData(host_curr_img_align_row_maj);

  const rmd::SE3<float> T_curr_ref = T_curr_world * m_T_world_ref;

  rmd::bindTexture(curr_img_tex, curr_img_);
  rmd::bindTexture(mu_tex, mu_);
  rmd::bindTexture(sigma_tex, sigma_);
  rmd::bindTexture(a_tex, a_);
  rmd::bindTexture(b_tex, b_);

  return true;
}
