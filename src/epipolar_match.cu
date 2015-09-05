#ifndef RMD_EPIPOLAR_MATCH_CU
#define RMD_EPIPOLAR_MATCH_CU

#include <float.h>
#include <rmd/se3.cuh>
#include <rmd/seed_matrix.cuh>
#include <rmd/device_data.cuh>
#include <rmd/texture_memory.cuh>
#include <rmd/helper_vector_types.cuh>

namespace rmd
{

__global__
void seedEpipolarMatch(
    DeviceData *dev_ptr,
    rmd::SE3<float> T_curr_ref)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= dev_ptr->width || y >= dev_ptr->height)
    return;

  const float xx = x+0.5f;
  const float yy = y+0.5f;

  const int8_t seed_state = tex2D(convergence_tex, xx, yy);
  if( (ConvergenceStates::BORDER    == seed_state) ||
      (ConvergenceStates::CONVERGED == seed_state) ||
      (ConvergenceStates::DIVERGED  == seed_state) )
  {
    return;
  }

  // Retrieve current estimations of depth
  const float mu = tex2D(mu_tex, xx, yy);
  const float sigma = sqrtf(tex2D(sigma_tex, xx, yy));

  const float2 px_ref = make_float2((float)x, (float)y);
  const float3 f_ref = normalize(dev_ptr->cam.cam2world(px_ref));
  const float2 px_mean_curr =
      dev_ptr->cam.world2cam( T_curr_ref * (f_ref * mu) );

  if( (px_mean_curr.x >= dev_ptr->width)  ||
      (px_mean_curr.y >= dev_ptr->height) ||
      (px_mean_curr.x < 0)                ||
      (px_mean_curr.y < 0) )
  {
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::NOT_VISIBLE;
    return;
  }

  const float2 px_min_curr =
      dev_ptr->cam.world2cam( T_curr_ref * (f_ref * fmaxf( mu - 3.0f*sigma, 0.01f)) );
  const float2 px_max_curr =
      dev_ptr->cam.world2cam( T_curr_ref * (f_ref * ( mu + (3.0f*sigma) ) ) );

  const float2 epi_line = px_max_curr - px_min_curr;
  const float2 epi_dir  = normalize(epi_line);
  const float  half_length = 0.5f * fminf(norm(epi_line), RMD_MAX_EXTENT_EPIPOLAR_SEARCH);
  float2 px_curr, best_px_curr;

  const int  &side   = dev_ptr->patch.side;
  const int2 &offset = dev_ptr->patch.offset;
  const float n = (float)side * (float)side;

  // Retrieve template statistics for NCC matching
  const float sum_templ = dev_ptr->sum_templ->atXY(x, y);
  const float const_templ_denom = dev_ptr->const_templ_denom->atXY(x, y);
  // init best match score
  float best_ncc = -1.0f;

  float sum_img;
  float sum_img_sq;
  float sum_img_templ;
  for(float l = -half_length; l <= half_length; l += 0.7f)
  {
    px_curr = px_mean_curr + l*epi_dir;
    if( (px_curr.x >= dev_ptr->width - dev_ptr->patch.side)  ||
        (px_curr.y >= dev_ptr->height - dev_ptr->patch.side) ||
        (px_curr.x < dev_ptr->patch.side)                    ||
        (px_curr.y < dev_ptr->patch.side) )
    {
      continue;
    }

    sum_img       = 0.0f;
    sum_img_sq    = 0.0f;
    sum_img_templ = 0.0f;

    for(int patch_y=0; patch_y<side; ++patch_y)
    {
      for(int patch_x=0; patch_x<side; ++patch_x)
      {
        const float templ = tex2D(
              ref_img_tex,
              px_ref.x+(float)(offset.x+patch_x)+0.5f,
              px_ref.y+(float)(offset.y+patch_y)+0.5f);
        const float img = tex2D(
              curr_img_tex,
              px_curr.x+(float)(offset.x+patch_x)+0.5f,
              px_curr.y+(float)(offset.y+patch_y)+0.5f);
        sum_img    += img;
        sum_img_sq += img*img;
        sum_img_templ += img*templ;
      }
    }
    const float ncc_numerator = n*sum_img_templ - sum_img*sum_templ;
    const float ncc_denominator = (n*sum_img_sq - sum_img*sum_img)*const_templ_denom;
    const float ncc = ncc_numerator * rsqrtf(ncc_denominator + FLT_MIN);

    if(ncc > best_ncc)
    {
      best_px_curr = px_curr;
      best_ncc = ncc;
    }
  }
  if(best_ncc < 0.0f)
  {
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::NO_MATCH;
  }
  else
  {
    dev_ptr->epipolar_matches->atXY(x, y) = best_px_curr;
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::UPDATE;
  }
}

} // rmd namespace

#endif
