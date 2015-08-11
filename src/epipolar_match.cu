#ifndef RMD_EPIPOLAR_MATCH_CU
#define RMD_EPIPOLAR_MATCH_CU

#include <float.h>
#include <rmd/se3.cuh>
#include <rmd/device_data.cuh>

namespace rmd
{

template<typename VectorType>
__device__ __forceinline__
float norm(const VectorType & v)
{
  return sqrtf(dot(v, v));
}

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

  const unsigned char seed_state = tex2D(convergence_tex, xx, yy);
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
    dev_ptr->convergence.data[y*dev_ptr->convergence.stride+x] =
        ConvergenceStates::NOT_VISIBLE;
    return;
  }

  const float2 px_min_curr =
      dev_ptr->cam.world2cam( T_curr_ref * (f_ref * fmaxf( mu - 3.0f*sqrtf(sigma), 0.01f)) );
  const float2 px_max_curr =
      dev_ptr->cam.world2cam( T_curr_ref * (f_ref * ( mu + (3.0f*sqrtf(sigma)) ) ) );

  const float2 epi_line = px_max_curr - px_min_curr;
  const float2 epi_dir  = normalize(epi_line);
  const float  half_length = 0.5f * fminf(norm(epi_line), MAX_EXTENT_EPIPOLAR_SEARCH);
  float2 px_curr, best_px_curr;

  // Retrieve template statistics for NCC matching
  const float sum_templ =
      dev_ptr->sum_templ.data[y*dev_ptr->sum_templ.stride+x];
  const float const_templ_denom =
      dev_ptr->const_templ_denom.data[y*dev_ptr->const_templ_denom.stride+x];
  float best_ncc = -1.0f;

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

    float sum_img       = 0.0f;
    float sum_img_sq    = 0.0f;
    float sum_img_templ = 0.0f;

    const int  &side   = dev_ptr->patch.side;
    const int2 &offset = dev_ptr->patch.offset;

    for(int patch_y=0; patch_y<side; ++patch_y)
    {
      for(int patch_x=0; patch_x<side; ++patch_x)
      {
        const float templ = tex2D(
              ref_img_tex,
              (float)(px_ref.x+offset.x+patch_x)+0.5f,
              (float)(px_ref.y+offset.y+patch_y)+0.5f);
        const float img = tex2D(
              curr_img_tex,
              (float)(px_curr.x+offset.x+patch_x)+0.5f,
              (float)(px_curr.y+offset.y+patch_y)+0.5f);
        sum_img    += img;
        sum_img_sq += img*img;
        sum_img_templ += img*templ;
      }
    }
    const float ncc_numerator = side*side*sum_img_templ - sum_img*sum_templ;
    const float ncc_denominator = ((float)side*(float)side*(float)sum_img_sq -
                                   (float)sum_img*(float)sum_img)*const_templ_denom;
    const float ncc = ncc_numerator * rsqrtf(ncc_denominator + FLT_MIN);

    if(ncc > best_ncc)
    {
      best_px_curr = px_curr;
      best_ncc = ncc;
    }
  }
  if(best_ncc < 0.5f)
  {
    dev_ptr->convergence.data[y*dev_ptr->convergence.stride+x] =
        ConvergenceStates::NO_MATCH;
  }
  else
  {
    dev_ptr->epipolar_matches.data[y*dev_ptr->epipolar_matches.stride+x] = best_px_curr;
    dev_ptr->convergence.data[y*dev_ptr->convergence.stride+x] =
        ConvergenceStates::UPDATE;
  }
}

} // rmd namespace

#endif
