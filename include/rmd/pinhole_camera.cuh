/*
 * pinhole_camera.cuh
 *
 *  Created on: Feb 11, 2014
 *      Author: matia
 */

#ifndef REMODE_PINHOLE_CAMERA_CUH_
#define REMODE_PINHOLE_CAMERA_CUH_

#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>

namespace rmd
{

struct PinholeCamera
{
  __host__
  PinholeCamera()
    : fx(0.0f), fy(0.0f), cx(0.0f), cy(0.0f)
  { }

  __host__
  PinholeCamera(float fx, float fy,
                float cx, float cy)
    : fx(fx), fy(fy), cx(cx), cy(cy)
  { }

  __host__ __device__ __forceinline__
  float3 cam2world(const float2 & uv) const
  {
    return make_float3((uv.x - cx)/fx,
                       (uv.y - cy)/fy,
                       1.0f);
  }

  __host__ __device__ __forceinline__
  float2 world2cam(const float3 & xyz) const
  {
    return make_float2(fx*xyz.x / xyz.z + cx,
                       fy*xyz.y / xyz.z + cy);
  }

  __host__ __device__ __forceinline__
  float getOnePixAngle() const
  {
    return atan2f(1.0f, 2.0f*fx)*2.0f;
  }

  float fx, fy;
  float cx, cy;
};

} // namespace rmd


#endif // REMODE_PINHOLE_CAMERA_CUH_
