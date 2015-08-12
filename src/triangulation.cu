#ifndef RMD_TRIANGULATION_CU
#define RMD_TRIANGULATION_CU

namespace rmd
{

// Returns 3D point in reference frame
// Non-linear formulation (ref. to the book 'Autonomous Mobile Robots')
__device__ __forceinline__
float3 triangulatenNonLin(
    const float3 &bearing_vector_ref,
    const float3 &bearing_vector_curr,
    const SE3<float> &T_ref_curr)
{
  const float3 t = T_ref_curr.getTranslation();
  float3 f2 = T_ref_curr.rotate(bearing_vector_curr);
  const float2 b = make_float2(dot(t, bearing_vector_ref),
                               dot(t, f2));
  float A[2*2];
  A[0] = dot(bearing_vector_ref, bearing_vector_ref);
  A[2] = dot(bearing_vector_ref, f2);
  A[1] = -A[2];
  A[3] = dot(-f2, f2);

  const float2 lambdavec = make_float2(A[3]*b.x - A[1]*b.y,
      -A[2]*b.x + A[0]*b.y) / (A[0]*A[3] - A[1]*A[2]);
  const float3 xm = lambdavec.x * bearing_vector_ref;
  const float3 xn = t + lambdavec.y * f2;
  return (xm + xn)/2.0f;
}

__device__ __forceinline__
float triangulationUncertainty(
    const float &z,
    const float3 &bearing_vector_ref,
    const float3 &t_ref_curr,
    const float &one_pix_angle)
{
  const float3 a = bearing_vector_ref * z - t_ref_curr;
  const float t_norm = norm(t_ref_curr);
  const float a_norm = norm(a);
  const float alpha = acosf(dot(bearing_vector_ref,t_ref_curr)/t_norm);
  const float beta = acosf((-dot(a,t_ref_curr))/(t_norm*a_norm));
  const float beta_plus = beta + one_pix_angle;
  const float gamma_plus = M_PI-alpha-beta_plus;              // triangle angles sum to PI
  const float z_plus = t_norm*sinf(beta_plus)/sinf(gamma_plus); // law of sines
  return z_plus - z;
}

}

#endif // RMD_TRIANGULATION_CU
