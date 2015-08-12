#ifndef HELPER_VECTOR_TYPES_CUH
#define HELPER_VECTOR_TYPES_CUH

#include <cuda_toolkit/helper_math.h>

template<typename VectorType>
__device__ __forceinline__
float norm(const VectorType & v)
{
  return sqrtf(dot(v, v));
}

#endif // HELPER_VECTOR_TYPES_CUH
