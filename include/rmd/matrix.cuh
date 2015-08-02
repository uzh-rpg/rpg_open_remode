/*
 * matrix.cuh
 *
 *  Created on: Feb 10, 2014
 *      Author: Matia Pizzoli
 */

#ifndef REMODE_CU_MATRIX_CUH_
#define REMODE_CU_MATRIX_CUH_

#include <cuda_runtime.h>
#include <ostream>

namespace rmd
{

template<typename Type, unsigned R, unsigned C>
struct Matrix
{
  __host__ __device__ __forceinline__
  Type operator()(int row, int col) const
  {
    return data[row*C+col];
  }

  __host__ __device__ __forceinline__
  Type & operator()(int row, int col)
  {
    return data[row*C+col];
  }

  __host__ __device__ __forceinline__
  Type operator[](int ind) const
  {
    return data[ind];
  }

  __host__ __device__ __forceinline__
  Type & operator[](int ind)
  {
    return data[ind];
  }

  __host__
  friend std::ostream & operator<<(std::ostream &out, const Matrix<Type, R, C> &m)
  {
    for(size_t row=0; row<R; ++row)
    {
      for(size_t col=0; col<C; ++col)
      {
        out << m(row, col);
      }
      out << std::endl;
    }
    return out;
  }

  Type data[R*C];
};

template<typename Type, unsigned R, unsigned CR, unsigned C>
__host__ __device__ __forceinline__
Matrix<Type, R, C> operator*(const Matrix<Type, R, CR> & lhs,
                             const Matrix<Type, CR, C> & rhs)
{
  Matrix<Type, R, C> result;
  for(size_t row=0; row<R; ++row)
  {
    for(size_t col=0; col<C; ++col)
    {
      result(row, col) = 0;
      for(size_t i=0; i<CR; ++i)
      {
        result(row, col) += lhs(row,i) * rhs(i,col);
      }
    }
  }
  return result;
}

template<typename Type>
__host__ __device__ __forceinline__
Matrix<Type, 2, 2> inv(const Matrix<Type, 2, 2> & in)
{
  Matrix<Type, 2, 2> out;
  float det = in[0]*in[3] - in[1]*in[2];
  out[0] =  in[3] / det;
  out[1] = -in[1] / det;
  out[2] = -in[2] / det;
  out[3] =  in[0] / det;
  return out;
}

} // namespace rmd

#endif // REMODE_CU_MATRIX_CUH_
