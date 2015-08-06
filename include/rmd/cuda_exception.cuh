#ifndef RMD_CUDA_EXCEPTION_CUH_
#define RMD_CUDA_EXCEPTION_CUH_

#include <sstream>
#include <cuda_runtime.h>

namespace rmd
{

struct CudaException : public std::exception
{
  CudaException(const std::string& what, cudaError err)
    : what_(what), err_(err) {}
  virtual ~CudaException() throw() {}
  virtual const char* what() const throw()
  {
    std::stringstream description;
    description << "CudaException: " << what_ << std::endl;
    if(err_ != cudaSuccess)
    {
      description << "cudaError code: " << cudaGetErrorString(err_);
      description << " (" << err_ << ")" << std::endl;
    }
    return description.str().c_str();
  }
  std::string what_;
  cudaError err_;
};

} // namespace rmd

#endif /* RMD_CUDA_EXCEPTION_CUH_ */
