#ifndef RMD_REDUCTION_KERNELS_CU
#define RMD_REDUCTION_KERNELS_CU

namespace rmd
{

template<typename T>
__global__
void reductionSumKernel(T *out_dev_ptr,
                        size_t out_stride,
                        const T *in_dev_ptr,
                        size_t in_stride,
                        size_t n,
                        size_t m)
{
  extern __shared__ T s_partial[];
  int count = 0;

  // Sum over 2D thread grid, use (x,y) indices
  for(int x = blockIdx.x * blockDim.x + threadIdx.x;
      x < n;
      x += blockDim.x*gridDim.x)
  {
    for(int y = blockIdx.y * blockDim.y + threadIdx.y;
        y < m;
        y += blockDim.y*gridDim.y)
    {
      count += in_dev_ptr[y*in_stride+x];
    }
  }
  // Sums are written to shared memory, single index
  s_partial[threadIdx.y*blockDim.x+threadIdx.x] = count;
  __syncthreads();

  // Reduce over block sums stored in shared memory
  // Start using half the block threads,
  // halve the active threads at each iteration
  const int tid = threadIdx.y*blockDim.x+threadIdx.x;
  for (int num_active_threads = (blockDim.x*blockDim.y)>>1;
       num_active_threads;
       num_active_threads >>= 1 ) {
    if ( tid < num_active_threads)
    {
      s_partial[tid] += s_partial[tid+num_active_threads];
    }
    __syncthreads();
  }
  // Thread 0 writes the result for the block
  if(0 == tid)
  {
    out_dev_ptr[blockIdx.y*out_stride+blockIdx.x] = s_partial[0];
  }
}

template<typename T>
__global__
void reductionFindEqKernel(T *out_dev_ptr,
                           size_t out_stride,
                           const T *in_dev_ptr,
                           size_t in_stride,
                           size_t n,
                           size_t m,
                           T value)
{
  extern __shared__ T s_partial[];
  int count = 0;

  // Sum over 2D thread grid, use (x,y) indices
  for(int x = blockIdx.x * blockDim.x + threadIdx.x;
      x < n;
      x += blockDim.x*gridDim.x)
  {
    for(int y = blockIdx.y * blockDim.y + threadIdx.y;
        y < m;
        y += blockDim.y*gridDim.y)
    {
      if(value == in_dev_ptr[y*in_stride+x])
      {
        count += 1;
      }
    }
  }
  // Sums are written to shared memory, single index
  s_partial[threadIdx.y*blockDim.x+threadIdx.x] = count;
  __syncthreads();

  // Reduce over block sums stored in shared memory
  // Start using half the block threads,
  // halve the active threads at each iteration
  const int tid = threadIdx.y*blockDim.x+threadIdx.x;
  for (int num_active_threads = (blockDim.x*blockDim.y)>>1;
       num_active_threads;
       num_active_threads >>= 1 ) {
    if (tid < num_active_threads)
    {
      s_partial[tid] += s_partial[tid+num_active_threads];
    }
    __syncthreads();
  }
  // Thread 0 writes the result for the block
  if(0 == tid)
  {
    out_dev_ptr[blockIdx.y*out_stride+blockIdx.x] = s_partial[0];
  }
}

} // rmd namespace

#endif // RMD_REDUCTION_KERNELS_CU
