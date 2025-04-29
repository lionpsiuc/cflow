#include "../../include/gpu/utils.h"

__global__ void average_rows_kernel(const int n, const int m,
                                    const int increment,
                                    const float* __restrict__ input,
                                    float* __restrict__ averages) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) {
    return;
  }
  float        sum       = 0.0f;
  const float* row_start = input + row * increment;
  for (int j = 0; j < m; j++) {
    sum += row_start[j];
  }
  averages[row] = sum / (float) m;
}

__global__ void average_rows_kernel_parallel(const int n, const int m,
                                             const int increment,
                                             const float* __restrict__ input,
                                             float* __restrict__ averages) {
  extern __shared__ float sdata[];
  const int               row = blockIdx.x;
  if (row >= n)
    return;
  const int    tid        = threadIdx.x;
  const float* row_start  = input + row * increment;
  float        thread_sum = 0.0f;
  for (int j = tid; j < m; j += blockDim.x) {
    thread_sum += row_start[j];
  }
  sdata[tid] = thread_sum;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    averages[row] = sdata[0] / (float) m;
  }
}

extern "C" void average_rows_gpu(const int n, const int m, const int increment,
                                 const float* device_input,
                                 float* host_averages, float* timing) {

  INIT();
  START();
  int       threadsPerBlock;
  int       numBlocks;
  bool      useParallelKernel;
  int       sharedMemSize           = 0;
  const int max_blocks_for_parallel = 1024;
  const int min_cols_for_parallel   = 64;
  if (n <= max_blocks_for_parallel && m >= min_cols_for_parallel) {
    threadsPerBlock   = 256;
    numBlocks         = n;
    useParallelKernel = true;
    sharedMemSize     = threadsPerBlock * sizeof(float);
  } else {
    threadsPerBlock   = 256;
    numBlocks         = (n + threadsPerBlock - 1) / threadsPerBlock;
    useParallelKernel = false;
    sharedMemSize     = 0;
  }
  END();
  START();
  float* device_averages = NULL;
  cudaMalloc((void**) &device_averages, n * sizeof(float));
  END();

  // timing[2] measures the time taken for transferring to device (i.e., this is
  // not required since we are averaging the matrix already stored in the
  // device)
  START();
  END();

  START();
  if (useParallelKernel) {
    average_rows_kernel_parallel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        n, m, increment, device_input, device_averages);
  } else {
    average_rows_kernel<<<numBlocks, threadsPerBlock>>>(
        n, m, increment, device_input, device_averages);
  }
  cudaDeviceSynchronize();
  END();
  START();
  cudaMemcpy(host_averages, device_averages, n * sizeof(float),
             cudaMemcpyDeviceToHost);
  END();

  cudaFree(device_averages);
  COMPLETE();
}
