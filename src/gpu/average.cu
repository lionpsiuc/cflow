#include <stdio.h>

#include "../../include/gpu/utils.h"

__global__ void average_rows_kernel(const int n, const int m,
                                    const int increment,
                                    const float* __restrict__ input,
                                    float* __restrict__ averages) {
  const int row = blockIdx.x;
  if (row >= n)
    return;

  const float* row_start = input + row * increment;

  // Use shared memory for the reduction
  extern __shared__ float sdata[];
  const int               tid = threadIdx.x;

  // Each thread calculates partial sum for its elements
  float thread_sum = 0.0f;
  for (int j = tid; j < m; j += blockDim.x) {
    thread_sum += row_start[j];
  }

  // Store in shared memory
  sdata[tid] = thread_sum;
  __syncthreads();

  // Reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result for this row
  if (tid == 0) {
    averages[row] = sdata[0] / (float) m;
  }
}

extern "C" void average_rows_gpu(const int n, const int m, const int increment,
                                 const float* device_input,
                                 float* host_averages, float* timing) {
  INIT();

  START();
  // Simple configuration - one block per row
  int threadsPerBlock = 256;
  int numBlocks       = n;
  // Ensure enough shared memory for the reduction
  int sharedMemSize = threadsPerBlock * sizeof(float);
  END();

  START();
  float* device_averages = NULL;
  cudaMalloc((void**) &device_averages, n * sizeof(float));
  // Initialize to zeros to ensure clean results
  cudaMemset(device_averages, 0, n * sizeof(float));
  END();

  // timing[2]: Transfer to...
  START();
  END(); // No host-to-device transferring needed

  START();
  // Launch with one block per row
  average_rows_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
      n, m, increment, device_input, device_averages);
  cudaDeviceSynchronize(); // Make sure kernel is finished
  END();

  START();
  cudaMemcpy(host_averages, device_averages, n * sizeof(float),
             cudaMemcpyDeviceToHost);
  END();

  cudaFree(device_averages);
  COMPLETE();
}
