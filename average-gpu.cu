#include <stdio.h>

#define TIME_INIT()                                                            \
  cudaEvent_t start;                                                           \
  cudaEvent_t end;                                                             \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&end);                                                       \
  int timing_index = 0

#define TIME_START() cudaEventRecord(start)

#define TIME_END()                                                             \
  cudaEventRecord(end);                                                        \
  if (timing != NULL) {                                                        \
    cudaEventSynchronize(start);                                               \
    cudaEventSynchronize(end);                                                 \
    cudaEventElapsedTime(timing + timing_index, start, end);                   \
    timing[timing_index] /= 1000.0f;                                           \
  }                                                                            \
  timing_index++

#define TIME_FINISH()                                                          \
  cudaEventDestroy(start);                                                     \
  cudaEventDestroy(end);

// Define maximum number of elements a thread will process
#define ELEMENTS_PER_THREAD 128

// CUDA kernel to compute row averages using shared memory and atomics
__global__ void average_rows_kernel(const int n, const int m,
                                    const int increment,
                                    const float* __restrict__ input,
                                    float* __restrict__ averages) {
  // Define shared memory for block-level reduction
  extern __shared__ float row_sums[];

  // Initialize shared memory to zero
  if (threadIdx.x < blockDim.x) {
    row_sums[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  // Calculate the number of elements this thread needs to process
  const int num_threads = blockDim.x * gridDim.x;
  const int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread processes multiple rows in a strided pattern
  for (int i = thread_id; i < n; i += num_threads) {
    // Compute partial sum for this row
    float partial_sum = 0.0f;

    // Process elements in chunks to improve cache locality
    for (int j = 0; j < m; j += ELEMENTS_PER_THREAD) {
      // Determine how many elements to process in this chunk
      int chunk_size = min(ELEMENTS_PER_THREAD, m - j);

      // Calculate partial sum for this chunk
      for (int k = 0; k < chunk_size; k++) {
        partial_sum += input[i * increment + j + k];
      }
    }

    // Use atomics to add this thread's partial sum to the shared row sum
    atomicAdd(&row_sums[threadIdx.x], partial_sum);
  }

  // Make sure all threads have added their partial sums
  __syncthreads();

  // One thread per row writes the final result to global memory
  for (int i = thread_id; i < n; i += num_threads) {
    if (i % blockDim.x == threadIdx.x) {
      averages[i] = row_sums[threadIdx.x] / m;
    }
  }
}

// Main function to compute averages on GPU
extern "C" void average_rows_gpu(const int n, const int m, const int increment,
                                 const float* host_input, float* host_averages,
                                 float* timing) {
  TIME_INIT();

  // Setup time - calculate kernel configuration
  TIME_START();
  int threadsPerBlock = 256;
  // Use a reasonable number of blocks for good occupancy
  int numBlocks = min(32, (n + threadsPerBlock - 1) / threadsPerBlock);

  // Calculate shared memory size - one float per thread
  int sharedMemSize = threadsPerBlock * sizeof(float);
  TIME_END();

  // Allocate device memory
  TIME_START();
  float* device_input;
  float* device_averages;

  cudaError_t error;

  error = cudaMalloc((void**) &device_input, n * increment * sizeof(float));
  if (error != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device input memory: %s\n",
            cudaGetErrorString(error));
    return;
  }

  error = cudaMalloc((void**) &device_averages, n * sizeof(float));
  if (error != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device averages memory: %s\n",
            cudaGetErrorString(error));
    cudaFree(device_input);
    return;
  }
  TIME_END();

  // Copy input data to device
  TIME_START();
  error = cudaMemcpy(device_input, host_input, n * increment * sizeof(float),
                     cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    fprintf(stderr, "Failed to copy input data to device: %s\n",
            cudaGetErrorString(error));
    cudaFree(device_input);
    cudaFree(device_averages);
    return;
  }
  TIME_END();

  // Launch kernel
  TIME_START();
  average_rows_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
      n, m, increment, device_input, device_averages);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "Error launching averaging kernel: %s\n",
            cudaGetErrorString(error));
    cudaFree(device_input);
    cudaFree(device_averages);
    return;
  }

  cudaDeviceSynchronize();
  TIME_END();

  // Copy results back to host
  TIME_START();
  error = cudaMemcpy(host_averages, device_averages, n * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    fprintf(stderr, "Failed to copy averages from device: %s\n",
            cudaGetErrorString(error));
  }
  TIME_END();

  // Free device memory
  cudaFree(device_input);
  cudaFree(device_averages);

  TIME_FINISH();
}
