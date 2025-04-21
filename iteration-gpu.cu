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

// Initialise the grid on the GPU
__global__ void init_gpu(const int n, const int m, const int increment,
                         float* const grid) {
  const int i = blockIdx.x;  // Row index
  const int j = threadIdx.x; // Column index

  // Skip threads that are out of bounds
  if (i >= n || j >= m)
    return;

  // Calculate the initial value for this element
  float col0 = 0.98f * (float) ((i + 1) * (i + 1)) / (float) (n * n);

  if (j == 0) {
    grid[i * increment + j]     = col0; // First column
    grid[i * increment + m + 0] = col0; // Set ghost column
  } else {

    // Interior points
    grid[i * increment + j] =
        col0 * ((float) (m - j) * (m - j) / (float) (m * m));

    // Set the second ghost column
    if (j == 1) {
      grid[i * increment + m + 1] = grid[i * increment + j];
    }
  }
}

// Perform a single iteration
__global__ void iteration_gpu(const int n, const int m, const int increment,
                              float* const dst, const float* const src) {
  const int i = blockIdx.x;  // Row index; one row per block
  const int j = threadIdx.x; // Column index within the row

  // Skip threads that are out of bounds
  if (i >= n || j >= m)
    return;

  const int rowOff =
      i *
      increment; // Calculate offset to the start of this row in global memory

  // Define shared memory layout
  extern __shared__ float srow[];

  // Calculate index into shared memory; note that we leave space for left halo
  const int sIdx = threadIdx.x + 2;

  if (j == 0) {
    float v             = src[rowOff + 0];
    dst[rowOff + 0]     = v;
    dst[rowOff + m + 0] = v;
    return;
  }

  // Calculate wrap-around indices for neighbours; we still need modulo even
  // with ghost columns because we are loading into shared memory
  const int jm2 = (j - 2 + m) % m;
  const int jm1 = (j - 1 + m) % m;
  const int jp1 = (j + 1) % m;
  const int jp2 = (j + 2) % m;

  // Load into shared memory
  srow[sIdx]     = src[rowOff + j];
  srow[sIdx - 1] = src[rowOff + jm1];
  srow[sIdx - 2] = src[rowOff + jm2];
  srow[sIdx + 1] = src[rowOff + jp1];
  srow[sIdx + 2] = src[rowOff + jp2];

  // Ensure all threads in the block have loaded their values
  __syncthreads();

  // Apply the five-point stencil using values from shared memory
  float result =
      ((1.60f * srow[sIdx - 2]) + (1.55f * srow[sIdx - 1]) + srow[sIdx] +
       (0.60f * srow[sIdx + 1]) + (0.25f * srow[sIdx + 2])) /
      5.0f;

  // Write result to global memory
  dst[rowOff + j] = result;

  // Update the second ghost column when handling second column
  if (j == 1) {
    dst[rowOff + (m + 1)] = result;
  }
}

// Simple wrapper to test above functions
extern "C" void test_wrapper(float* host_grid, int n, int m, float* timing) {
  const int increment       = m + 2;
  int       threadsPerBlock = 256; // Define block size

  // Ensure we don't exceed maximum threads per block
  if (threadsPerBlock > 1024)
    threadsPerBlock = 1024;

  // If m is small, adjust threadsPerBlock to avoid wasting threads; this will
  // still call 32 in the smallest cases so I need to figure out a better
  // alternative
  if (threadsPerBlock > m)
    threadsPerBlock = m;

  // Define grid size, where we have one block per row
  int numBlocks = n;

  // Allocate device memory
  float* device_src = NULL;
  float* device_dst = NULL;
  size_t grid_size  = n * increment * sizeof(float);

  // Calculate shared memory size per block
  const int sharedSzPerRow = m + 4; // Entire row enough for two extra columns
                                    // on each side for the halo columns
  const size_t shm_size = sharedSzPerRow * sizeof(float);

  cudaError_t error;
  TIME_INIT();

  // Allocate and initialise device memory
  TIME_START();
  error = cudaMalloc((void**) &device_src, grid_size);
  if (error != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device source memory: %s\n",
            cudaGetErrorString(error));
    return;
  }
  error = cudaMalloc((void**) &device_dst, grid_size);
  if (error != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device destination memory: %s\n",
            cudaGetErrorString(error));
    cudaFree(device_src);
    return;
  }
  cudaMemset(device_src, 0, grid_size);
  cudaMemset(device_dst, 0, grid_size);
  TIME_END();

  // Run initialisation kernel
  TIME_START();
  init_gpu<<<numBlocks, threadsPerBlock>>>(n, m, increment, device_src);
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    fprintf(stderr, "Error in initialisation kernel: %s\n",
            cudaGetErrorString(error));
    cudaFree(device_src);
    cudaFree(device_dst);
    return;
  }
  TIME_END();

  // Run iteration kernel
  TIME_START();
  iteration_gpu<<<numBlocks, threadsPerBlock, shm_size>>>(
      n, m, increment, device_dst, device_src);
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    fprintf(stderr, "Error in iteration kernel: %s\n",
            cudaGetErrorString(error));
    cudaFree(device_src);
    cudaFree(device_dst);
    return;
  }
  TIME_END();

  // Copy results back to host
  TIME_START();
  error = cudaMemcpy(host_grid, device_dst, grid_size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    fprintf(stderr, "Failed to copy result back to host: %s\n",
            cudaGetErrorString(error));
    cudaFree(device_src);
    cudaFree(device_dst);
    return;
  }
  TIME_END();

  // Free device memory
  cudaFree(device_src);
  cudaFree(device_dst);

  TIME_FINISH();
}
