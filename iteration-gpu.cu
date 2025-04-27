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

__global__ void init_gpu(const int n, const int m, const int increment,
                         float* const grid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  float col0 = 0.98f * (float) ((i + 1) * (i + 1)) / (float) (n * n);
  grid[i * increment + 0] = col0;

  // Set interior points
  for (int j = 1; j < m; j++) {
    grid[i * increment + j] =
        col0 * ((float) (m - j) * (m - j) / (float) (m * m));
  }

  // Set wraparound columns
  grid[i * increment + m]     = grid[i * increment + 0];
  grid[i * increment + m + 1] = grid[i * increment + 1];
}

__global__ void iteration_gpu_shared(const int n, const int m,
                                     const int increment, float* const dst,
                                     const float* const src) {
  int row      = blockIdx.x; // Each block handles one row
  int local_id = threadIdx.x;

  if (row >= n)
    return;

  const int row_offset = row * increment;

  // Allocate shared memory for a block of the row plus halo regions
  extern __shared__ float shared_row[];

  // Each thread loads one or more elements from global to shared memory
  for (int col = local_id; col < m + 4; col += blockDim.x) {
    // Map to the correct position in global memory with wraparound
    int global_col;
    if (col < 2) {
      // Left halo comes from right edge of row (m-2 and m-1)
      global_col = m - 2 + col;
    } else if (col >= m + 2) {
      // Right halo comes from left edge of row (0 and 1)
      global_col = col - m;
    } else {
      // Regular column
      global_col = col - 2;
    }

    // Load global memory into shared memory
    shared_row[col] = src[row_offset + global_col];
  }

  // Synchronize to ensure all data is loaded into shared memory
  __syncthreads();

  // Handle column 0 separately as a boundary condition
  if (local_id == 0) {
    dst[row_offset] = src[row_offset];
  }

  // Process columns in parallel using shared memory
  for (int col = local_id + 1; col < m; col += blockDim.x) {
    // Shared memory indices account for the 2-element halo on each side
    // Col 1 in global memory is at index 3 in shared memory
    int shared_idx = col + 2;

    // Apply the stencil using shared memory
    float result =
        ((1.60f * shared_row[shared_idx - 2]) +
         (1.55f * shared_row[shared_idx - 1]) + shared_row[shared_idx] +
         (0.60f * shared_row[shared_idx + 1]) +
         (0.25f * shared_row[shared_idx + 2])) /
        5.0f;

    dst[row_offset + col] = result;
  }

  // Update wraparound columns
  if (local_id == 0) {
    dst[row_offset + m]     = dst[row_offset + 0];
    dst[row_offset + m + 1] = dst[row_offset + 1];
  }
}

extern "C" void heat_propagation_gpu(const int iters, const int n, const int m,
                                     float* host_grid, float* timing) {
  const int increment = m + 2; // Include extra columns for wraparound

  // Set up kernel parameters
  int threadsPerBlock = 256;
  int blocksForInit   = (n + threadsPerBlock - 1) / threadsPerBlock;

  // Calculate shared memory size - need space for m elements plus halo regions
  int sharedMemSize = (m + 4) * sizeof(float);

  // Use 1D grid for shared memory kernel - one block per row
  dim3 blockSize(256);
  dim3 gridSize(n);

  // Allocate memory on GPU
  float* device_src = NULL;
  float* device_dst = NULL;
  size_t grid_size  = n * increment * sizeof(float);

  cudaError_t error;
  TIME_INIT();

  // Allocate and initialise device memory
  TIME_START();
  error = cudaMalloc((void**) &device_src, grid_size);
  if (error != cudaSuccess) {
    fprintf(stderr, "Failed to allocate source memory: %s\n",
            cudaGetErrorString(error));
    return;
  }

  error = cudaMalloc((void**) &device_dst, grid_size);
  if (error != cudaSuccess) {
    fprintf(stderr, "Failed to allocate destination memory: %s\n",
            cudaGetErrorString(error));
    cudaFree(device_src);
    return;
  }

  // Initialise with zeros to start
  cudaMemset(device_src, 0, grid_size);
  cudaMemset(device_dst, 0, grid_size);
  TIME_END();

  // Set up initial conditions
  TIME_START();
  init_gpu<<<blocksForInit, threadsPerBlock>>>(n, m, increment, device_src);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "Error launching init kernel: %s\n",
            cudaGetErrorString(error));
    cudaFree(device_src);
    cudaFree(device_dst);
    return;
  }

  cudaDeviceSynchronize();

  // Copy initial conditions to dst buffer too
  cudaMemcpy(device_dst, device_src, grid_size, cudaMemcpyDeviceToDevice);
  TIME_END();

  // Perform iterations
  TIME_START();

  printf("Starting iterations on GPU...\n");

  // Handle special case for odd iteration count
  if (iters % 2 != 0) {
    iteration_gpu_shared<<<gridSize, blockSize, sharedMemSize>>>(
        n, m, increment, device_dst, device_src);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "Error launching iteration kernel: %s\n",
              cudaGetErrorString(error));
      cudaFree(device_src);
      cudaFree(device_dst);
      return;
    }

    cudaDeviceSynchronize();

    // Swap pointers for the next iterations
    float* temp = device_src;
    device_src  = device_dst;
    device_dst  = temp;
  }

  // Run paired iterations
  for (int iter = 0; iter < iters / 2; iter++) {
    // Print progress for large iterations
    if (iters >= 100 && iter % 100 == 0) {
      printf("GPU completed %d iterations\n", iter * 2);
    }

    // First iteration in the pair
    iteration_gpu_shared<<<gridSize, blockSize, sharedMemSize>>>(
        n, m, increment, device_dst, device_src);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "Error in iteration %d (first): %s\n", iter,
              cudaGetErrorString(error));
      break;
    }

    cudaDeviceSynchronize();

    // Second iteration in the pair
    iteration_gpu_shared<<<gridSize, blockSize, sharedMemSize>>>(
        n, m, increment, device_src, device_dst);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "Error in iteration %d (second): %s\n", iter,
              cudaGetErrorString(error));
      break;
    }

    cudaDeviceSynchronize();
  }

  printf("GPU iterations complete\n");
  TIME_END();

  // Copy results back to the host
  TIME_START();
  error = cudaMemcpy(host_grid, device_src, grid_size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    fprintf(stderr, "Failed to copy results back to host: %s\n",
            cudaGetErrorString(error));
  }
  TIME_END();

  // Free GPU memory
  cudaFree(device_src);
  cudaFree(device_dst);

  TIME_FINISH();
}
