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

// Maximum elements to process in each tile (excluding halos)
#define MAX_TILE_SIZE 4096

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

__global__ void iteration_gpu_tiled(const int n, const int m,
                                    const int increment, float* const dst,
                                    const float* const src, const int tile_size,
                                    const int tiles_per_row) {
  // Calculate row and tile indices
  int row_idx  = blockIdx.x / tiles_per_row;
  int tile_idx = blockIdx.x % tiles_per_row;

  // Skip if we're out of bounds
  if (row_idx >= n)
    return;

  // Calculate row offset in global memory
  const int row_offset = row_idx * increment;

  // Calculate start and end of this tile (in global memory indices)
  int tile_start =
      tile_idx * tile_size + 1; // Start from column 1 (col 0 is boundary)
  int tile_end = min(tile_start + tile_size, m);

  // Calculate size of this tile
  int actual_tile_size = tile_end - tile_start;

  // Shared memory size calculation: actual tile + 4 halo elements
  extern __shared__ float sh_mem[];

  // Load data into shared memory (including halo regions)
  // Each thread loads multiple elements
  for (int i = threadIdx.x; i < actual_tile_size + 4; i += blockDim.x) {
    int global_col;

    // Handle halo regions with wraparound
    if (i < 2) {
      // Left halo
      if (tile_start <= 2) {
        // Special case: wrapping around to end of row
        global_col = (tile_start - 2 + i + m - 1) % m;
        if (global_col == 0)
          global_col = m;
      } else {
        // Regular case: previous columns
        global_col = tile_start - 2 + i;
      }
    } else if (i >= actual_tile_size + 2) {
      // Right halo
      if (tile_end >= m - 1) {
        // Special case: wrapping around to beginning of row
        global_col = (i - (actual_tile_size + 2) + tile_end) % m;
        if (global_col == 0)
          global_col = 0; // Use boundary value
      } else {
        // Regular case: next columns
        global_col = tile_end + (i - (actual_tile_size + 2));
      }
    } else {
      // Regular tile data
      global_col = tile_start + (i - 2);
    }

    // Load from global memory
    sh_mem[i] = src[row_offset + global_col];
  }

  // Ensure all data is loaded
  __syncthreads();

  // Process tile
  for (int i = threadIdx.x; i < actual_tile_size; i += blockDim.x) {
    // Skip column 0 (boundary condition)
    if (tile_start + i == 0)
      continue;

    // Shared memory indices (i+2 to account for halo)
    int sh_idx = i + 2;

    // Apply stencil
    float result =
        ((1.60f * sh_mem[sh_idx - 2]) + (1.55f * sh_mem[sh_idx - 1]) +
         sh_mem[sh_idx] + (0.60f * sh_mem[sh_idx + 1]) +
         (0.25f * sh_mem[sh_idx + 2])) /
        5.0f;

    // Write result to global memory
    dst[row_offset + tile_start + i] = result;
  }

  // Handle boundary condition for column 0
  if (tile_start <= 1 && threadIdx.x == 0) {
    dst[row_offset] = src[row_offset];
  }

  // Update wraparound columns
  if (tile_end >= m && threadIdx.x == 0) {
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

  // Calculate tile size and number of tiles per row
  // Subtract 4 for the halo regions (2 on each side)
  const int tile_size     = MAX_TILE_SIZE;
  const int tiles_per_row = (m + tile_size - 1) / tile_size;

  // Calculate actual shared memory needed per block
  // Add 4 for halo regions (2 on each side)
  int max_elements_in_tile = tile_size + 4;
  for (int i = 0; i < tiles_per_row - 1; i++) {
    max_elements_in_tile =
        max(max_elements_in_tile, min(tile_size, m - i * tile_size) + 4);
  }
  int sharedMemSize = max_elements_in_tile * sizeof(float);

  printf("Tile configuration:\t\t%d tiles per row, max %d elements per tile\n",
         tiles_per_row, max_elements_in_tile);
  printf("Shared memory per block:\t%d bytes (%.1f KB)\n", sharedMemSize,
         sharedMemSize / 1024.0f);

  // Use 1D grid - one block per tile per row
  dim3 blockSize(threadsPerBlock);
  dim3 gridSize(n * tiles_per_row);

  printf("Grid size:\t\t\t%d blocks (%d threads per block)\n",
         n * tiles_per_row, threadsPerBlock);

  // Allocate memory on GPU
  float* device_src = NULL;
  float* device_dst = NULL;
  size_t grid_size  = n * increment * sizeof(float);

  cudaError_t error;
  TIME_INIT();

  // Allocate and initialize device memory
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

  // Initialize with zeros to start
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

  // Handle special case for odd iteration count
  if (iters % 2 != 0) {
    iteration_gpu_tiled<<<gridSize, blockSize, sharedMemSize>>>(
        n, m, increment, device_dst, device_src, tile_size, tiles_per_row);
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

    // First iteration in the pair
    iteration_gpu_tiled<<<gridSize, blockSize, sharedMemSize>>>(
        n, m, increment, device_dst, device_src, tile_size, tiles_per_row);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "Error in iteration %d (first): %s\n", iter,
              cudaGetErrorString(error));
      break;
    }

    cudaDeviceSynchronize();

    // Second iteration in the pair
    iteration_gpu_tiled<<<gridSize, blockSize, sharedMemSize>>>(
        n, m, increment, device_src, device_dst, tile_size, tiles_per_row);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "Error in iteration %d (second): %s\n", iter,
              cudaGetErrorString(error));
      break;
    }

    cudaDeviceSynchronize();
  }

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
