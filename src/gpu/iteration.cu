#include "../../include/gpu/utils.h"

/**
 * @brief Explain briefly.
 */
#define MAX_TILE_SIZE 4096

/**
 * @brief Explain briefly.
 *
 * @param n Explain briefly.
 * @param m Explain briefly.
 * @param increment Explain briefly.
 * @param grid Explain briefly.
 *
 * @return Explain briefly.
 */
__global__ void init_gpu(const int n, const int m, const int increment,
                         float* const grid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float col0 = 0.98f * (float) ((i + 1) * (i + 1)) / (float) (n * n);
  grid[i * increment + 0] = col0;
  for (int j = 1; j < m; j++) {
    grid[i * increment + j] =
        col0 * ((float) (m - j) * (m - j) / (float) (m * m));
  }
  grid[i * increment + m]     = grid[i * increment + 0];
  grid[i * increment + m + 1] = grid[i * increment + 1];
}

/**
 * @brief Explain briefly.
 *
 * @param n Explain briefly.
 * @param m Explain briefly.
 * @param increment Explain briefly.
 * @param dst Explain briefly.
 * @param src Explain briefly.
 * @param tile_size Explain briefly.
 * @param tiles_per_row Explain briefly.
 *
 * @return Explain briefly.
 */
__global__ void iteration_gpu_tiled(const int n, const int m,
                                    const int increment, float* const dst,
                                    const float* const src, const int tile_size,
                                    const int tiles_per_row) {
  int row_idx  = blockIdx.x / tiles_per_row;
  int tile_idx = blockIdx.x % tiles_per_row;
  if (row_idx >= n) {
    return;
  }
  const int               row_offset       = row_idx * increment;
  int                     tile_start       = tile_idx * tile_size + 1;
  int                     tile_end         = min(tile_start + tile_size, m);
  int                     actual_tile_size = tile_end - tile_start;
  extern __shared__ float sh_mem[];
  for (int i = threadIdx.x; i < actual_tile_size + 4; i += blockDim.x) {
    int global_col;
    if (i < 2) {
      global_col = tile_start - 2 + i;
      if (global_col < 0) {
        global_col += m;
      }
    } else if (i >= actual_tile_size + 2) {
      global_col = tile_end + (i - (actual_tile_size + 2));
      if (global_col >= m) {
        global_col -= m;
      }
    } else {
      global_col = tile_start + (i - 2);
    }
    sh_mem[i] = src[row_offset + global_col];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < actual_tile_size; i += blockDim.x) {
    int   global_col = tile_start + i;
    int   sh_idx     = i + 2;
    float result =
        ((1.60f * sh_mem[sh_idx - 2]) + (1.55f * sh_mem[sh_idx - 1]) +
         sh_mem[sh_idx] + (0.60f * sh_mem[sh_idx + 1]) +
         (0.25f * sh_mem[sh_idx + 2])) /
        5.0f;
    dst[row_offset + global_col] = result;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (tile_idx == 0) {
      dst[row_offset + 0] = src[row_offset + 0];
    }
    if (tile_end == m) {
      dst[row_offset + m]     = dst[row_offset + 0];
      dst[row_offset + m + 1] = dst[row_offset + 1];
    }
  }
}

/**
 * @brief Explain briefly.
 *
 * @param iters Explain briefly.
 * @param n Explain briefly.
 * @param m Explain briefly.
 * @param host_grid Explain briefly.
 * @param timing Explain briefly.
 * @param device_grid_out Explain briefly.
 *
 * @return Explain briefly.
 */
extern "C" void heat_propagation_gpu(const int iters, const int n, const int m,
                                     float* host_grid, float* timing,
                                     float** device_grid_out) {
  const int increment = m + 2;
  INIT();
  int       threadsPerBlock = 256;
  const int tile_size       = MAX_TILE_SIZE;
  const int tiles_per_row   = (m > 0) ? (m + tile_size - 1) / tile_size : 1;
  int       max_elements_in_tile = 0;
  for (int t = 0; t < tiles_per_row; ++t) {
    int current_tile_start  = t * tile_size + 1;
    int current_tile_end    = min(current_tile_start + tile_size, m);
    int current_actual_size = current_tile_end - current_tile_start;
    max_elements_in_tile = max(max_elements_in_tile, current_actual_size + 4);
  }
  int sharedMemSize = max_elements_in_tile * sizeof(float);
  printf(
      "\n    Configuration:           %d tiles per row, with a maximum of %d "
      "elements "
      "per tile\n",
      tiles_per_row, max_elements_in_tile);
  printf("    Shared memory per block: %d bytes (i.e., %.1f kB)\n",
         sharedMemSize, sharedMemSize / 1024.0f);
  dim3 blockSize(threadsPerBlock);
  dim3 gridSize(n * tiles_per_row);
  printf("    Grid size:               %d blocks, with %d threads per block\n",
         (int) gridSize.x, (int) blockSize.x);
  float* device_src   = NULL;
  float* device_dst   = NULL;
  float* device_final = NULL;
  float* device_temp  = NULL;
  size_t grid_bytes   = n * increment * sizeof(float);
  START();
  cudaMalloc((void**) &device_src, grid_bytes);
  cudaMalloc((void**) &device_dst, grid_bytes);
  END();
  START();
  int blocksForInit = (n + threadsPerBlock - 1) / threadsPerBlock;
  init_gpu<<<blocksForInit, threadsPerBlock>>>(n, m, increment, device_src);
  cudaMemcpy(device_dst, device_src, grid_bytes, cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  END();
  START();
  float* current_src = device_src;
  float* current_dst = device_dst;
  for (int iter = 0; iter < iters; iter++) {
    iteration_gpu_tiled<<<gridSize, blockSize, sharedMemSize>>>(
        n, m, increment, current_dst, current_src, tile_size, tiles_per_row);
    float* temp = current_src;
    current_src = current_dst;
    current_dst = temp;
  }
  cudaDeviceSynchronize();
  device_final = current_src;
  device_temp  = current_dst;
  END();
  START();
  if (host_grid != NULL) {
    cudaMemcpy(host_grid, device_final, grid_bytes, cudaMemcpyDeviceToHost);
  }
  END();
  *device_grid_out = device_final;
  cudaFree(device_temp);
  COMPLETE();
}
