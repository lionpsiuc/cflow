#include "../../include/gpu/utils.h"

/**
 * @brief Defines the fixed width of tiles used for processing rows. This
 *        determines how many columns are processed by a block in one go (before
 *        the addition of halo columns).
 */
#define MAX_TILE_SIZE 4096

/**
 * @brief CUDA kernel to initialise the grid directly on the GPU.
 *
 * Each thread calculates the initial value for one row element based on its row
 * and column index according to the predefined formula. It also sets the two
 * extra padding columns at the end of each row for boundary handling. Launched
 * with a 1D grid where threads map linearly to rows.
 *
 * @param[in]  n         The number of rows in the matrix (i.e., height).
 * @param[in]  m         The number of columns in the matrix (i.e., width,
 *                       excluding padding).
 * @param[in]  increment The stride (i.e., number of elements) between the start
 *                       of consecutive rows.
 * @param[out] grid      Device pointer to the matrix data array to be
 *                       initialised.
 */
__global__ void init_gpu(const int n, const int m, const int increment,
                         float* const grid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) // Ensure the calculated row index is within the matrix height
    return;

  // Calculate the value for the first column (i.e., j = 0)
  float col0 = 0.98f * (float) ((i + 1) * (i + 1)) / (float) (n * n);

  grid[i * increment + 0] = col0; // Write to global memory

  // Calculate values for the interior columns (i.e., j = 1 to m - 1)
  for (int j = 1; j < m; j++) {
    grid[i * increment + j] =
        col0 *
        ((float) (m - j) * (m - j) / (float) (m * m)); // Write to global memory
  }

  // Set the padded columns at the end of the row for wrap-around boundary
  // conditions
  grid[i * increment + m]     = grid[i * increment + 0];
  grid[i * increment + m + 1] = grid[i * increment + 1];
}

/**
 * @brief CUDA kernel to perform one iteration of the stencil computation using
 *        tiling.
 *
 * Each block processes one horizontal tile of a specific row. Threads within
 * the block cooperate to load the necessary input data (tile + halo) from
 * global memory (src) into shared memory (sh_mem). After synchronisation,
 * threads compute the stencil for elements within their tile using the data
 * from shared memory and write the results to global memory (dst). Handles
 * wrap-around boundary conditions using modulo arithmetic when loading halo
 * elements. Thread 0 handles copying the first column value and updating
 * padding columns if needed.
 *
 * @param[in]  n             The total number of rows in the matrix.
 * @param[in]  m             The total number of columns (i.e., width) in the
 *                           matrix excluding padding.
 * @param[in]  increment     The stride (i.e., number of elements) between the
 *                           start of consecutive rows.
 * @param[out] dst           Device pointer to the destination matrix data for
 *                           the current iteration.
 * @param[in]  src           Device pointer to the source matrix data from the
 *                           previous iteration.
 * @param[in]  tile_size     The target width of each tile; note that the actual
 *                           width might be smaller for the last tile.
 * @param[in]  tiles_per_row The number of tiles needed to cover the width m.
 */
__global__ void iteration_gpu_tiled(const int n, const int m,
                                    const int increment, float* const dst,
                                    const float* const src, const int tile_size,
                                    const int tiles_per_row) {
  int row_idx =
      blockIdx.x /
      tiles_per_row; // Calculate the row index this block is responsible for
  int tile_idx =
      blockIdx.x % tiles_per_row; // Calculate the tile index within the row
                                  // this block is responsible for
  if (row_idx >= n) { // Exit if the block's row index is out of bounds
    return;
  }
  const int row_offset =
      row_idx * increment; // Calculate the starting global memory offset for
                           // the assigned row
  int tile_start = tile_idx * tile_size + 1; // Calculate the starting column
                                             // index (inclusive) for this tile
  int tile_end = min(tile_start + tile_size,
                     m); // Calculate the ending column index (exclusive) for
                         // this tile, clamping to matrix width m
  int actual_tile_size =
      tile_end -
      tile_start; // Calculate the actual number of elements this tile covers

  // Declare shared memory; size is determined dynamically at launch based on
  // max_elements_in_tile
  extern __shared__ float sh_mem[];

  // Load data from global to shared memory
  for (int i = threadIdx.x; i < actual_tile_size + 4; i += blockDim.x) {
    int global_col; // The column index in global memory to load from

    // Calculate global_col index, handling wrap-around for halo elements
    if (i < 2) { // Loading left halo elements
      global_col = tile_start - 2 + i;
      if (global_col < 0) {
        global_col += m;
      }
    } else if (i >= actual_tile_size + 2) { // Loading right halo elements
      global_col = tile_end + (i - (actual_tile_size + 2));
      if (global_col >= m) {
        global_col -= m;
      }
    } else { // Loading elements within the actual tile
      global_col = tile_start + (i - 2); // Adjust index relative to tile_start
    }

    // Load the value from global memory (src) into the correct shared memory
    // position
    sh_mem[i] = src[row_offset + global_col];
  }

  // Synchronise threads within the block to ensure all data is loaded into
  // shared memory before any thread starts computation
  __syncthreads();

  // Compute stencil using shared memory
  for (int i = threadIdx.x; i < actual_tile_size; i += blockDim.x) {

    // Calculate the global column index corresponding to the current element i
    // within the tile
    int global_col = tile_start + i;

    // Calculate the index within shared memory corresponding to the current
    // element; '+ 2' because shared memory includes the left halo
    int sh_idx = i + 2;

    float result =
        ((1.60f * sh_mem[sh_idx - 2]) + (1.55f * sh_mem[sh_idx - 1]) +
         sh_mem[sh_idx] + (0.60f * sh_mem[sh_idx + 1]) +
         (0.25f * sh_mem[sh_idx + 2])) /
        5.0f; // Apply the five-point stencil using values from shared memory
    dst[row_offset + global_col] = result; // Write the computed result to the
                                           // destination grid in global memory
  }

  __syncthreads(); // Synchronise threads before updating padding columns to
                   // ensure all stencil calculations for the tile are complete

  // Handle the initial column
  if (threadIdx.x == 0) {
    if (tile_idx == 0) {
      dst[row_offset + 0] =
          src[row_offset + 0]; // Copy the value of the column directly from src
                               // to dst as it's not computed by the stencil
    }

    // If this is the last tile in the row...
    if (tile_end == m) {

      // Update the padding columns in the destination grid based on the newly
      // computed values
      dst[row_offset + m]     = dst[row_offset + 0];
      dst[row_offset + m + 1] = dst[row_offset + 1];
    }
  }
}

/**
 * @brief Carries out the GPU heat propagation simulation over multiple
 *        iterations.
 *
 * This host function manages the overall GPU computation. It works as follows:
 *
 *   1. Checks if matrix dimensions are compatible with the block size.
 *   2. Allocates device memory for source and destination grids.
 *   3. Initialises the grid data directly on the GPU using the init_gpu kernel.
 *   4. Performs the specified number of iterations by repeatedly launching the
 *      propagation_kernel_tiled kernel, swapping source and destination
 *      pointers (i.e., ping-pong buffering).
 *   5. Optionally copies the final result back to the host.
 *   6. Returns a pointer to the final result grid on the device.
 *
 * @param[in]  iters            The number of simulation iterations to perform.
 * @param[in]  n                The number of rows in the matrix (i.e., height).
 * @param[in]  m                The number of columns (i.e., width), excluding
 *                              padding.
 * @param[out] host_grid        Host pointer where the final grid can be copied
 *                              back.
 * @param[out] timing           Host pointer to a float array (storing four
 *                              values) to store timing results.
 * @param[out] device_grid_out  Host pointer to store the device pointer of the
 *                              final result grid.
 *
 * @return int Returns 0 on success, non-zero if dimension checks fail.
 */
extern "C" int heat_propagation_gpu(const int iters, const int n, const int m,
                                    float* host_grid, float* timing,
                                    float** device_grid_out) {

  // Calculate the increment (i.e., stride) between rows, including padding
  const int increment = m + 2;

  int threadsPerBlock =
      256; // Define the block size used in this function's kernels

  // Check if dimensions are divisible by block size before proceeding
  if (n % threadsPerBlock != 0) {
    fprintf(stderr,
            "Error: Matrix height (i.e., n), which is %d, must be divisible by "
            "block size, which is %d\n",
            n, threadsPerBlock);
    return 1; // Indicate failure
  }
  if (m % threadsPerBlock != 0) {
    fprintf(stderr,
            "Error: Matrix width (i.e., m), which is %d, must be divisible by "
            "block size, which is %d\n",
            m, threadsPerBlock);
    return 1; // Indicate failure
  }

  INIT(); // Initialise CUDA event timers

  // Calculate tiling and shared memory configuration
  const int tile_size = MAX_TILE_SIZE;

  // Calculate how many tiles are needed horizontally to cover m columns
  const int tiles_per_row = (m > 0) ? (m + tile_size - 1) / tile_size : 1;

  // Calculate the maximum number of elements any tile needs in shared memory,
  // including the four halo elements, which in turn determines required shared
  // memory size
  int max_elements_in_tile = 0;
  for (int t = 0; t < tiles_per_row; ++t) {
    int current_tile_start  = t * tile_size + 1;
    int current_tile_end    = min(current_tile_start + tile_size, m);
    int current_actual_size = current_tile_end - current_tile_start;
    max_elements_in_tile =
        max(max_elements_in_tile,
            current_actual_size +
                4); // Need space for actual_size plus the four halo elements
  }

  // Calculate the required shared memory size in bytes
  int sharedMemSize = max_elements_in_tile * sizeof(float);

  // Define kernel launch parameters
  dim3 blockSize(threadsPerBlock); // Block dimension
  dim3 gridSize(
      n * tiles_per_row); // Grid dimension with one block per tile per row

  // Print configuration details
  printf(
      "\n    Configuration:           %d tiles per row, with a maximum of %d "
      "elements "
      "per tile\n",
      tiles_per_row, max_elements_in_tile);
  printf("    Shared memory per block: %d bytes (i.e., %.1f kB)\n",
         sharedMemSize, sharedMemSize / 1024.0f);
  printf("    Grid size:               %d blocks, with %d threads per block\n",
         (int) gridSize.x, (int) blockSize.x);

  // Pointers for device memory grids
  float* device_src   = NULL; // Source grid for an iteration
  float* device_dst   = NULL; // Destination grid for an iteration
  float* device_final = NULL; // Points to the grid holding the final result
  float* device_temp  = NULL; // Temporary pointer used for cleanup

  // Calculate total bytes needed for one grid
  size_t grid_bytes = n * increment * sizeof(float);

  // Start timing for allocation
  START();

  // Allocate memory on the device for the source and destination grids
  cudaMalloc((void**) &device_src, grid_bytes);
  cudaMalloc((void**) &device_dst, grid_bytes);

  END(); // Stop timing for allocation (i.e., timing[0])

  // Start timing initialisation
  START();

  // Calculate grid size needed for the init_gpu kernel which covers n rows
  int blocksForInit = (n + threadsPerBlock - 1) / threadsPerBlock;

  // Launch the kernel to initialise device_src directly on the GPU
  init_gpu<<<blocksForInit, threadsPerBlock>>>(n, m, increment, device_src);

  // Copy the initialised data from device_src to device_dst; both grids need
  // the initial state for the ping-pong buffering.
  cudaMemcpy(device_dst, device_src, grid_bytes, cudaMemcpyDeviceToDevice);

  // Ensure initialisation is complete before starting iterations
  cudaDeviceSynchronize();

  END(); // Stop timing initialisation (i.e., timing[1])

  // Start timing computation loop
  START();

  // Set up pointers for ping-pong buffering
  float* current_src = device_src;
  float* current_dst = device_dst;

  // Perform the main iteration loop
  for (int iter = 0; iter < iters; iter++) {
    iteration_gpu_tiled<<<gridSize, blockSize, sharedMemSize>>>(
        n, m, increment, current_dst, current_src, tile_size, tiles_per_row);
    // Check for kernel errors (optional, but useful for debugging).
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) printf("Kernel launch error: %s\n",
    // cudaGetErrorString(err));

    // Swap the roles of src and dst pointers for the next iteration.
    float* temp = current_src;
    current_src = current_dst;
    current_dst = temp;
  }

  cudaDeviceSynchronize();    // Wait for all iterations to complete
  device_final = current_src; // After the loop, current_src holds the pointer
                              // to the final result grid
  device_temp = current_dst;  // current_dst points to the other buffer which is
                              // now temporary
  END();                      // Stop timing computation loop (i.e., timing[2])

  // Start timing device-to-host transfer
  START();

  // If a host_grid pointer was provided, copy the final result back
  if (host_grid != NULL) {
    cudaMemcpy(host_grid, device_final, grid_bytes, cudaMemcpyDeviceToHost);
  }

  END(); // Stop timing device-to-host transfer (i.e., timing[3])

  // Set the output pointer to the final device grid
  *device_grid_out = device_final;

  // Free the temporary device buffer
  cudaFree(device_temp);

  COMPLETE(); // Destroy CUDA event timers
  return 0;
}
