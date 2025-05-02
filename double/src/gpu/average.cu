#include <stdio.h>

#include "../../include/gpu/utils.h"

/**
 * @brief CUDA kernel to calculate the average of each row in parallel.
 *
 * Each block processes one row. Threads within a block cooperate to calculate
 * the sum of the first m elements of their assigned row using a parallel
 * reduction algorithm implemented with shared memory. Thread 0 of each block
 * writes the final average for its row.
 *
 * @param[in]  n         The total number of rows in the input matrix.
 * @param[in]  m         The number of columns to include in the average for
 *                       each row.
 * @param[in]  increment The stride (i.e., number of elements) between the start
 *                       of consecutive rows in the input array.
 * @param[in]  input     Device pointer to the input matrix data.
 * @param[out] averages  Device pointer to the output array where row averages
 *                       will be stored.
 */
__global__ void average_rows_kernel(const int n, const int m,
                                    const int increment,
                                    const float* __restrict__ input,
                                    float* __restrict__ averages) {
  const int row = blockIdx.x; // Map block index directly to the row index

  // Ensure the block index is within the valid number of rows
  if (row >= n)
    return; // Block does no work if its index is out of bounds

  const float* row_start =
      input + row * increment; // Calculate the starting address of the assigned
                               // row in global memory

  // Use shared memory for the reduction; the size is determined by
  // threadsPerBlock * sizeof(float) during launch
  extern __shared__ float sdata[];
  const int tid = threadIdx.x; // Get the thread index within the block

  // Each thread calculates partial sum for its elements
  float thread_sum = 0.0f;
  for (int j = tid; j < m; j += blockDim.x) {
    thread_sum += row_start[j]; // Accumulate values from global memory
  }

  // Store in shared memory
  sdata[tid] = thread_sum;

  __syncthreads(); // Synchronise all threads within the block to ensure all
                   // partial sums are written to shared memory

  // Perform a reduction operation within the shared memory array sdata; the
  // loop halves the number of active threads in each iteration
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) { // Only the first s threads participate in adding values
      sdata[tid] +=
          sdata[tid +
                s]; // Add value from the corresponding thread in the other half
    }
    __syncthreads(); // Synchronise after each step of the reduction to ensure
                     // correct partial sums are read
  }

  // Thread 0 of the block now holds the total sum for the row in sdata[0]; it
  // calculates the average and writes it to the global memory output array
  if (tid == 0) {
    averages[row] = sdata[0] / (float) m;
  }
}

/**
 * @brief Carries out the GPU calculation of row averages for a matrix.
 *
 * This host function sets up the configuration for the row_average_kernel,
 * allocates necessary device memory for the results, launches the kernel,
 * copies the results back to the host, and handles timing measurements. It
 * assumes the input matrix data is already present on the device.
 *
 * @param[in]  n             The number of rows in the matrix.
 * @param[in]  m             The number of columns to average per row.
 * @param[in]  increment     The stride between the start of consecutive rows i
 *                           device_input.
 * @param[in]  device_input  Device pointer to the input matrix data (assumed to
 *                           be already on GPU).
 * @param[out] host_averages Host pointer to the array where the calculated row
 *                           averages will be copied.
 * @param[out] timing        Host pointer to a float array (storing five values)
 *                           to store timing results for different stages.
 *
 * @return int Returns 0 on success, non-zero if dimension checks fail.
 */
extern "C" int average_rows_gpu(const int n, const int m, const int increment,
                                const float* device_input, float* host_averages,
                                float* timing) {

  // Define the block size for the kernel; this kernel uses one block per row,
  // so block size mainly affects the reduction performance
  int threadsPerBlock = 256;

  // Check if dimensions are divisible by block size before proceeding
  if (n % threadsPerBlock != 0) {
    fprintf(
        stderr,
        "  Error: Matrix height (i.e., n), which is %d, must be divisible by "
        "block size, which is %d\n",
        n, threadsPerBlock);
    return 1; // Indicate failure
  }
  if (m % threadsPerBlock != 0) {
    fprintf(
        stderr,
        "  Error: Matrix width (i.e., m), which is %d, must be divisible by "
        "block size, which is %d\n",
        m, threadsPerBlock);
    return 1; // Indicate failure
  }

  INIT(); // Initialise CUDA event timers

  // Start timing for setup
  START();

  int numBlocks = n; // Launch one block for each row

  // Calculate the required shared memory size per block for the reduction
  int sharedMemSize = threadsPerBlock * sizeof(float);

  END(); // Stop timing for setup (i.e., timing[0])

  // Start timing for allocation
  START();

  float* device_averages = NULL; // Pointer for device memory to store results

  // Allocate memory on the GPU for the output averages array
  cudaMalloc((void**) &device_averages, n * sizeof(float));

  // Initialise the allocated device memory to zeros to ensure clean results
  cudaMemset(device_averages, 0, n * sizeof(float));

  END(); // Stop timing for allocation (i.e., timing[1])

  // Start timing for host-to-device transfer; this section is timed but
  // intentionally left empty because the input matrix (i.e., device_input) is
  // assumed to be already on the device from the previous propagation step
  START();

  END(); // Stop timing for transfer (i.e., timing[2])

  // Start timing for computation
  START();

  // Launch the kernel on the GPU
  average_rows_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
      n, m, increment, device_input, device_averages);

  // Wait for the kernel to complete execution before proceeding
  cudaDeviceSynchronize();
  END(); // Stop timing for computation (i.e., timing[3])

  // Start timing for device-to-host transfer
  START();

  // Copy the results from the device memory (i.e., device_averages) back to the
  // host memory (i.e., host_averages)
  cudaMemcpy(host_averages, device_averages, n * sizeof(float),
             cudaMemcpyDeviceToHost);

  END(); // Stop timing for device-to-host transfer (i.e., timing[4])

  // Free the allocated device memory for the averages
  cudaFree(device_averages);

  COMPLETE(); // Destroy CUDA event timers
  return 0;
}
