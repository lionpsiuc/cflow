#include <stdio.h>
#include <stlib.h>

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
__global__ void init(const int n, const int m, const int increment,
                     float* const grid) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if the thread is within bounds
  if (i < n && j < m) {
    float col0 = 0.98f * (float) ((i + 1) * (i + 1)) / (float) (n * n);
    if (j == 0) {
      grid[i * increment + j]     = col0; // First column
      grid[i * increment + m + 0] = col0; // Set ghost column
    } else {

      // Interior points
      grid[i * increment + j] =
          col0 * ((float) (m - j) * (m - j) / (float) (m * m));

      // Moreover, set the other ghost column
      if (j == 1) {
        grid[i * increment + m + 1] = grid[i * increment + j];
      }
    }
  }
}
