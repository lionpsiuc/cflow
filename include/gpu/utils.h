#pragma once

// #define INIT()                                                                 \
//   cudaEvent_t start;                                                           \
//   cudaEvent_t end;                                                             \
//   cudaEventCreate(&start);                                                     \
//   cudaEventCreate(&end);                                                       \
//   int index = 0

#define INIT()                                                                 \
  cudaEvent_t start;                                                           \
  cudaEvent_t end;                                                             \
  int         event_error_flag = 0;                                            \
  if (cudaEventCreate(&start) != cudaSuccess ||                                \
      cudaEventCreate(&end) != cudaSuccess) {                                  \
    event_error_flag = 1;                                                      \
    fprintf(stderr, "ERROR: Failed to create CUDA events.\n");                 \
  }                                                                            \
  int index = 0

#define START() cudaEventRecord(start)

#define END()                                                                  \
  cudaEventRecord(end);                                                        \
  if (timing != NULL) {                                                        \
    cudaEventSynchronize(start);                                               \
    cudaEventSynchronize(end);                                                 \
    cudaEventElapsedTime(timing + index, start, end);                          \
    timing[index] /= 1000.0f;                                                  \
  }                                                                            \
  index++

#define COMPLETE()                                                             \
  cudaEventDestroy(start);                                                     \
  cudaEventDestroy(end)

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s in %s at line %d\n",                     \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
