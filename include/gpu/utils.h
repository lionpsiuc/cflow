#pragma once

#define INIT()                                                                 \
  cudaEvent_t start;                                                           \
  cudaEvent_t end;                                                             \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&end);                                                       \
  int index = 0

#define START() cudaEventRecord(start)

#define END()                                                                  \
  cudaEventRecord(end);                                                        \
  if (timings != NULL) {                                                       \
    cudaEventSynchronize(start);                                               \
    cudaEventSynchronize(end);                                                 \
    cudaEventElapsedTime(timings + index, start, end);                         \
    timings[index] /= 1000.0f;                                                 \
  }                                                                            \
  index++

#define COMPLETE()                                                             \
  cudaEventDestroy(start);                                                     \
  cudaEventDestroy(end)
