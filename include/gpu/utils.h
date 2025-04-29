#pragma once

#include <stdio.h>

#define INIT()                                                                 \
  cudaEvent_t start;                                                           \
  cudaEvent_t end;                                                             \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&end);                                                       \
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
