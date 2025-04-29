#pragma once

#include <stdio.h>

/**
 * @brief Explain briefly.
 */
#define INIT()                                                                 \
  cudaEvent_t start;                                                           \
  cudaEvent_t end;                                                             \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&end);                                                       \
  int index = 0

/**
 * @brief Explain briefly.
 */
#define START() cudaEventRecord(start)

/**
 * @brief Explain briefly.
 */
#define END()                                                                  \
  cudaEventRecord(end);                                                        \
  if (timing != NULL) {                                                        \
    cudaEventSynchronize(start);                                               \
    cudaEventSynchronize(end);                                                 \
    cudaEventElapsedTime(timing + index, start, end);                          \
    timing[index] /= 1000.0f;                                                  \
  }                                                                            \
  index++

/**
 * @brief Explain briefly.
 */
#define COMPLETE()                                                             \
  cudaEventDestroy(start);                                                     \
  cudaEventDestroy(end)
