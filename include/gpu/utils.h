#pragma once

#include <stdio.h>

/**
 * @brief Initialises CUDA event timers and a timing array index.
 *
 * This macro should be called once at the beginning of a function where timing
 * is needed.
 */
#define INIT()                                                                 \
  cudaEvent_t start;                                                           \
  cudaEvent_t end;                                                             \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&end);                                                       \
  int index = 0

/**
 * @brief Records the start CUDA event in the default stream.
 *
 * Marks the beginning of a code section to be timed. Assumes start event has
 * been created by INIT.
 */
#define START() cudaEventRecord(start)

/**
 * @brief Records the end CUDA event, calculates elapsed time, and stores it.
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
 * @brief Destroys the CUDA event objects created by INIT.
 *
 * This should be called once at the end of the function where INIT was called.
 */
#define COMPLETE()                                                             \
  cudaEventDestroy(start);                                                     \
  cudaEventDestroy(end)

/**
 * @brief Frees memory previously allocated on the CUDA device.
 *
 * A wrapper around cudaFree to allow calling from C code.
 *
 * @param[in] devptr Pointer to the device memory to free.
 */
void freedeviceptr(void* devptr);
