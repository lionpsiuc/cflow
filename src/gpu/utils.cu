#include <stdio.h>

/**
 * @brief Frees memory previously allocated on the CUDA device.
 *
 * A wrapper around cudaFree to allow calling from C code.
 *
 * @param[in] devptr Pointer to the device memory to free.
 */
extern "C" void freedeviceptr(void* devptr) {
  if (devptr != NULL) {
    cudaFree(devptr);
  }
}
