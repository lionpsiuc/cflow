#include "gpu_utils.h"

/**
 * @brief Gets the index of the currently selected CUDA device; helper function
 *        used internally.
 *
 * @return Device index, or -1 on error.
 */
static int get_current_device_index(void) {
  int         device_index = -1;
  cudaError_t error        = cudaGetDevice(&device_index);
  if (error != cudaSuccess) {
    return -1;
  }
  return device_index;
}

/**
 * @brief Gets the maximum shared memory per block usable by the currently
 *        selected device. Taken from your sample code.
 *
 * @return Maximum shared memory in bytes, or negative on error.
 */
int sm_per_block(void) {
  int device_index = get_current_device_index();
  if (device_index < 0) {
    return -1;
  }
  int         max_shared_bytes = -1;
  cudaError_t error            = cudaDeviceGetAttribute(
      &max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_index);
  if (error != cudaSuccess ||
      max_shared_bytes <= 0) { // If cudaDevAttrMaxSharedMemoryPerBlockOptin
                               // fails or isn't supported/meaningful
    error = cudaDeviceGetAttribute(
        &max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device_index);
  }
  if (error != cudaSuccess) {
    return -1;
  }
  return max_shared_bytes;
}

/**
 * @brief Gets the compute capability of the currently selected device.
 *
 * @return Compute capability value, or negative on error.
 */
int current_cc(void) {
  int device_index = get_current_device_index();
  if (device_index < 0) {
    return -1;
  }
  int         major_cc = -1;
  cudaError_t error    = cudaDeviceGetAttribute(
      &major_cc, cudaDevAttrComputeCapabilityMajor, device_index);
  if (error != cudaSuccess) {
    return -1;
  }
  int minor_cc = -1;
  error = cudaDeviceGetAttribute(&minor_cc, cudaDevAttrComputeCapabilityMinor,
                                 device_index);
  if (error != cudaSuccess) {
    return -1;
  }
  return major_cc * 100 + minor_cc * 10;
}
