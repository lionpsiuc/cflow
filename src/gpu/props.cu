#include "../../include/gpu/props.h"

/**
 * @brief Explain briefly.
 *
 * @return Explain briefly.
 */
int current_device(void) {
  int index = -1;
  if (cudaGetDevice(&index) != cudaSuccess) {
    return -1;
  }
  return index;
}

/**
 * @brief Explain briefly.
 *
 * @param index Explain briefly.
 *
 * @return Explain briefly.
 */
int compute_capability(int index) {
  if (index < 0) {
    index = current_device();
    if (index < 0)
      return -1;
  }
  int major_cc = -1;
  if (cudaDeviceGetAttribute(&major_cc, cudaDevAttrComputeCapabilityMajor,
                             index) != cudaSuccess) {
    return -1;
  }
  int minor_cc = -1;
  if (cudaDeviceGetAttribute(&minor_cc, cudaDevAttrComputeCapabilityMinor,
                             index) != cudaSuccess) {
    return -1;
  }
  return 100 * major_cc + 10 * minor_cc;
}

/**
 * @brief Explain briefly.
 *
 * @param index Explain briefly.
 *
 * @return Explain briefly.
 */
int max_grid_dim_x(int index) {
  if (index < 0) {
    index = current_device();
    if (index < 0)
      return -1;
  }
  int dim = -1;
  if (cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimX, index) !=
      cudaSuccess) {
    return -1;
  }
  return dim;
}

/**
 * @brief Explain briefly.
 *
 * @param index Explain briefly.
 *
 * @return Explain briefly.
 */
int max_shared_per_block(int index) {
  if (index < 0) {
    index = current_device();
    if (index < 0)
      return -1;
  }
  int max = -1;
  if (cudaDeviceGetAttribute(&max, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             index) != cudaSuccess) {
    return -1;
  }
  return max;
}

/**
 * @brief Explain briefly.
 *
 * @param index Explain briefly.
 *
 * @return Explain briefly.
 */
int multiprocessor_count(int index) {
  if (index < 0) {
    index = current_device();
    if (index < 0)
      return -1;
  }
  int sm_count = -1;
  if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                             index) != cudaSuccess) {
    return -1;
  }
  return sm_count;
}

/**
 * @brief Explain briefly.
 *
 * @param index Explain briefly.
 *
 * @return Explain briefly.
 */
int cuda_core_count(int index) {
  if (index < 0) {
    index = current_device();
    if (index < 0)
      return -1;
  }
  int sm_count = multiprocessor_count(index);
  if (sm_count < 0) {
    return -1;
  }
  int major_cc = -1;
  if (cudaDeviceGetAttribute(&major_cc, cudaDevAttrComputeCapabilityMajor,
                             index) != cudaSuccess) {
    return -1;
  }
  int minor_cc = -1;
  if (cudaDeviceGetAttribute(&minor_cc, cudaDevAttrComputeCapabilityMinor,
                             index) != cudaSuccess) {
    return -1;
  }
  int cc_per_sm = -1;
  switch (major_cc) {
    case 1: // Tesla, T10
      cc_per_sm = 8;
      break;
    case 2: // Fermi
      cc_per_sm = 32;
      break;
    case 3: // Kepler
      cc_per_sm = 192;
      break;
    case 5: // Maxwell
      cc_per_sm = 128;
      break;
    case 6: // Pascal
      switch (minor_cc) {
        case 0: // GP100 - 64 CUDA cores per SM
          cc_per_sm = 64;
          break;
        case 1: // GP102, GP104, GP106, and GP107 - 128 CUDA cores per SM
          cc_per_sm = 128;
          break;
        default: // Unknown
          cc_per_sm = -1;
          break;
      }
      break;
    case 7: // Volta is 7.0 and 7.2 - 64 CUDA cores per SM
      switch (minor_cc) {
        case 0: cc_per_sm = 64; break;
        case 2: cc_per_sm = 64; break;
        case 5: // Turing is 7.5 - 64 CUDA cores per SM
          cc_per_sm = 64;
          break;
        default: // Unknown
          cc_per_sm = -1;
          break;
      }
      break;
    case 8: // Ampere - 64 CUDA cores per SM
      cc_per_sm = 64;
      break;
    default: // Unknown
      cc_per_sm = -1;
      break;
  }
  return sm_count * cc_per_sm;
}

/**
 * @brief Explain briefly.
 *
 * @param index Explain briefly.
 *
 * @return Explain briefly.
 */
int set_device(const int index) {
  if (cudaSetDevice(index) != cudaSuccess) {
    return -1;
  }
  return 0;
}
