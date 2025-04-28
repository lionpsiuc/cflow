#ifndef KERNEL_UTILS_CUH
#define KERNEL_UTILS_CUH

/**
 * @brief Integer division that rounds up. Returns 0 if numerator is negative.
 *        Used for calculating grid/block dimensions.
 *
 * @param numerator The number to be divided.
 * @param denominator The number to divide by.
 * @return The result of ceil(numerator / denominator), or 0 if numerator < 0.
 */
inline __host__ __device__ int divide_rounding_up(int numerator,
                                                  int denominator) {
  if (numerator < 0) {
    return 0;
  }

  // Avoid division by zero, though denominator should always be positive here
  if (denominator <= 0) {
    return 0;
  }
  return (numerator + denominator - 1) / denominator;
}

/**
 * @brief Basic maximum macro.
 */
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

/**
 * @brief Basic minimum macro.
 */
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#endif // KERNEL_UTILS_CUH
