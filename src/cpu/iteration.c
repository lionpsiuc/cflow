#include "../../include/iteration.h"

/**
 * @brief Explain briefly.
 *
 * @param n Explain briefly.
 * @param m Explain briefly.
 * @param grid Explain briefly.
 *
 * @return Explain briefly.
 */
static void init(const int n, const int m, PRECISION* const grid) {
  const int increment = m + 2; // We need to skip the last two columns since
                               // they are copies of the first and second column

  // Loop over rows
  for (int i = 0; i < n; i++) {
    float col0 =
        0.98f * (float) ((i + 1) * (i + 1)) /
        (float) (n * n); // Set it as a variable to make initialisation easier
    grid[i * increment + 0] = col0; // We have a '+ 0' here for consistency

    // Interior points
    for (int j = 1; j < m; j++) {
      grid[i * increment + j] =
          col0 * ((float) (m - j) * (m - j) / (float) (m * m));
    }

    // Extra columns
    grid[i * increment + m + 0] = grid[i * increment + 0];
    grid[i * increment + m + 1] = grid[i * increment + 1];
  }

  return;
}

/**
 * @brief Explain briefly.
 *
 * @param m Explain briefly.
 * @param dst Explain briefly.
 * @param src Explain briefly.
 *
 * @return Explain briefly.
 */
static void iteration(const int m, PRECISION* const restrict dst,
                      const PRECISION* const restrict src) {

  // Here, we deal with the column at j = 1, since it also wraps around
  {
    int j = 1;

    // Left neighbours
    PRECISION old_l2 = src[m - 1];
    PRECISION old_l1 = src[0];

    // Right neighbours
    PRECISION old_r1 = src[j + 1];
    PRECISION old_r2 = src[j + 2];

#ifdef double
    dst[j] = ((1.60 * old_l2) + (1.55 * old_l1) + src[j] + (0.60 * old_r1) +
              (0.25 * old_r2)) /
             5.0;
#else
    dst[j] = ((1.60f * old_l2) + (1.55f * old_l1) + src[j] + (0.60f * old_r1) +
              (0.25f * old_r2)) /
             5.0f;
#endif
  }

  // Other updates
  for (int j = 2; j < m; j++) {
#ifdef double
    dst[j] = ((1.60 * src[j - 2]) + (1.55 * src[j - 1]) + src[j] +
              (0.60 * src[j + 1]) + (0.25 * src[j + 2])) /
             5.0;
#else
    dst[j] = ((1.60f * src[j - 2]) + (1.55f * src[j - 1]) + src[j] +
              (0.60f * src[j + 1]) + (0.25f * src[j + 2])) /
             5.0f;
#endif
  }

  // Refresh our extra columns for further iterations
  dst[0]     = src[0];
  dst[m]     = dst[0];
  dst[m + 1] = dst[1];
}

/**
 * @brief Explain briefly.
 *
 * @param iters Explain briefly.
 * @param m Explain briefly.
 * @param dst Explain briefly.
 * @param src Explain briefly.
 *
 * @return Explain briefly.
 */
static void iterations(const int iters, const int m, PRECISION* restrict dst,
                       PRECISION* restrict src) {
  if (iters % 2 != 0) {
    iteration(m, dst, src);
  }
  for (int iter = 0; iter < iters / 2; iter++) {
    iteration(m, src, dst);
    iteration(m, dst, src);
  }
  return;
}

/**
 * @brief Explain briefly.
 *
 * @param iters Explain briefly.
 * @param n Explain briefly.
 * @param m Explain briefly.
 * @param dst Explain briefly.
 * @param src Explain briefly.
 *
 * @return Explain briefly.
 */
void heat_propagation(const int iters, const int n, const int m,
                      PRECISION* restrict dst, PRECISION* restrict src) {
  const int increment = m + 2;

  // Set initial conditions
  init(n, m, dst);
  init(n, m, src);

  for (int i = 0; i < n; i++) {
    PRECISION* const row_dst = dst + i * increment;
    PRECISION* const row_src = src + i * increment;
    iterations(iters, m, row_dst, row_src); // Perform the iterations
  }
  return;
}
