#include "../../include/cpu/iteration.h"

/**
 * @brief Initialises the grid with specific starting values and padding.
 *
 * Calculates initial values for each cell based on its row and column index
 * according to the predefined formula. It also sets the two extra columns at
 * the end of each row (padding) for boundary handling.
 *
 * @param[in]  n    The number of rows in the matrix (i.e., height).
 * @param[in]  m    The number of columns in the matrix (i.e., width, excluding
 *                  padding).
 * @param[out] grid Pointer to the matrix data array to be initialised.
 */
static void init(const int n, const int m, float* const grid) {
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
 * @brief Performs a single stencil iteration for one row.
 *
 * Calculates the next state for each element in the destination row dst based
 * on the five-point stencil applied to the source row src. Handles wrap-around
 * boundary conditions using the padded columns in src. Updates the padding
 * columns in dst.
 *
 * @param[in]  m   The width of the row (excluding padding).
 * @param[out] dst Pointer to the destination row array (m + 2 elements).
 * @param[in]  src Pointer to the source row array (m + 2 elements).
 */
static void iteration(const int m, float* const restrict dst,
                      const float* const restrict src) {

  // Here, we deal with the column at j = 1, since it also wraps around
  {
    int j = 1;

    // Left neighbours
    float old_l2 = src[m - 1];
    float old_l1 = src[0];

    // Right neighbours
    float old_r1 = src[j + 1];
    float old_r2 = src[j + 2];

    dst[j] = ((1.60f * old_l2) + (1.55f * old_l1) + src[j] + (0.60f * old_r1) +
              (0.25f * old_r2)) /
             5.0f;
  }

  // Other updates
  for (int j = 2; j < m; j++) {
    dst[j] = ((1.60f * src[j - 2]) + (1.55f * src[j - 1]) + src[j] +
              (0.60f * src[j + 1]) + (0.25f * src[j + 2])) /
             5.0f;
  }

  // Refresh our extra columns for further iterations
  dst[0]     = src[0];
  dst[m]     = dst[0];
  dst[m + 1] = dst[1];
}

/**
 * @brief Performs multiple stencil iterations on a single row.
 *
 * Applies the iteration function iters times to the given row data. Uses dst
 * and src as ping-pong buffers, swapping their roles in consecutive iterations.
 * The final result will be in dst if iters is even, and in src if iters is odd.
 *
 * @param[in]     iters The total number of iterations to perform.
 * @param[in]     m     The width of the row (excluding padding).
 * @param[in,out] dst   Pointer to one buffer for the row data.
 * @param[in,out] src   Pointer to the other buffer for the row data.
 */
static void iterations(const int iters, const int m, float* restrict dst,
                       float* restrict src) {
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
 * @brief Performs heat propagation simulation on the entire grid.
 *
 * Initialises the source (i.e., src) and destination (i.e., dst) grids. Then,
 * for each row, performs the specified number of iterations using the
 * iterations function, which handles the computation and buffer swapping for
 * that row.
 *
 * @param[in]     iters The total number of iterations to perform per row.
 * @param[in]     n     The number of rows in the matrix (i.e., height).
 * @param[in]     m     The number of columns in the matrix (i.e., width,
 *                      excluding padding).
 * @param[in,out] dst   Pointer to one buffer for the matrix data.
 * @param[in,out] src   Pointer to the other buffer for the matrix data.
 */
void heat_propagation(const int iters, const int n, const int m,
                      float* restrict dst, float* restrict src) {
  const int increment = m + 2;

  // Set initial conditions
  init(n, m, dst);
  init(n, m, src);

  for (int i = 0; i < n; i++) {
    float* const row_dst = dst + i * increment;
    float* const row_src = src + i * increment;
    iterations(iters, m, row_dst, row_src); // Perform the iterations
  }
  return;
}
