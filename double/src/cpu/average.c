#include "../../include/cpu/average.h"

/**
 * @brief Calculates the average value for each row of a matrix.
 *
 * This function iterates through each row of the input matrix dst, sums the
 * first m elements of that row, and then divides by m to find the average. The
 * result for each row is stored in the averages array. It assumes the matrix is
 * stored row-major but uses an increment to handle potential padding between
 * rows.
 *
 * @param[in]  n         The number of rows in the matrix (i.e., height).
 * @param[in]  m         The number of columns to average in each row (i.e.,
 *                       width, excluding padding).
 * @param[in]  increment The distance (i.e., number of elements) between the
 *                       start of consecutive rows in the dst array (accountsfor
 *                       padding).
 * @param[in]  dst       Pointer to the input matrix data.
 * @param[out] averages  Pointer to the output array to store row averages.
 *
 * @return int Returns 0, indicating success.
 */
int average_rows(const int n, const int m, const int increment,
                 const double* const dst, double* const averages) {

  // Initialise averages to zero
  for (int i = 0; i < n; i++) {
    averages[i] = 0.0;
  }

  // Calculate sum for each row
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      averages[i] += dst[i * increment + j];
    }
  }

  // Divide by width to get the average
  for (int i = 0; i < n; i++) {
    averages[i] /= m;
  }

  return 0;
}
