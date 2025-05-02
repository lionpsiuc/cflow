#pragma once

// #include "../precision.h"

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
 *                       width).
 * @param[in]  increment The distance (i.e., number of elements) between the
 *                       start of consecutive rows in the dst array (accountsfor
 *                       padding).
 * @param[in]  dst       Pointer to the input matrix data.
 * @param[out] averages  Pointer to the output array to store row averages.
 *
 * @return int Returns 0, indicating success.
 */
int average_rows(const int n, const int m, const int increment,
                 const double* const dst, double* const averages);
