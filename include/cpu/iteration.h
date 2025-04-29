#pragma once

// #include "../precision.h"

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
                      float* restrict dst, float* restrict src);
