#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Carries out the GPU heat propagation simulation over multiple
 *        iterations.
 *
 * This host function manages the overall GPU computation. It works as follows:
 *
 *   1. Checks if matrix dimensions are compatible with the block size.
 *   2. Allocates device memory for source and destination grids.
 *   3. Initialises the grid data directly on the GPU using the init_gpu kernel.
 *   4. Performs the specified number of iterations by repeatedly launching the
 *      propagation_kernel_tiled kernel, swapping source and destination
 *      pointers (i.e., ping-pong buffering).
 *   5. Optionally copies the final result back to the host.
 *   6. Returns a pointer to the final result grid on the device.
 *
 * @param[in]  iters            The number of simulation iterations to perform.
 * @param[in]  n                The number of rows in the matrix (i.e., height).
 * @param[in]  m                The number of columns (i.e., width), excluding
 *                              padding.
 * @param[out] host_grid        Host pointer where the final grid can be copied
 *                              back.
 * @param[out] timing           Host pointer to a float array (storing four
 *                              values) to store timing results.
 * @param[out] device_grid_out  Host pointer to store the device pointer of the
 *                              final result grid.
 *
 * @return int Returns 0 on success, non-zero if dimension checks fail.
 */
int heat_propagation_gpu(const int iters, const int n, const int m,
                         float* host_grid, float* timing,
                         float** device_grid_out);

#ifdef __cplusplus
}
#endif
