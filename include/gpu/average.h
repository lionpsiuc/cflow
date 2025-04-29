#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Carries out the GPU calculation of row averages for a matrix.
 *
 * This host function sets up the configuration for the row_average_kernel,
 * allocates necessary device memory for the results, launches the kernel,
 * copies the results back to the host, and handles timing measurements. It
 * assumes the input matrix data is already present on the device.
 *
 * @param[in]  n             The number of rows in the matrix.
 * @param[in]  m             The number of columns to average per row.
 * @param[in]  increment     The stride between the start of consecutive rows i
 *                           device_input.
 * @param[in]  device_input  Device pointer to the input matrix data (assumed to
 *                           be already on GPU).
 * @param[out] host_averages Host pointer to the array where the calculated row
 *                           averages will be copied.
 * @param[out] timing        Host pointer to a float array (storing five values)
 *                           to store timing results for different stages.
 *
 * @return int Returns 0 on success, non-zero if dimension checks fail.
 */
int average_rows_gpu(const int n, const int m, const int increment,
                     const float* device_input, float* host_averages,
                     float* timing);

#ifdef __cplusplus
}
#endif
