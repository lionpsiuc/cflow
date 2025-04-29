#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Explain briefly.
 *
 * @param iters Explain briefly.
 * @param n Explain briefly.
 * @param m Explain briefly.
 * @param host_grid Explain briefly.
 * @param timing Explain briefly.
 * @param device_grid_out Explain briefly.
 *
 * @return Explain briefly.
 */
void heat_propagation_gpu(const int iters, const int n, const int m,
                          float* host_grid, float* timing,
                          float** device_grid_out);

#ifdef __cplusplus
}
#endif
