#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Explain briefly.
 *
 * @param n Explain briefly.
 * @param m Explain briefly.
 * @param increment Explain briefly.
 * @param device_input Explain briefly.
 * @param host_averages Explain briefly.
 * @param timing Explain briefly.
 *
 * @return Explain briefly.
 */
void average_rows_gpu(const int n, const int m, const int increment,
                      const float* device_input, float* host_averages,
                      float* timing);

#ifdef __cplusplus
}
#endif
