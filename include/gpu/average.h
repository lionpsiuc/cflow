#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int average_rows_gpu(const int n, const int m, const int increment,
                     const float* device_input, float* host_averages,
                     float* timing);

#ifdef __cplusplus
}
#endif
