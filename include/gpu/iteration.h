#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void heat_propagation_gpu(const int iters, const int n, const int m,
                          float* host_grid, float* timing,
                          float** device_grid_out);

#ifdef __cplusplus
}
#endif
