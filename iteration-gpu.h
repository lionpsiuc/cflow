#ifndef ITERATION_GPU_H
#define ITERATION_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

extern void heat_propagation_gpu(const int iters, const int n, const int m,
                                 float* host_grid, float* timing);

#endif
