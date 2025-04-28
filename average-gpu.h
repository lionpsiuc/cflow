#ifndef AVERAGING_GPU_H
#define AVERAGING_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

extern void average_rows_gpu(const int n, const int m, const int increment,
                             const float* host_input, float* host_averages,
                             float* timing);

#ifdef __cplusplus
}
#endif

#endif // AVERAGING_GPU_H
