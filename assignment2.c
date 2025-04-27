#include <stdio.h>
#include <stdlib.h>

#include "average.h"
#include "iteration-gpu.h"
#include "iteration.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  arguments  args      = parse(argc, argv); // Parse command-line arguments
  const int  n         = args.n;
  const int  m         = args.m;
  const int  increment = m + 2;
  const int  iters     = args.iters;
  const bool average   = args.average;
  const bool cpu       = args.cpu;
  const bool timing    = args.timing;

  // Print parameters used
  printf("\nMatrix:\t\t%d x %d (%d iterations)\n", n, n, iters);
  printf("Precision:\t32-bit float\n\n");

  // Allocate memory on the host
  float* const averages     = calloc(n, sizeof(float));
  float* const averages_gpu = calloc(n, sizeof(float)); // Not used yet
  float* const dst          = calloc(n * increment, sizeof(float));
  float* const dst_gpu      = calloc(n * increment, sizeof(float));
  float* const src          = calloc(n * increment, sizeof(float));
  if (averages == NULL || averages_gpu == NULL || dst == NULL ||
      dst_gpu == NULL || src == NULL) {
    fprintf(stderr, "Failed to allocate memory on host\n");
    exit(EXIT_FAILURE);
  }

  // Timing on the CPU
  double start_time   = get_current_time();
  double cpu_times[2] = {0}; // Array to store timing results
  int    timing_index = 0;

  // Iterations on the CPU
  if (!cpu) {
    printf("CPU\n");
    printf("Performing %d iterations...\n", iters);
    heat_propagation(iters, n, m, dst, src);
    cpu_times[timing_index++] = get_duration(&start_time);

    if (timing) {
      printf("Iterations completed in %.6f seconds\n", cpu_times[0]);
    }
    // Averages, if requested
    if (average) {
      printf("Calculating row averages...\n");

      // Calculate averages
      start_time = get_current_time();
      average_rows(m, m, increment, dst, averages);
      cpu_times[timing_index++] = get_duration(&start_time);

      if (timing) {
        printf("Row averages calculated in %.6f seconds\n", cpu_times[1]);
      }
    }

    printf("Completed the work on the CPU\n\n");
  }
  // print_matrix(n, m, increment, dst);

  // GPU test
  printf("GPU\n");
  printf("Performing %d iterations...\n", iters);
  float gpu_timing[4] = {0};
  heat_propagation_gpu(iters, n, m, dst_gpu, gpu_timing);

  if (timing) {
    printf("Everything completed in %.6f seconds\n",
           gpu_timing[0] + gpu_timing[1] + gpu_timing[2] + gpu_timing[3]);
    printf("Memory allocation on the GPU took %.6f seconds\n", gpu_timing[0]);
    printf("Initialisation on the GPU took %.6f seconds\n", gpu_timing[1]);
    printf("Iterations completed in %.6f seconds\n", gpu_timing[2]);
    printf("Memory transfer back to the CPU took %.6f seconds\n",
           gpu_timing[3]);
  }
  printf("Completed the work on the GPU\n\n");
  // print_matrix(n, m, increment, dst_gpu);

  // Compare CPU and GPU results if CPU was not skipped
  if (!cpu) {
    printf("COMPARISON\n");

    // Compare matrix results
    int matrix_mismatches =
        mismatches(n, m, increment, dst, increment, dst_gpu);
    float matrix_maxdiff = maxdiff(n, m, increment, dst, increment, dst_gpu);

    printf("Matrix comparison:\n");
    printf("\tMismatches: %d\n", matrix_mismatches);
    printf("\tMaximum difference: %e\n\n", (float) matrix_maxdiff);
  }

  // Free memory
  free(averages);
  free(dst);
  free(dst_gpu);
  free(src);

  return EXIT_SUCCESS;
}
