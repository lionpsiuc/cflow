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

  // Print parameters used
  printf("\nMatrix:\t\t%d x %d (%d iterations)\n", n, n, iters);
  printf("Precision:\t32-bit float\n\n");

  // Allocate memory on the host
  float* const averages = calloc(n, sizeof(float));
  float* const dst      = calloc(n * increment, sizeof(float));
  float* const dst_gpu  = calloc(n * increment, sizeof(float));
  float* const src      = calloc(n * increment, sizeof(float));
  if (averages == NULL || dst == NULL || dst_gpu == NULL || src == NULL) {
    fprintf(stderr, "Failed to allocate memory on host\n");
    exit(EXIT_FAILURE);
  }

  // Timing on the CPU
  double start_time   = get_current_time();
  double cpu_times[2] = {0}; // Array to store timing results
  int    timing_index = 0;

  // Iterations on the CPU
  printf("Performing %d iterations...\n", iters);
  heat_propagation(iters, n, m, dst, src);
  cpu_times[timing_index++] = get_duration(&start_time);
  printf("Iterations completed in %.6f seconds\n\n", cpu_times[0]);

  // Averages, if requested
  if (average) {
    printf("Calculating row averages...\n");

    // Calculate averages
    start_time = get_current_time();
    average_rows(m, m, increment, dst, averages);
    cpu_times[timing_index++] = get_duration(&start_time);
    printf("Row averages calculated in %.6f seconds\n\n", cpu_times[1]);
  }

  // Free memory
  free(averages);
  free(dst);
  free(dst_gpu);
  free(src);

  return EXIT_SUCCESS;
}
