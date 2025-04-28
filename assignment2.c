#include <stdio.h>
#include <stdlib.h>

#include "average-gpu.h"
#include "average.h"
#include "iteration-gpu.h"
#include "iteration.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  arguments   args      = parse(argc, argv); // Parse command-line arguments
  const int   n         = args.n;
  const int   m         = args.m;
  const int   increment = m + 2;
  const int   iters     = args.iters;
  const bool  average   = args.average;
  const bool  cpu       = args.cpu;
  const bool  timing    = args.timing;
  const float tolerance = 1e-4f;

  // Print parameters used
  printf("Matrix:\t\t%d x %d (x %d iterations)\n", n, m, iters);
  printf("Precision:\tFP32\n");
  printf("Tolerance:\t%.2e\n", tolerance);

  // Allocate memory on the host
  float* const cpu_matrix   = calloc(n * increment, sizeof(float));
  float* const cpu_averages = calloc(n, sizeof(float));
  float* const gpu_matrix   = calloc(n * increment, sizeof(float));
  float* const gpu_averages = calloc(n, sizeof(float));
  float* const temp_matrix  = calloc(n * increment, sizeof(float));

  if (cpu_matrix == NULL || cpu_averages == NULL || gpu_matrix == NULL ||
      gpu_averages == NULL || temp_matrix == NULL) {
    fprintf(stderr, "Failed to allocate memory on host\n");
    exit(EXIT_FAILURE);
  }

  // Timing on the CPU
  double start_time   = get_current_time();
  double cpu_times[2] = {0}; // Array to store timing results
  int    timing_index = 0;

  // Iterations on the CPU
  if (!cpu) {
    printf("Performing CPU iterations...\n");
    heat_propagation(iters, n, m, cpu_matrix, temp_matrix);
    cpu_times[timing_index++] = get_duration(&start_time);
    printf("CPU iterations done.\n");

    // Averages, if requested
    if (average) {
      printf("Performing CPU averaging...\n");
      start_time = get_current_time();
      average_rows(n, m, increment, cpu_matrix, cpu_averages);
      cpu_times[timing_index++] = get_duration(&start_time);
      printf("CPU averaging done.\n");
    }
  }

  // GPU iterations
  printf("Performing GPU iterations...\n");
  float gpu_iteration_timing[5] = {
      0}; // Setup, Allocation, Transfer to, Computation, Transfer from
  heat_propagation_gpu(iters, n, m, gpu_matrix, gpu_iteration_timing);
  printf("GPU iterations done.\n");

  // GPU averaging
  float gpu_averaging_timing[5] = {
      0}; // Setup, Allocation, Transfer to, Computation, Transfer from
  if (average) {
    printf("Performing GPU averaging...\n");
    average_rows_gpu(n, m, increment, gpu_matrix, gpu_averages,
                     gpu_averaging_timing);
    printf("GPU averaging done.\n");
  }

  // Output comparison and timing info
  printf("ITERATIONS\n");

  if (!cpu) {
    // Compare matrix results
    int matrix_mismatches =
        mismatches(n, m, increment, cpu_matrix, increment, gpu_matrix);
    float matrix_maxdiff =
        maxdiff(n, m, increment, cpu_matrix, increment, gpu_matrix);

    printf("\tMismatches: %d\n", matrix_mismatches);
    printf("\tMaximum difference: %.2e\n", matrix_maxdiff);

    if (timing) {
      printf("\t-------------------------------------------------\n");

      // Calculate total GPU time
      float total_gpu_time = 0.0f;
      for (int i = 0; i < 5; i++) {
        total_gpu_time += gpu_iteration_timing[i];
      }

      printf("\tSpeedup (overall): %.2f\n", cpu_times[0] / total_gpu_time);
      printf("\tSpeedup (computation): %.2f\n",
             cpu_times[0] / gpu_iteration_timing[3]);
      printf("\t-------------------------------------------------\n");

      printf("\tTIMINGS\t\tCPU\t\tGPU\n");
      printf("\tTotal time\t%.2e\t%.2e\n", cpu_times[0], total_gpu_time);

      printf("\tSetup\t\t\t\t%.2e (%.2f%%)\n", gpu_iteration_timing[0],
             100.0f * gpu_iteration_timing[0] / total_gpu_time);
      printf("\tAllocation\t\t\t%.2e (%.2f%%)\n", gpu_iteration_timing[1],
             100.0f * gpu_iteration_timing[1] / total_gpu_time);
      printf("\tTransfer to\t\t\t%.2e (%.2f%%)\n", gpu_iteration_timing[2],
             100.0f * gpu_iteration_timing[2] / total_gpu_time);
      printf("\tComputation\t\t\t%.2e (%.2f%%)\n", gpu_iteration_timing[3],
             100.0f * gpu_iteration_timing[3] / total_gpu_time);
      printf("\tTransfer from\t\t\t%.2e (%.2f%%)\n", gpu_iteration_timing[4],
             100.0f * gpu_iteration_timing[4] / total_gpu_time);
    }
  }

  // Output averaging results
  if (average) {
    printf("\nAVERAGES\n");

    if (!cpu) {
      // Compare averaging results
      int averages_mismatches =
          mismatches(n, 1, 1, cpu_averages, 1, gpu_averages);
      float averages_maxdiff = maxdiff(n, 1, 1, cpu_averages, 1, gpu_averages);

      printf("\tMismatches: %d\n", averages_mismatches);
      printf("\tMaximum difference: %.2e\n", averages_maxdiff);
      printf("\t-------------------------------------------------\n");

      // Calculate overall averages (average of all row averages)
      float cpu_overall = 0.0f;
      float gpu_overall = 0.0f;
      for (int i = 0; i < n; i++) {
        cpu_overall += cpu_averages[i];
        gpu_overall += gpu_averages[i];
      }
      cpu_overall /= n;
      gpu_overall /= n;

      printf("\tOverall (CPU): %.2e\n", cpu_overall);
      printf("\tOverall (GPU): %.2e\n", gpu_overall);

      if (timing) {
        printf("\t-------------------------------------------------\n");

        // Calculate total GPU time
        float total_gpu_time = 0.0f;
        for (int i = 0; i < 5; i++) {
          total_gpu_time += gpu_averaging_timing[i];
        }

        printf("\tSpeedup (overall): %.2f\n", cpu_times[1] / total_gpu_time);
        printf("\tSpeedup (computation): %.2f\n",
               cpu_times[1] / gpu_averaging_timing[3]);
        printf("\t-------------------------------------------------\n");

        printf("\tTIMINGS\t\tCPU\t\tGPU\n");
        printf("\tTotal time\t%.2e\t%.2e\n", cpu_times[1], total_gpu_time);

        printf("\tSetup\t\t\t\t%.2e (%.2f%%)\n", gpu_averaging_timing[0],
               100.0f * gpu_averaging_timing[0] / total_gpu_time);
        printf("\tAllocation\t\t\t%.2e (%.2f%%)\n", gpu_averaging_timing[1],
               100.0f * gpu_averaging_timing[1] / total_gpu_time);
        printf("\tTransfer to\t\t\t%.2e (%.2f%%)\n", gpu_averaging_timing[2],
               100.0f * gpu_averaging_timing[2] / total_gpu_time);
        printf("\tComputation\t\t\t%.2e (%.2f%%)\n", gpu_averaging_timing[3],
               100.0f * gpu_averaging_timing[3] / total_gpu_time);
        printf("\tTransfer from\t\t\t%.2e (%.2f%%)\n", gpu_averaging_timing[4],
               100.0f * gpu_averaging_timing[4] / total_gpu_time);
      }
    }
  }

  // Free memory
  printf("Freeing memory...\n");
  free(cpu_matrix);
  free(cpu_averages);
  free(gpu_matrix);
  free(gpu_averages);
  free(temp_matrix);
  printf("Finished.\n");

  return EXIT_SUCCESS;
}
