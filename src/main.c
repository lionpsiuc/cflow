#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/cpu/average.h"
#include "../include/cpu/iteration.h"
#include "../include/gpu/average.h"
#include "../include/gpu/iteration.h"
#include "../include/utils.h"

int main(int argc, char* argv[]) {
  arguments  args      = parse(argc, argv);
  const int  n         = args.n;
  const int  m         = args.m;
  const int  increment = m + 2;
  const int  iters     = args.iters;
  const bool average   = args.average;
  const bool cpu       = args.cpu;
  const bool timing    = args.timing;

  printf("Matrix:\t\t%d x %d (x %d iterations)\n", n, m, iters);
  printf("Precision:\tFP32\n");

  float* cpu_matrix   = NULL;
  float* cpu_averages = NULL;
  float* gpu_matrix   = NULL;
  float* gpu_averages = NULL;
  float* temp_matrix  = NULL;

  cpu_matrix  = calloc(n * increment, sizeof(float));
  gpu_matrix  = calloc(n * increment, sizeof(float));
  temp_matrix = calloc(n * increment, sizeof(float));
  if (average) {
    cpu_averages = calloc(n, sizeof(float));
    gpu_averages = calloc(n, sizeof(float));
  }

  if (cpu_matrix == NULL || gpu_matrix == NULL || temp_matrix == NULL ||
      (average && (cpu_averages == NULL || gpu_averages == NULL))) {
    fprintf(stderr, "Failed to allocate host memory\n");
    free(cpu_matrix);
    free(cpu_averages);
    free(gpu_matrix);
    free(gpu_averages);
    free(temp_matrix);
    return EXIT_FAILURE;
  }

  double start_time       = get_current_time();
  double cpu_times[2]     = {0.0, 0.0};
  int    cpu_timing_index = 0;

  if (!cpu) {
    printf("Performing CPU iterations...\n");
    heat_propagation(iters, n, m, cpu_matrix, temp_matrix);
    cpu_times[cpu_timing_index++] = get_duration(&start_time);
    printf("CPU iterations done.\n");
    if (average) {
      printf("Performing CPU averaging...\n");
      start_time = get_current_time();
      average_rows(n, m, increment, cpu_matrix, cpu_averages);
      cpu_times[cpu_timing_index++] = get_duration(&start_time);
      printf("CPU averaging done.\n");
    }
  } else {
    printf("Skipping CPU computations.\n");
  }

  // --- GPU Computations ---
  float* device_final_matrix = NULL;
  int    gpu_error_status    = 0; // Use int for status

  printf("Performing GPU iterations...\n");
  float gpu_iteration_timing[4] = {0.0f};
  // Call modified GPU iteration function, check int return value
  gpu_error_status = heat_propagation_gpu(
      iters, n, m, gpu_matrix, // Pass host buffer for result copy
      gpu_iteration_timing, &device_final_matrix);

  if (gpu_error_status != 0) { // Check for non-zero (failure)
    fprintf(stderr, "GPU Iteration failed. Exiting.\n");
    // Attempt cleanup
    if (device_final_matrix)
      CUDA_CHECK(cudaFree(
          device_final_matrix)); // Still use CUDA_CHECK for free if defined
    free(cpu_matrix);
    free(cpu_averages);
    free(gpu_matrix);
    free(gpu_averages);
    free(temp_matrix);
    CUDA_CHECK(cudaDeviceReset()); // Try to reset device state
    return EXIT_FAILURE;
  }
  printf("GPU iterations done.\n");

  // GPU Averaging (Optional)
  float gpu_averaging_timing[5] = {0.0f};
  if (average) {
    if (device_final_matrix == NULL) {
      fprintf(stderr, "Error: Device matrix pointer is NULL before averaging "
                      "(GPU iteration likely failed). Exiting.\n");
      free(cpu_matrix);
      free(cpu_averages);
      free(gpu_matrix);
      free(gpu_averages);
      free(temp_matrix);
      CUDA_CHECK(cudaDeviceReset());
      return EXIT_FAILURE;
    }
    printf("Performing GPU averaging...\n");
    // Call modified GPU averaging function, check int return value
    gpu_error_status =
        average_rows_gpu(n, m, increment,
                         device_final_matrix, // Pass device pointer
                         gpu_averages, gpu_averaging_timing);
    if (gpu_error_status != 0) { // Check for non-zero (failure)
      fprintf(stderr, "GPU Averaging failed. Exiting.\n");
      // Attempt cleanup, including the matrix pointer received from iterations
      CUDA_CHECK(cudaFree(device_final_matrix));
      free(cpu_matrix);
      free(cpu_averages);
      free(gpu_matrix);
      free(gpu_averages);
      free(temp_matrix);
      CUDA_CHECK(cudaDeviceReset());
      return EXIT_FAILURE;
    }
    printf("GPU averaging done.\n");
  }

  // --- Output Comparison and Timing Info ---
  // (Comparison and timing printing logic remains largely the same)
  printf("\nITERATIONS\n");
  if (!cpu) {
    int matrix_mismatches =
        mismatches(n, m, increment, cpu_matrix, increment, gpu_matrix);
    float matrix_maxdiff =
        maxdiff(n, m, increment, cpu_matrix, increment, gpu_matrix);
    printf("\tMismatches: %d\n", matrix_mismatches);
    printf("\tMaximum difference: %.2e\n", matrix_maxdiff);
    if (timing) {
      printf("\t-------------------------------------------------\n");
      float total_gpu_iter_time = 0.0f;
      for (int i = 0; i < 4; i++) {
        total_gpu_iter_time += gpu_iteration_timing[i];
      }
      if (total_gpu_iter_time > 0)
        printf("\tSpeedup (overall): %.2f\n",
               cpu_times[0] / total_gpu_iter_time);
      else
        printf("\tSpeedup (overall): N/A\n");
      if (gpu_iteration_timing[2] > 0)
        printf("\tSpeedup (computation): %.2f\n",
               cpu_times[0] / gpu_iteration_timing[2]);
      else
        printf("\tSpeedup (computation): N/A\n");
      printf("\t-------------------------------------------------\n");
      printf("\tTIMINGS\t\tCPU\t\tGPU\n");
      printf("\tTotal time\t%.2e\t%.2e\n", cpu_times[0], total_gpu_iter_time);
      if (total_gpu_iter_time > 0) {
        printf("\tAllocation\t\t\t%.2e (%.2f%%)\n", gpu_iteration_timing[0],
               100.0f * gpu_iteration_timing[0] / total_gpu_iter_time);
        printf("\tInitialization\t\t\t%.2e (%.2f%%)\n", gpu_iteration_timing[1],
               100.0f * gpu_iteration_timing[1] / total_gpu_iter_time);
        printf("\tComputation\t\t\t%.2e (%.2f%%)\n", gpu_iteration_timing[2],
               100.0f * gpu_iteration_timing[2] / total_gpu_iter_time);
        printf("\tTransfer From\t\t\t%.2e (%.2f%%)\n", gpu_iteration_timing[3],
               100.0f * gpu_iteration_timing[3] / total_gpu_iter_time);
      } else {
        printf("\tGPU Timings:\t\t\tAll Zero\n");
      }
    }
  } else {
    printf("\tSkipping comparison (CPU run disabled).\n");
  }

  if (average) {
    printf("\nAVERAGES\n");
    if (!cpu) {
      int averages_mismatches =
          mismatches(n, 1, 1, cpu_averages, 1, gpu_averages);
      float averages_maxdiff = maxdiff(n, 1, 1, cpu_averages, 1, gpu_averages);
      printf("\tMismatches: %d\n", averages_mismatches);
      printf("\tMaximum difference: %.2e\n", averages_maxdiff);
      printf("\t-------------------------------------------------\n");
      float cpu_overall_avg = 0.0f, gpu_overall_avg = 0.0f;
      for (int i = 0; i < n; i++) {
        cpu_overall_avg += cpu_averages[i];
        gpu_overall_avg += gpu_averages[i];
      }
      cpu_overall_avg /= (n > 0 ? n : 1);
      gpu_overall_avg /= (n > 0 ? n : 1);
      printf("\tOverall (CPU): %.2e\n", cpu_overall_avg);
      printf("\tOverall (GPU): %.2e\n", gpu_overall_avg);
      if (timing) {
        printf("\t-------------------------------------------------\n");
        float total_gpu_avg_time = 0.0f;
        for (int i = 0; i < 5; i++) {
          total_gpu_avg_time += gpu_averaging_timing[i];
        }
        if (total_gpu_avg_time > 0)
          printf("\tSpeedup (overall): %.2f\n",
                 cpu_times[1] / total_gpu_avg_time);
        else
          printf("\tSpeedup (overall): N/A\n");
        if (gpu_averaging_timing[3] > 0)
          printf("\tSpeedup (computation): %.2f\n",
                 cpu_times[1] / gpu_averaging_timing[3]);
        else
          printf("\tSpeedup (computation): N/A\n");
        printf("\t-------------------------------------------------\n");
        printf("\tTIMINGS\t\tCPU\t\tGPU\n");
        printf("\tTotal time\t%.2e\t%.2e\n", cpu_times[1], total_gpu_avg_time);
        if (total_gpu_avg_time > 0) {
          printf("\tSetup\t\t\t\t%.2e (%.2f%%)\n", gpu_averaging_timing[0],
                 100.0f * gpu_averaging_timing[0] / total_gpu_avg_time);
          printf("\tAllocation\t\t\t%.2e (%.2f%%)\n", gpu_averaging_timing[1],
                 100.0f * gpu_averaging_timing[1] / total_gpu_avg_time);
          printf("\tTransfer to\t\t\t%.2e (%.2f%%)\n", gpu_averaging_timing[2],
                 100.0f * gpu_averaging_timing[2] / total_gpu_avg_time);
          printf("\tComputation\t\t\t%.2e (%.2f%%)\n", gpu_averaging_timing[3],
                 100.0f * gpu_averaging_timing[3] / total_gpu_avg_time);
          printf("\tTransfer from\t\t\t%.2e (%.2f%%)\n",
                 gpu_averaging_timing[4],
                 100.0f * gpu_averaging_timing[4] / total_gpu_avg_time);
        } else {
          printf("\tGPU Timings:\t\t\tAll Zero\n");
        }
      }
    } else {
      printf("\tSkipping comparison (CPU run disabled).\n");
    }
  }

  // --- Free Memory ---
  printf("Freeing memory...\n");
  free(cpu_matrix);
  free(cpu_averages);
  free(gpu_matrix);
  free(gpu_averages);
  free(temp_matrix);

  // Free the final matrix buffer on the device
  if (device_final_matrix != NULL) {
    // Use the basic CUDA_CHECK for freeing memory if it's defined and useful
    // Otherwise, call cudaFree directly and check its return status manually if
    // needed
    CUDA_CHECK(cudaFree(device_final_matrix));
    // cudaError_t free_status = cudaFree(device_final_matrix);
    // if (free_status != cudaSuccess) {
    //     fprintf(stderr, "Warning: Failed to free final device matrix.\n");
    // }
  }

  // Optional: Reset device context
  // CUDA_CHECK(cudaDeviceReset());

  printf("Finished.\n");
  return EXIT_SUCCESS;
}
