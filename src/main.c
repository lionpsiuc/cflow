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

  printf("\n  Matrix:    %d x %d (%d iterations)\n", n, m, iters);
  printf("  Precision: FP32\n");

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
    printf("\n  Performing CPU iterations...\n");
    heat_propagation(iters, n, m, cpu_matrix, temp_matrix);
    cpu_times[cpu_timing_index++] = get_duration(&start_time);
    printf("  CPU iterations done\n");
    if (average) {
      printf("\n  Performing CPU averaging...\n");
      start_time = get_current_time();
      average_rows(n, m, increment, cpu_matrix, cpu_averages);
      cpu_times[cpu_timing_index++] = get_duration(&start_time);
      printf("  CPU averaging done\n");
    }
  } else {
    printf("\n  Skipping CPU computations\n");
  }
  float* device_final_matrix = NULL;

  printf("\n  Performing GPU iterations...\n");
  float gpu_iteration_timing[4] = {0.0f};
  heat_propagation_gpu(iters, n, m, gpu_matrix, gpu_iteration_timing,
                       &device_final_matrix);
  printf("\n  GPU iterations done\n");

  float gpu_averaging_timing[5] = {0.0f};
  if (average) {
    printf("\n  Performing GPU averaging...\n");
    // Call modified GPU averaging function
    average_rows_gpu(n, m, increment,
                     device_final_matrix, // Pass device pointer
                     gpu_averages, gpu_averaging_timing);
    printf("  GPU averaging done\n");
  }

  printf("\n  Results for propagation\n\n");
  if (!cpu) {
    int matrix_mismatches =
        mismatches(n, m, increment, cpu_matrix, increment, gpu_matrix);
    float matrix_maxdiff =
        maxdiff(n, m, increment, cpu_matrix, increment, gpu_matrix);
    printf("    Mismatches:         %d\n", matrix_mismatches);
    printf("    Maximum difference: %.2e\n\n", matrix_maxdiff);
  } else if (!timing) {
    printf("\n    No further information requested\n");
  }

  if (timing) {
    float total_gpu_iter_time = 0.0f;
    for (int i = 0; i < 4; i++) {
      total_gpu_iter_time += gpu_iteration_timing[i];
    }

    // Print speedup calculations if CPU wasn't skipped
    if (!cpu) {
      printf("    Overall speedup:       %.2f\n",
             cpu_times[0] / total_gpu_iter_time);
      printf("    Computational speedup: %.2f\n\n",
             cpu_times[0] / gpu_iteration_timing[2]);
    } else {
      printf("    Overall speedup:       N/A\n");
      printf("    Computational speedup: N/A\n\n");
    }

    // Print timing table header
    printf("    Step\t\tCPU\t\tGPU\n");
    printf("    --------------------------------------------\n");

    // Print total time row
    if (!cpu) {
      printf("    Total time\t\t%.2e\t%.2e\n\n", cpu_times[0],
             total_gpu_iter_time);
    } else {
      printf("    Total time\t\tN/A\t\t%.2e\n\n", total_gpu_iter_time);
    }

    // Print detailed GPU timing steps
    printf("    Allocation\t\t\t\t%.2e\n", gpu_iteration_timing[0]);
    printf("    Initialisation\t\t\t%.2e\n", gpu_iteration_timing[1]);
    printf("    Computation\t\t\t\t%.2e\n", gpu_iteration_timing[2]);
    printf("    Transfer from\t\t\t%.2e\n", gpu_iteration_timing[3]);
  }

  if (average) {
    printf("\n  Results for averaging\n\n");
    if (!cpu) {
      int averages_mismatches =
          mismatches(n, 1, 1, cpu_averages, 1, gpu_averages);
      float averages_maxdiff = maxdiff(n, 1, 1, cpu_averages, 1, gpu_averages);
      printf("    Mismatches:         %d\n", averages_mismatches);
      printf("    Maximum difference: %.2e\n\n", averages_maxdiff);

      float cpu_overall_avg = 0.0f, gpu_overall_avg = 0.0f;
      for (int i = 0; i < n; i++) {
        cpu_overall_avg += cpu_averages[i];
        gpu_overall_avg += gpu_averages[i];
      }
      cpu_overall_avg /= (n > 0 ? n : 1);
      gpu_overall_avg /= (n > 0 ? n : 1);
      printf("    Overall (CPU): %.2e\n", cpu_overall_avg);
      printf("    Overall (GPU): %.2e\n\n", gpu_overall_avg);
    } else if (!timing) {
      printf("\n    No further information requested\n");
    }

    if (timing) {
      float total_gpu_avg_time = 0.0f;
      for (int i = 0; i < 5; i++) {
        total_gpu_avg_time += gpu_averaging_timing[i];
      }

      // Print speedup calculations if CPU wasn't skipped
      if (!cpu) {
        printf("    Overall speedup:       %.2f\n",
               cpu_times[1] / total_gpu_avg_time);
        printf("    Computational speedup: %.2f\n\n",
               cpu_times[1] / gpu_averaging_timing[3]);
      } else {
        printf("    Overall speedup:       N/A\n");
        printf("    Computational speedup: N/A\n\n");
      }

      // Print timing table header
      printf("    Step\t\tCPU\t\tGPU\n");
      printf("    --------------------------------------------\n");

      // Print total time row
      if (!cpu) {
        printf("    Total time\t\t%.2e\t%.2e\n\n", cpu_times[1],
               total_gpu_avg_time);
      } else {
        printf("    Total time\t\tN/A\t\t%.2e\n\n", total_gpu_avg_time);
      }

      // Print detailed GPU timing steps
      printf("    Setup\t\t\t\t%.2e\n", gpu_averaging_timing[0]);
      printf("    Allocation\t\t\t\t%.2e\n", gpu_averaging_timing[1]);
      printf("    Transfer to\t\t\t\t%.2e\n", gpu_averaging_timing[2]);
      printf("    Computation\t\t\t\t%.2e\n", gpu_averaging_timing[3]);
      printf("    Transfer from\t\t\t%.2e\n", gpu_averaging_timing[4]);
    }
  }

  // --- Free Memory ---
  printf("\n  Freeing memory...\n");
  free(cpu_matrix);
  free(cpu_averages);
  free(gpu_matrix);
  free(gpu_averages);
  free(temp_matrix);

  if (device_final_matrix != NULL) {
    cudaFree(device_final_matrix);
  }

  printf("  Finished\n");
  return EXIT_SUCCESS;
}
