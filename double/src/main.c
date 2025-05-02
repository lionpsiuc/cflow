#include <math.h>

#include "../include/cpu/average.h"
#include "../include/cpu/iteration.h"
#include "../include/gpu/average.h"
#include "../include/gpu/iteration.h"
#include "../include/gpu/utils.h"
#include "../include/utils.h"

/**
 * @brief Main entry point for the heat propagation simulation application.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of strings containing the command-line arguments.
 *
 * @return int Returns EXIT_SUCCESS (i.e., 0) if the programme completes
 *             successfully, or EXIT_FAILURE (i.e., 1) if an error occurs (e.g.,
 *             memory allocation failure, invalid arguments, GPU dimension check
 *             failure).
 */
int main(int argc, char* argv[]) {
  arguments args = parse(argc, argv); // Parse command-line options

  // Extract parsed arguments
  const int  n         = args.n;       // Matrix height
  const int  m         = args.m;       // Matrix width
  const int  increment = m + 2;        // Row stride (i.e., width and padding)
  const int  iters     = args.iters;   // Number of iterations
  const bool average   = args.average; // Flag to perform averaging
  const bool cpu       = args.cpu;     // Flag to skip CPU computation
  const bool timing    = args.timing;  // Flag to display timing info

  printf("\n  Matrix:    %d x %d (%d iterations)\n", n, m, iters);
  printf("  Precision: FP32\n");

  // Pointers for host memory buffers
  float* cpu_matrix   = NULL; // Stores final CPU propagation result
  float* cpu_averages = NULL; // Stores CPU row averages
  float* gpu_matrix =
      NULL; // Stores final GPU propagation result (which is copied back)
  float* gpu_averages = NULL; // Stores GPU row averages (which is copied back)
  float* temp_matrix  = NULL; // Temporary buffer for CPU propagation

  // Allocate memory for the main matrices
  cpu_matrix  = calloc(n * increment, sizeof(float));
  gpu_matrix  = calloc(n * increment, sizeof(float));
  temp_matrix = calloc(n * increment, sizeof(float));

  // Allocate memory for average results if requested
  if (average) {
    cpu_averages = calloc(n, sizeof(float));
    gpu_averages = calloc(n, sizeof(float));
  }

  // Check if any host memory allocation failed
  if (cpu_matrix == NULL || gpu_matrix == NULL || temp_matrix == NULL ||
      (average && (cpu_averages == NULL || gpu_averages == NULL))) {
    fprintf(stderr, "  Error: Failed to allocate host memory\n");

    // Free any potentially allocated memory before exiting
    free(cpu_matrix);
    free(cpu_averages);
    free(gpu_matrix);
    free(gpu_averages);
    free(temp_matrix);

    return EXIT_FAILURE;
  }

  double start_time       = get_current_time(); // For timing CPU parts
  double cpu_times[2]     = {0.0, 0.0};
  int    cpu_timing_index = 0;

  // Run CPU part unless the -c flag was given
  if (!cpu) {
    printf("\n  Performing CPU iterations...\n");
    start_time = get_current_time(); // Reset timer before propagation

    // Run CPU propagation
    heat_propagation(iters, n, m, cpu_matrix, temp_matrix);

    cpu_times[cpu_timing_index++] = get_duration(&start_time);
    printf("  CPU iterations done\n");

    // Perform CPU averaging if requested
    if (average) {
      printf("\n  Performing CPU averaging...\n");
      start_time = get_current_time(); // Reset timer before averaging

      // Run CPU averaging
      average_rows(n, m, increment, cpu_matrix, cpu_averages);

      cpu_times[cpu_timing_index++] = get_duration(&start_time);
      printf("  CPU averaging done\n");
    }
  } else {
    printf("\n  Skipping CPU computations\n");
  }

  // Pointer to receive the final result grid location on the device
  float* device_final_matrix = NULL;

  printf("\n  Performing GPU iterations...\n");
  float gpu_iteration_timing[4] = {0.0f};

  // Call the GPU propagation function
  int prop_status = heat_propagation_gpu(
      iters, n, m, gpu_matrix, gpu_iteration_timing, &device_final_matrix);

  if (prop_status != 0) {
    fprintf(stderr, "  Error: Exiting due to GPU propagation error\n\n");

    // Clean allocated host memory before exiting
    free(cpu_matrix);
    free(cpu_averages);
    free(gpu_matrix);
    free(gpu_averages);
    free(temp_matrix);

    exit(EXIT_FAILURE);
  }
  printf("\n  GPU iterations done\n");
  float gpu_averaging_timing[5] = {0.0f};

  // Perform GPU averaging if requested
  if (average) {
    printf("\n  Performing GPU averaging...\n");

    // Call GPU averaging function
    int avg_status = average_rows_gpu(n, m, increment, device_final_matrix,
                                      gpu_averages, gpu_averaging_timing);

    if (avg_status != 0) {
      fprintf(stderr, "  Error: Exiting due to GPU averaging error\n\n");

      // Clean allocated host memory and the final device matrix
      free(cpu_matrix);
      free(cpu_averages);
      free(gpu_matrix);
      free(gpu_averages);
      free(temp_matrix);
      freedeviceptr(device_final_matrix);

      exit(EXIT_FAILURE);
    }
    printf("  GPU averaging done\n");
  }

  // Calculating precision
  float matrix_mismatches   = NAN;
  float matrix_maxdiff      = NAN;
  float averages_mismatches = NAN;
  float averages_maxdiff    = NAN;
  if (!cpu) {
    matrix_mismatches =
        mismatches(n, m, increment, cpu_matrix, increment, gpu_matrix);
    matrix_maxdiff =
        maxdiff(n, m, increment, cpu_matrix, increment, gpu_matrix);
    if (average) {
      averages_mismatches = mismatches(n, 1, 1, cpu_averages, 1, gpu_averages);
      averages_maxdiff    = maxdiff(n, 1, 1, cpu_averages, 1, gpu_averages);
    }
  }

  // Display results for the propagation step
  printf("\n  Results for propagation\n\n");
  if (!cpu) {
    printf("    Mismatches:         %d\n", (int) matrix_mismatches);
    printf("    Maximum difference: %.2e\n", matrix_maxdiff);
  } else if (!timing) {
    printf("    No further information requested\n");
  }
  if (timing) {
    float total_gpu_iter_time = 0.0f;
    for (int i = 0; i < 4; i++) {
      if (!isnan(gpu_iteration_timing[i])) {
        total_gpu_iter_time += gpu_iteration_timing[i];
      }
    }

    // Propagation step speedups
    float prop_comp_speedup =
        (!cpu && !isnan(cpu_times[0]) && gpu_iteration_timing[2] != 0.0f &&
         !isnan(gpu_iteration_timing[2]))
            ? (cpu_times[0] / gpu_iteration_timing[2])
            : NAN;
    float prop_overall_speedup =
        (!cpu && !isnan(cpu_times[0]) && total_gpu_iter_time != 0.0f)
            ? (cpu_times[0] / total_gpu_iter_time)
            : NAN;

    // Print speedup calculations if CPU wasn't skipped
    if (!cpu) {
      printf("\n    Overall speedup:       %.2f\n", prop_overall_speedup);
      printf("    Computational speedup: %.2f\n\n", prop_comp_speedup);
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

  // Display results for the averaging step if performed
  if (average) {
    printf("\n  Results for averaging\n\n");
    if (!cpu) {
      printf("    Mismatches:         %d\n", (int) averages_mismatches);
      printf("    Maximum difference: %.2e\n", averages_maxdiff);
      float cpu_overall_avg = 0.0f, gpu_overall_avg = 0.0f;
      for (int i = 0; i < n; i++) {
        if (!isnan(cpu_averages[i])) {
          cpu_overall_avg += cpu_averages[i];
        }
        if (!isnan(gpu_averages[i])) {
          gpu_overall_avg += gpu_averages[i];
        }
      }
      cpu_overall_avg /= (n > 0 ? n : 1);
      gpu_overall_avg /= (n > 0 ? n : 1);
      printf("\n    Average of averages (CPU): %.2e\n", cpu_overall_avg);
      printf("    Average of averages (GPU): %.2e\n", gpu_overall_avg);
    } else if (!timing) {
      printf("    No further information requested\n");
    }
    if (timing) {
      float total_gpu_avg_time = 0.0f;
      for (int i = 0; i < 5; i++) {
        if (!isnan(gpu_averaging_timing[i])) {
          total_gpu_avg_time += gpu_averaging_timing[i];
        }
      }
      float avg_comp_speedup =
          (!cpu && !isnan(cpu_times[1]) && gpu_averaging_timing[3] != 0.0f &&
           !isnan(gpu_averaging_timing[3]))
              ? (cpu_times[1] / gpu_averaging_timing[3])
              : NAN;
      float avg_overall_speedup =
          (!cpu && !isnan(cpu_times[1]) && total_gpu_avg_time != 0.0f)
              ? (cpu_times[1] / total_gpu_avg_time)
              : NAN;

      // Print speedup calculations if CPU wasn't skipped
      if (!cpu) {
        printf("\n    Overall speedup:       %.2f\n", avg_overall_speedup);
        printf("    Computational speedup: %.2f\n\n", avg_comp_speedup);
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

  // Writing results to file
  FILE* outfile = fopen("timings.txt", "a");
  if (outfile == NULL) {
    fprintf(stderr, "  Error: Could not open timings.txt for writing\n");
  } else {

    // Check if file is empty to write header
    fseek(outfile, 0, SEEK_END);
    long size = ftell(outfile);
    if (size == 0) {
      fprintf(outfile, "n m p block_size prop_max_diff prop_speedup "
                       "prop_cpu_time prop_gpu_time average_max_diff "
                       "average_speedup average_cpu_time average_gpu_time\n");
    }

    // Prepare data points for writing
    int block_size = 256; // Hardcoded based on GPU kernel implementation

    // Propagation results
    float p_max_diff = !cpu ? matrix_maxdiff : NAN;

    // Ensure denominator is not zero or NAN before calculating speedup
    float p_speedup =
        (!cpu && !isnan(cpu_times[0]) && gpu_iteration_timing[2] != 0.0f &&
         !isnan(gpu_iteration_timing[2]))
            ? (cpu_times[0] / gpu_iteration_timing[2])
            : NAN;

    double p_cpu_time = !cpu ? cpu_times[0] : NAN;
    float  p_gpu_time = gpu_iteration_timing[2];

    // Averaging results
    float a_max_diff = (!cpu && average) ? averages_maxdiff : NAN;

    // Ensure denominator is not zero or NAN before calculating speedup
    float a_speedup =
        (!cpu && average && !isnan(cpu_times[1]) &&
         gpu_averaging_timing[3] != 0.0f && !isnan(gpu_averaging_timing[3]))
            ? (cpu_times[1] / gpu_averaging_timing[3])
            : NAN;

    double a_cpu_time = (!cpu && average) ? cpu_times[1] : NAN;
    float  a_gpu_time = average ? gpu_averaging_timing[3] : NAN;

    // Write results
    fprintf(outfile, "%d %d %d %d %.2e %.2f %.2e %.2e %.2e %.2f %.2e %.2e\n", n,
            m, iters, block_size, p_max_diff, p_speedup, p_cpu_time, p_gpu_time,
            a_max_diff, a_speedup, a_cpu_time, a_gpu_time);

    // Close the file
    fclose(outfile);
  }

  // Free memory
  printf("\n  Freeing memory...\n");
  free(cpu_matrix);
  free(cpu_averages);
  free(gpu_matrix);
  free(gpu_averages);
  free(temp_matrix);
  if (device_final_matrix != NULL) {
    freedeviceptr(device_final_matrix);
  }

  printf("  Finished\n\n");
  return EXIT_SUCCESS;
}
