#include <stdio.h>
#include <stdlib.h>

#include "average.h"
#include "iteration.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  arguments args = parse(argc, argv); // Parse command-line arguments
  printf("\nMatrix:\t\t%d x %d (%d iterations)\n", args.n, args.m,
         args.iterations);
  printf("Precision:\t32-bit float\n\n");

  // Allocated memory for the matrix
  float* matrix = (float*) calloc(args.n * args.m, sizeof(float));
  if (matrix == NULL) {
    fprintf(stderr, "Failed to allocate memory for matrix\n");
    return EXIT_FAILURE;
  }

  // Timing
  double start_time   = get_current_time();
  double cpu_times[2] = {0}; // Array to store timing results
  int    timing_index = 0;

  // Iterations to calculate heat propagation
  printf("Performing %d iterations...\n", args.iterations);
  iterations(args.iterations, args.n, args.m, matrix);
  cpu_times[timing_index++] = get_duration(&start_time);

  printf("Iterations completed in %.6f seconds\n\n", cpu_times[0]);

  // Averages, if requested
  if (args.average) {
    printf("Calculating row averages...\n");

    // Allocate memory for averages
    float* averages = (float*) calloc(args.n, sizeof(float));
    if (averages == NULL) {
      fprintf(stderr, "Failed to allocate memory for averages\n");
      free(matrix);
      return EXIT_FAILURE;
    }

    // Calculate averages
    start_time = get_current_time();
    average_rows(args.n, args.m, matrix, averages);
    cpu_times[timing_index++] = get_duration(&start_time);

    printf("Row averages calculated in %.6f seconds\n\n", cpu_times[1]);
    free(averages);
  }

  // Free the matrix
  free(matrix);

  return EXIT_SUCCESS;
}
