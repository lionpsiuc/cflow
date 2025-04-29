#include "../include/utils.h"

/**
 * @brief Parses a command-line argument value associated with a flag.
 *
 * Uses sscanf to parse the global optarg (which is set by getopt) according to
 * the provided format string and stores the result in variable. Prints an error
 * message and exits if parsing fails.
 *
 * @param[in]  flag     The command-line flag character (e.g., a, c, m, n, p,
 *                      or t).
 * @param[out] variable Pointer to the variable to store the parsed value.
 * @param[in]  format   The sscanf format string for parsing (e.g., %d).
 */
static void parse_argument(const char flag, void* const variable,
                           const char format[]) {
  if (sscanf(optarg, format, variable) != 1) {
    fprintf(stderr, "Error: Couldn't read -%c argument\n", flag);
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Provides default values for command-line arguments.
 *
 * Creates an arguments struct and initialises its fields with predefined
 * default settings for matrix dimensions, iterations, and flags.
 *
 * @return An arguments struct containing the default settings.
 */
static arguments defaults(void) {
  arguments args;
  args.n       = 32;
  args.m       = 32;
  args.iters   = 10;
  args.average = false;
  args.cpu     = false;
  args.timing  = false;
  return args;
}

/**
 * @brief Parses command-line arguments for the application.
 *
 * Initialises an arguments struct with defaults, then uses getopt to parse
 * command-line flags (e.g., -a, -c, -m, -n, -p, or -t). It calls parse_argument
 * for flags requiring values. Performs basic validation (e.g., positive
 * dimensions, non-negative iterations).
 *
 * @param[in] argc The argument count passed to main.
 * @param[in] argv The argument vector passed to main.
 *
 * @return An arguments struct populated with parsed or default values.
 */
arguments parse(const int argc, char* const argv[]) {
  arguments args = defaults(); // Start with the default arguments

  // Parse arguments
  const char list[] = "n:m:p:acth";
  int        flag;
  while ((flag = getopt(argc, argv, list)) != -1) {
    switch (flag) {
      case 'n': parse_argument(flag, &(args.n), "%d"); break;
      case 'm': parse_argument(flag, &(args.m), "%d"); break;
      case 'p': parse_argument(flag, &(args.iters), "%d"); break;
      case 'a': args.average = true; break;
      case 'c': args.cpu = true; break;
      case 't': args.timing = true; break;
      default: exit(EXIT_FAILURE); break;
    }
  }

  // Some extra checks
  if (args.n <= 0) {
    fprintf(stderr, "Error: The matrix must have positive height\n");
    exit(EXIT_FAILURE);
  }
  if (args.m <= 0) {
    fprintf(stderr, "Error: The matrix must have positive width\n");
    exit(EXIT_FAILURE);
  }
  if (args.iters < 0) {
    fprintf(stderr, "Error: The number of iterations must be non-negative\n");
    exit(EXIT_FAILURE);
  }

  return args;
}

/**
 * @brief Gets the current high-resolution monotonic time.
 *
 * Uses clock_gettime with CLOCK_MONOTONIC to retrieve the current time.
 *
 * @return The current time in seconds, as a double-precision float.
 */
double get_current_time(void) {
  struct timespec current_time;
  clock_gettime(CLOCK_MONOTONIC, &current_time);
  return (double) current_time.tv_sec + (double) current_time.tv_nsec * 1e-9;
}

/**
 * @brief Calculates elapsed time and updates the start time.
 *
 * Gets the current time, calculates the difference between the current time and
 * the time pointed to by the input time pointer, updates the value pointed to
 * by time to the current time, and returns the difference.
 *
 * @param[in,out] time Pointer to a double storing a previous time value; this
 *                     value is updated to the current time upon exit.
 *
 * @return The elapsed time duration in seconds.
 */
double get_duration(double* const time) {
  const double now  = get_current_time();
  const double diff = now - *time;
  *time             = now;
  return diff;
}

/**
 * @brief Prints a matrix to standard output.
 *
 * Iterates through the specified rows and columns of the matrix data pointed to
 * by dst and prints each element formatted to six decimal places. Uses
 * increment to correctly calculate the position of elements in padded rows.
 *
 * @param[in] n         The number of rows to print.
 * @param[in] m         The number of columns to print per row.
 * @param[in] increment The distance (i.e., number of elements) between starts
 *                      of rows.
 * @param[in] dst       Pointer to the matrix data to be printed.
 */
void print_matrix(const int n, const int m, const int increment,
                  float* const dst) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%8.6f   ", dst[i * increment + j]);
    }
    printf("\n");
  }
}

/**
 * @brief Counts mismatches between two matrices within a tolerance.
 *
 * Compares corresponding elements of matrices A and B. If the absolute
 * difference between an element pair is greater than or equal to a predefined
 * tolerance, it increments a counter.
 *
 * @param[in] n          The number of rows to compare.
 * @param[in] m          The number of columns to compare per row.
 * @param[in] incrementA The row increment (i.e., stride) for matrix A.
 * @param[in] A          Pointer to the first matrix data.
 * @param[in] incrementB The row increment (i.e., stride) for matrix B.
 * @param[in] B          Pointer to the second matrix data.
 *
 * @return The total number of element pairs whose difference exceeds tolerance.
 */
int mismatches(const int n, const int m, const int incrementA,
               const float* const A, const int incrementB,
               const float* const B) {
  int         count = 0;
  const float tol   = 1e-4f;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (fabsf(A[i * incrementA + j] - B[i * incrementB + j]) >= tol) {
        count++;
      }
    }
  }
  return count;
}

/**
 * @brief Finds the maximum absolute difference between two matrices.
 *
 * Compares corresponding elements of matrices A and B and returns the largest
 * absolute difference found between any element pair.
 *
 * @param[in] n          The number of rows to compare.
 * @param[in] m          The number of columns to compare per row.
 * @param[in] incrementA The row increment (i.e., stride) for matrix A.
 * @param[in] A          Pointer to the first matrix data.
 * @param[in] incrementB The row increment (i.e., stride) for matrix B.
 * @param[in] B          Pointer to the second matrix data.
 *
 * @return The maximum absolute difference found between corresponding elements.
 */
float maxdiff(const int n, const int m, const int incrementA,
              const float* const A, const int incrementB,
              const float* const B) {
  float maxdiff  = 0;
  float currdiff = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      currdiff = fabsf(A[i * incrementA + j] - B[i * incrementB + j]);
      if (currdiff > maxdiff) {
        maxdiff = currdiff;
      }
    }
  }
  return maxdiff;
}
