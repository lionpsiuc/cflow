#include "../include/utils.h"

/**
 * @brief Explain briefly.
 *
 * @param flag Explain briefly.
 * @param variable Explain briefly.
 * @param format Explain briefly.
 *
 * @return Explain briefly.
 */
static void parse_argument(const char flag, void* const variable,
                           const char format[]) {
  if (sscanf(optarg, format, variable) != 1) {
    fprintf(stderr, "Error: Couldn't read -%c argument\n", flag);
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Explain briefly.
 *
 * @return Explain briefly.
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
 * @brief Explain briefly.
 *
 * @param argc Explain briefly.
 * @param argv Explain briefly.
 *
 * @return Explain briefly.
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
 * @brief Explain briefly.
 *
 * @return Explain briefly.
 */
double get_current_time(void) {
  struct timespec current_time;
  clock_gettime(CLOCK_MONOTONIC, &current_time);
  return (double) current_time.tv_sec + (double) current_time.tv_nsec * 1e-9;
}

/**
 * @brief Explain briefly.
 *
 * @param time Explain briefly.
 *
 * @return Explain briefly.
 */
double get_duration(double* const time) {
  const double now  = get_current_time();
  const double diff = now - *time;
  *time             = now;
  return diff;
}

/**
 * @brief Explain briefly.
 *
 * @param n Explain briefly.
 * @param m Explain briefly.
 * @param increment Explain briefly.
 * @param dst Explain briefly.
 *
 * @return Explain briefly.
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
 * @brief Explain briefly.
 *
 * @param n Explain briefly.
 * @param m Explain briefly.
 * @param incrementA Explain briefly.
 * @param A Explain briefly.
 * @param incrementB Explain briefly.
 * @param B Explain briefly.
 *
 * @return Explain briefly.
 */
int mismatches(const int n, const int m, const int incrementA,
               const precision* const A, const int incrementB,
               const precision* const B) {
  int          count = 0;
  const double tol   = 1e-4;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (fabs(A[i * incrementA + j] - B[i * incrementB + j]) >= tol) {
        count++;
      }
    }
  }
  return count;
}

/**
 * @brief Explain briefly.
 *
 * @param n Explain briefly.
 * @param m Explain briefly.
 * @param incrementA Explain briefly.
 * @param A Explain briefly.
 * @param incrementB Explain briefly.
 * @param B Explain briefly.
 *
 * @return Explain briefly.
 */
double maxdiff(const int n, const int m, const int incrementA,
               const precision* const A, const int incrementB,
               const precision* const B) {
  double maxdiff  = 0;
  double currdiff = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      currdiff = fabs(A[i * incrementA + j] - B[i * incrementB + j]);
      if (currdiff > maxdiff) {
        maxdiff = currdiff;
      }
    }
  }
  return maxdiff;
}
