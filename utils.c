#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "utils.h"

// Prints command-line usage
static void help() {
  printf("Usage: ");
  printf("./assignment2 ");
  printf("-n <rows> ");
  printf("-m <columns> ");
  printf("-p <iters> ");
  printf("-a ");
  printf("-c ");
  printf("-t ");
  printf("-h\n");
}

// Reads variables from strings
static void parse_argument(const char flag, void* const variable,
                           const char format[]) {
  if (sscanf(optarg, format, variable) != 1) {
    fprintf(stderr, "Couldn't read -%c argument\n", flag);
    help();
    exit(EXIT_FAILURE);
  }
}

// Creates an arguments struct with default values
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

// Parses command-line arguments
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
      case 'h':
        help();
        exit(EXIT_SUCCESS);
        break;
      default:
        help();
        exit(EXIT_FAILURE);
        break;
    }
  }

  // Some extra checks
  if (args.n <= 0) {
    fprintf(stderr, "Matrix must have positive height\n");
    exit(EXIT_FAILURE);
  }
  if (args.m <= 0) {
    fprintf(stderr, "Matrix must have positive width\n");
    exit(EXIT_FAILURE);
  }
  if (args.iters < 0) {
    fprintf(stderr, "Number of iterations must be non-negative\n");
    exit(EXIT_FAILURE);
  }

  return args;
}

// Returns the current time in seconds
double get_current_time(void) {
  struct timespec current_time;
  clock_gettime(CLOCK_MONOTONIC, &current_time);
  return (double) current_time.tv_sec + (double) current_time.tv_nsec * 1e-9;
}

// Calculates the duration since the given time and updates the time to now
double get_duration(double* const time) {
  const double now  = get_current_time();
  const double diff = now - *time;
  *time             = now;
  return diff;
}

// Prints the matrix
void print_matrix(const int n, const int m, const int increment,
                  float* const dst) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%8.6f   ", dst[i * increment + j]);
    }
    printf("\n");
  }
}

// Count the number of mismatches larger than the given tolerance
int mismatches(const int n, const int m, const int incrementA,
               const float* const A, const int incrementB,
               const float* const B) {
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

// Maximum (absolute) difference
int maxdiff(const int n, const int m, const int incrementA,
            const float* const A, const int incrementB, const float* const B) {
  float maxdiff  = 0;
  float currdiff = 0;
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
