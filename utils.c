#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
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
  return args;
}

// Parses command-line arguments
arguments parse(const int argc, char* const argv[]) {
  arguments args = defaults(); // Start with the default arguments

  // Parse arguments
  const char list[] = "n:m:p:ah";
  int        flag;
  while ((flag = getopt(argc, argv, list)) != -1) {
    switch (flag) {
      case 'n': parse_argument(flag, &(args.n), "%d"); break;
      case 'm': parse_argument(flag, &(args.m), "%d"); break;
      case 'p': parse_argument(flag, &(args.iters), "%d"); break;
      case 'a': args.average = true; break;
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
  struct timeval current_time;
  gettimeofday(&current_time, NULL);
  return (double) (current_time.tv_sec + current_time.tv_usec * 1e-6);
}

// Calculates the duration since the given time and updates the time to now
double get_duration(double* const time) {
  const double diff = get_current_time() - *time;
  *time             = get_current_time();
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
