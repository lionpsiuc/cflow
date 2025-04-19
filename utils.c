#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void print_help() {
  printf("Usage: ");
  printf("./assignment2 ");
  printf("-n <rows> ");
  printf("-m <columns> ");
  printf("-p <iterations> ");
  printf("-a ");
  printf("-h\n");
}

void read(const char flag, void* const variable, const char format[]) {
  if (sscanf(optarg, format, variable) != 1) {
    fprintf(stderr, "Couldn't read -%c argument\n", flag);
    print_help();
    exit(EXIT_FAILURE);
  }
}

typedef struct {
  int  n;
  int  m;
  int  iterations;
  bool average;
} arguments;

arguments defaults(void) {
  arguments args;
  args.n          = 32;
  args.m          = 32;
  args.iterations = 10;
  args.average    = false;
  return args;
}

arguments parse(const int argc, char* const argv[]) {
  arguments args = defaults(); // Start with the default arguments

  // Parse arguments
  const char list[] = "n:m:p:ah";
  int        flag;
  while ((flag = getopt(argc, argv, list)) != -1) {
    switch (flag) {
      case 'n': read(flag, &(args.n), "%d"); break;
      case 'm': read(flag, &(args.m), "%d"); break;
      case 'p': read(flag, &(args.iterations), "%d"); break;
      case 'a': args.average = true; break;
      case 'h':
        print_help();
        exit(EXIT_SUCCESS);
        break;
      default:
        print_help();
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
  if (args.iterations < 0) {
    fprintf(stderr, "Number of iterations must be non-negative\n");
    exit(EXIT_FAILURE);
  }

  return args;
}
