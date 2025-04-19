#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

typedef struct {
  int  n;       // Number of rows (i.e., height)
  int  m;       // Number of columns (i.e., width)
  int  iters;   // Number of iterations
  bool average; // Whether to compute the average (i.e., the temperature)
} arguments;

arguments parse(const int argc, char* const argv[]);
double    get_current_time(void);
double    get_duration(double* const time);
void      print_matrix(const int n, const int m, const int increment,
                       float* const dst);

#endif // UTILS_H
