#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>

typedef struct {
  int  n;
  int  m;
  int  iterations;
  bool average;
} arguments;

void print_help(void);
void parse_argument(const char flag, void* const variable, const char format[]);
arguments defaults(void);
arguments parse(const int argc, char* const argv[]);
double    get_current_time(void);
double    get_duration(double* const time);
void      print_matrix(const int n, const int m, const float* const matrix);

#endif // UTILS_H
