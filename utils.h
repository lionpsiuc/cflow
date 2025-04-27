#ifndef UTILS_H
#define UTILS_H

typedef struct {
  int  n;       // Number of rows (i.e., height)
  int  m;       // Number of columns (i.e., width)
  int  iters;   // Number of iterations
  bool average; // Whether to compute the average (i.e., the temperature)
  bool cpu;     // Whether to skip CPU computations
  bool timing;  // Whether to show timing
} arguments;

arguments parse(const int argc, char* const argv[]);
double    get_current_time(void);
double    get_duration(double* const time);
void      print_matrix(const int n, const int m, const int increment,
                       float* const dst);
int       mismatches(const int n, const int m, const int incrementA,
                     const float* const A, const int incrementB,
                     const float* const B);
float     maxdiff(const int n, const int m, const int incrementA,
                  const float* const A, const int incrementB, const float* const B);

#endif // UTILS_H
