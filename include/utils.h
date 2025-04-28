#pragma once

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "precision.h"

/**
 * @brief Explain briefly.
 */
typedef struct {
  int  n;       // Number of rows (i.e., height)
  int  m;       // Number of columns (i.e., width)
  int  iters;   // Number of iterations
  bool average; // Whether to compute the average (i.e., the temperature)
  bool cpu;     // Whether to skip CPU computations
  bool timing;  // Whether to show timing
} arguments;

/**
 * @brief Explain briefly.
 *
 * @param argc Explain briefly.
 * @param argv Explain briefly.
 *
 * @return Explain briefly.
 */
arguments parse(const int argc, char* const argv[]);

/**
 * @brief Explain briefly.
 *
 * @return Explain briefly.
 */
double get_current_time(void);

/**
 * @brief Explain briefly.
 *
 * @param time Explain briefly.
 *
 * @return Explain briefly.
 */
double get_duration(double* const time);

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
                  PRECISION* const dst);

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
               const PRECISION* const A, const int incrementB,
               const PRECISION* const B);

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
               const PRECISION* const A, const int incrementB,
               const PRECISION* const B);
