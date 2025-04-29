#pragma once

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// #include "precision.h"

/**
 * @brief Structure to hold parsed command-line arguments.
 *
 * This struct encapsulates the configuration parameters provided by the user
 * via command-line flags, controlling the matrix dimensions, number of
 * iterations, and optional behaviours like averaging, skipping CPU computation,
 * and displaying timing information.
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
 * @return arguments An arguments struct populated with parsed or default
 *                   values.
 */
arguments parse(const int argc, char* const argv[]);

/**
 * @brief Gets the current high-resolution monotonic time.
 *
 * Uses clock_gettime with CLOCK_MONOTONIC to retrieve the current time.
 *
 * @return double The current time in seconds, as a double-precision float.
 */
double get_current_time(void);

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
 * @return double The elapsed time duration in seconds.
 */
double get_duration(double* const time);

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
                  float* const dst);

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
 * @return int The total number of element pairs whose difference exceeds
 *             tolerance.
 */
int mismatches(const int n, const int m, const int incrementA,
               const float* const A, const int incrementB,
               const float* const B);

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
 * @return float The maximum absolute difference found between corresponding
 *               elements.
 */
float maxdiff(const int n, const int m, const int incrementA,
              const float* const A, const int incrementB, const float* const B);
