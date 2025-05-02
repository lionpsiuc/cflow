#pragma once

#include <math.h>

/**
 * @brief Defines floating-point precision type and absolute value macro.
 *
 * This header allows the code to be compiled for either single or double
 * precision floating-point numbers. If the preprocessor macro DOUBLE is defined
 * during compilation, PRECISION is defined as double and abs maps to fabs.
 * Otherwise, PRECISION defaults to float and abs maps to fabsf.
 */
#ifdef DOUBLE
typedef double PRECISION;
#define abs(input) fabs(input)
#else
typedef float PRECISION;
#define abs(input) fabsf(input)
#endif
