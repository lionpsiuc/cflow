#pragma once

#include <math.h>

#ifdef DOUBLE
typedef double PRECISION;
#define abs(input) fabs(input)
#else
typedef float PRECISION;
#define abs(input) fabsf(input)
#endif
