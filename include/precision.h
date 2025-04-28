#ifndef PRECISION_H
#define PRECISION_H

#include <math.h>

#ifdef DOUBLE
typedef double precision;
#define abs(input) fabs(input)
#else
typedef float precision;
#define abs(input) fabsf(input)
#endif

#endif // PRECISION_H
