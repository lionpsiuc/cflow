#pragma once

#include "../precision.h"

/**
 * @brief Explain briefly.
 *
 * @param iters Explain briefly.
 * @param n Explain briefly.
 * @param m Explain briefly.
 * @param dst Explain briefly.
 * @param src Explain briefly.
 *
 * @return Explain briefly.
 */
void heat_propagation(const int iters, const int n, const int m,
                      PRECISION* restrict dst, PRECISION* restrict src);
