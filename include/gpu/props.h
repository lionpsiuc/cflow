#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Explain briefly.
 *
 * @return Explain briefly.
 */
int current_device(void);

/**
 * @brief Explain briefly.
 *
 * @param index Explain briefly.
 *
 * @return Explain briefly.
 */
int compute_capability(int index = -1);

/**
 * @brief Explain briefly.
 *
 * @param index Explain briefly.
 *
 * @return Explain briefly.
 */
int max_grid_dim_x(int index = -1);

/**
 * @brief Explain briefly.
 *
 * @param index Explain briefly.
 *
 * @return Explain briefly.
 */
int max_shared_per_block(int index = -1);

/**
 * @brief Explain briefly.
 *
 * @param index Explain briefly.
 *
 * @return Explain briefly.
 */
int multiprocessor_count(int index = -1);

/**
 * @brief Explain briefly.
 *
 * @param index Explain briefly.
 *
 * @return Explain briefly.
 */
int cuda_core_count(int index = -1);

/**
 * @brief Explain briefly.
 *
 * @param index Explain briefly.
 *
 * @return Explain briefly.
 */
int set_device(int index);

#ifdef __cplusplus
}
#endif
