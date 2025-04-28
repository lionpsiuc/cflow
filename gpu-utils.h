#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Gets the maximum shared memory per block usable by the currently
 *        selected device. Taken from your sample code.
 *
 * @return Maximum shared memory in bytes, or negative on error.
 */
int max_shared_memory_per_block(void);

/**
 * @brief Gets the compute capability of the currently selected device.
 *
 * @return Compute capability value, or negative on error.
 */
int current_cc(void);

#ifdef __cplusplus
}
#endif

#endif // GPU_UTILS_H
