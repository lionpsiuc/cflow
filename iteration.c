#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// void set_initial_conditions(const int n, const int m, float* const matrix) {

//   // Add two extra columns in order to simplify the 'wrapping around' problem
//   const int row_length = n + 2;

//   // Loop over rows and set the initial values
//   for (int i = 0; i < n; i++) {
//     float col0 = 0.98f * (float) ((i + 1) * (i + 1)) / (float) (n * n);
//     matrix[i * row_length + 0] = col0; // We add + 0 for consistency

//     // Set interior points
//     for (int j = 1; j < m; j++) {
//       matrix[i * row_length + j] =
//           col0 * ((float) (m - j) * (m - j)) / (float) (m * m);
//     }

//     // We set the extra columns here
//     matrix[i * row_length + m + 0] = matrix[i * row_length + 0];
//     matrix[i * row_length + m + 1] = matrix[i * row_length + 1];
//   }

//   return;
// }

// void heat_propagation(const int m, float* const restrict new,
//                       const float* const restrict old) {
//   for (int j = 1; j < width;
//        j++) { // Start at j = 1, since the first column is fixed
//     new[j] = ((1.60f * old[j - 2]) + (1.55f * old[j - 1]) + old[j] +
//               (0.60f * old[j + 1]) + (0.25f * old[j + 2])) /
//              5.0f; // Formula as per the assignment instructions
//   }
//   return;
// }

void iterations(const int iterations, const int n, const int m, float* matrix) {

  // Allocate a temporary matrix for calculations
  float* temp_matrix = (float*) malloc(n * m * sizeof(float));
  if (temp_matrix == NULL) {
    fprintf(stderr, "Failed to allocate memory for the temporary matrix\n");
    exit(EXIT_FAILURE);
  }

  // Set initial conditions
  for (int i = 0; i < n; i++) {
    float col0        = 0.98f * (float) ((i + 1) * (i + 1)) / (float) (n * n);
    matrix[i * m + 0] = col0; // We add + 0 for consistency

    // Set initial values for other columns
    for (int j = 1; j < m; j++) {
      matrix[i * m + j] = col0 * ((float) (m - j) * (m - j)) / (float) (m * m);
    }
  }

  // Copy initial values to the temporary matrix
  memcpy(temp_matrix, matrix, n * m * sizeof(float));

  // Perform iterations
  for (int iter = 0; iter < iterations; iter++) {
    for (int i = 0; i < height; i++) {
      for (int j = 1; j < m;
           j++) { // Start at j = 1, since the first column is fixed

        // Handle wrapping for points near the left edge
        float old_minus2 =
            (j >= 2) ? matrix[i * m + (j - 2)] : matrix[i * m + (m + j - 2)];
        float old_minus1 =
            (j >= 1) ? matrix[i * m + (j - 1)] : matrix[i * m + (m + j - 1)];

        // Current value
        float old = matrix[i * m + j];

        // Handle wrapping for points near the right edge
        float old_plus1 = (j < m - 1) ? matrix[i * m + (j + 1)]
                                      : matrix[i * m + ((j + 1) % m)];
        float old_plus2 = (j < m - 2) ? matrix[i * m + (j + 2)]
                                      : matrix[i * m + ((j + 2) % m)];

        // Heat propagation
        temp_matrix[i * m + j] =
            ((1.60f * old_minus2) + (1.55f * old_minus1) + old +
             (0.60f * old_plus1) + (0.25f * old_plus2)) /
            5.0f;
      }
    }

    // Copy updated values back to the original matrix, except for the first
    // column, since it is fixed
    for (int i = 0; i < n; i++) {
      for (int j = 1; j < m; j++) {
        matrix[i * m + j] = temp_matrix[i * m + j];
      }
    }
  }

  // Free temporary matrix
  free(temp_matrix);
}
