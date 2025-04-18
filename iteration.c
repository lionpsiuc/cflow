void set_initial_conditions(const int n, const int m, float* const matrix) {

  // Add two extra columns in order to simplify the 'wrapping around' problem
  const int row_length = n + 2;

  // Loop over rows and set the initial values
  for (int i = 0; i < n; i++) {
    float col0 = 0.98f * (float) ((i + 1) * (i + 1)) / (float) (n * n);
    matrix[i * row_length + 0] = col0; // We add + 0 for consistency

    // Set interior points
    for (int j = 1; j < m; j++) {
      matrix[i * row_length + j] =
          col0 * ((float) (m - j) * (m - j)) / (float) (m * m);
    }

    // We set the extra columns here
    matrix[i * row_length + m + 0] = matrix[i * row_length + 0];
    matrix[i * row_length + m + 1] = matrix[i * row_length + 1];
  }

  return;
}

void heat_propagation(const int m, float* const restrict new,
                      const float* const restrict old) {
  for (int j = 1; j < width;
       j++) { // Start at j = 1, since the first column is fixed
    new[j] = ((1.60f * old[j - 2]) + (1.55f * old[j - 1]) + old[j] +
              (0.60f * old[j + 1]) + (0.25f * old[j + 2])) /
             5.0f; // Formula as per the assignment instructions
  }
  return;
}
