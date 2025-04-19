// Updated for optimised matrices
int average_rows(const int n, const int m, const int increment,
                 const float* const dst, float* const averages) {

  // Initialise averages to zero
  for (int i = 0; i < n; i++) {
    averages[i] = 0.0f;
  }

  // Calculate sum for each row
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      averages[i] += dst[i * increment + j];
    }
  }

  // Divide by width to get the average
  for (int i = 0; i < n; i++) {
    averages[i] /= m;
  }

  return 0;
}
