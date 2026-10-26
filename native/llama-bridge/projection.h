#pragma once
#include <cstddef>
#include <cmath>
#include <stdexcept>

namespace synapse {
// A unit residual direction, not a neuron coordinate or a calibrated add dose.
inline void validate_direction(const float *u, int width, float strength) {
  if (!std::isfinite(strength) || std::abs(strength) > 1.f)
    throw std::runtime_error("Projection strength must be finite and within [-1, 1]");
  if (!u) {
    if (strength != 0.f)
      throw std::runtime_error("Nonzero projection requires a direction");
    return;
  }
  double norm2 = 0;
  for (int h = 0; h < width; ++h) {
    if (!std::isfinite(u[h]))
      throw std::runtime_error("Non-finite projection direction");
    norm2 += double(u[h]) * u[h];
  }
  if (std::abs(norm2 - 1.) > 1e-4)
    throw std::runtime_error("Projection direction must have unit norm");
}

// stats: token visits, squared correction norm, squared original write norm.
inline void project_write(float *y, int width, int tokens, const float *u,
                          float strength, double *stats,
                          const double *coefficients = nullptr) {
  for (int j = 0; j < tokens; ++j) {
    float *row = y + std::size_t(j) * width;
    double dot = 0, norm2 = 0;
    for (int h = 0; h < width; ++h) {
      if (!std::isfinite(row[h]))
        throw std::runtime_error("Non-finite residual write");
      dot += double(row[h]) * u[h];
      norm2 += double(row[h]) * row[h];
    }
    if (coefficients)
      dot = coefficients[j];
    if (!std::isfinite(dot))
      throw std::runtime_error("Non-finite low-rank correction");
    double correction = double(strength) * dot;
    if (strength != 0.f) // preserve exact identity at zero
      for (int h = 0; h < width; ++h) {
        row[h] = float(double(row[h]) - correction * u[h]);
        if (!std::isfinite(row[h]))
          throw std::runtime_error("Projection overflowed a residual write");
      }
    stats[0] += 1;
    stats[1] += correction * correction;
    stats[2] += norm2;
  }
}
} // namespace synapse
