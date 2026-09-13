// Propagate the original-color corpus's normalized XYB reconstruction precision.
// This oracle contains no production Rust/WGSL calculations or GPU-derived values.
#pragma once
#include <icc/linear.hpp>
#include <limits>

struct Bounds {
  std::vector<float> lower, upper;
};

inline Bounds ReconstructionBounds(const scalar::Profile& profile,
                                    const std::vector<float>& linear) {
  std::vector<float> corners;
  for (size_t pixel = 0; pixel < linear.size() / 3; ++pixel) {
    for (unsigned mask = 0; mask < 8; ++mask) for (unsigned c = 0; c < 3; ++c) {
      const double value = linear[3 * pixel + c];
      const double sign = mask & (1u << c) ? 1 : -1;
      const double error = (1 + std::abs(value)) / 1024;
      corners.push_back(std::nextafter(static_cast<float>(value + sign * error),
          static_cast<float>(sign) * std::numeric_limits<float>::infinity()));
    }
  }
  const auto converted = connection::Reference(profile, connection::kSpaces[0], corners, false);
  const size_t colors = profile.curves.size();
  Bounds result;
  for (size_t pixel = 0; pixel < linear.size() / 3; ++pixel) for (size_t c = 0; c < colors; ++c) {
    float lower = std::numeric_limits<float>::infinity();
    float upper = -lower;
    for (unsigned corner = 0; corner < 8; ++corner) {
      const size_t index = (pixel * 8 + corner) * colors + c;
      lower = std::min(lower, converted.lower[index]);
      upper = std::max(upper, converted.upper[index]);
    }
    // The RGB-to-PCS matrix is affine and the inverse curves are increasing.
    // Every extremum, including the independent arithmetic bound, is at a corner.
    result.lower.push_back(lower);
    result.upper.push_back(upper);
  }
  return result;
}
