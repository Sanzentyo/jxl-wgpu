// Independent ICC.1:2022 Annex F / Table 68 oracle, using Little CMS tag decoding.
// Matrix inversion uses pivoted elimination and curve inversion uses analytical roots or
// exhaustive segment search, independent of the Rust/WGSL parser and binary searches.
#pragma once

#include <lcms2.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace scalar {
using Matrix = std::array<std::array<double, 3>, 3>;

struct Curve {
  int type = 0;
  std::array<double, 7> p{};
  std::vector<double> samples;

  explicit Curve(const cmsToneCurve* curve) {
    if (!curve) throw std::runtime_error("missing scalar curve");
    type = cmsGetToneCurveParametricType(curve);
    if (type > 0) {
      const auto* segment = cmsGetToneCurveSegment(0, curve);
      if (!segment || type > 5) throw std::runtime_error("unsupported scalar curve");
      std::copy(segment->Params, segment->Params + 7, p.begin());
    } else {
      const size_t count = cmsGetToneCurveEstimatedTableEntries(curve);
      const auto* table = cmsGetToneCurveEstimatedTable(curve);
      if (count < 2 || !table) throw std::runtime_error("missing scalar table");
      for (size_t i = 0; i < count; ++i) samples.push_back(table[i] / 65535.0);
    }
  }

  double Forward(double x) const {
    x = std::clamp(x, 0.0, 1.0);
    if (type == 0) {
      const double position = x * static_cast<double>(samples.size() - 1);
      const size_t left = std::min(static_cast<size_t>(position), samples.size() - 2);
      return samples[left] + (samples[left + 1] - samples[left]) * (position - left);
    }
    double y;
    if (type == 1) y = std::pow(x, p[0]);
    else if (type <= 3) {
      const double offset = type == 3 ? p[3] : 0.0;
      y = offset + (x < -p[2] / p[1] ? 0.0 : std::pow(std::max(0.0, p[1] * x + p[2]), p[0]));
    } else if (x < p[4]) y = p[3] * x + (type == 5 ? p[6] : 0.0);
    else y = std::pow(std::max(0.0, p[1] * x + p[2]), p[0]) + (type == 5 ? p[5] : 0.0);
    return std::clamp(y, 0.0, 1.0);
  }

  double Inverse(double y) const {
    y = std::clamp(y, Forward(0.0), Forward(1.0));
    const bool terminal = y == Forward(1.0);
    double best_x = 0;
    double best_error = std::numeric_limits<double>::infinity();
    auto candidate = [&](double x) {
      x = std::clamp(x, 0.0, 1.0);
      const double error = std::abs(Forward(x) - y);
      if (error < best_error || (error == best_error && (terminal ? x < best_x : x > best_x))) {
        best_x = x; best_error = error;
      }
    };
    candidate(0); candidate(1);
    if (type == 0) {
      const double scale = static_cast<double>(samples.size() - 1);
      for (size_t i = 0; i + 1 < samples.size(); ++i) {
        candidate(i / scale); candidate((i + 1) / scale);
        if (samples[i + 1] > samples[i]) {
          const double t = std::clamp((y - samples[i]) / (samples[i + 1] - samples[i]), 0.0, 1.0);
          candidate((i + t) / scale);
        }
      }
    } else if (type == 1) candidate(std::pow(y, 1.0 / p[0]));
    else {
      const double split = std::clamp(type <= 3 ? -p[2] / p[1] : p[4], 0.0, 1.0);
      candidate(split);
      if (split > 0) candidate(std::nextafter(split, 0.0));
      const double offset = type == 3 ? p[3] : (type == 5 ? p[5] : 0.0);
      candidate(std::clamp((std::pow(std::max(0.0, y - offset), 1.0 / p[0]) - p[2]) / p[1], split, 1.0));
      if (type >= 4 && p[3] > 0 && split > 0) {
        const double lower_offset = type == 5 ? p[6] : 0.0;
        candidate(std::clamp((y - lower_offset) / p[3], 0.0, std::nextafter(split, 0.0)));
      }
    }
    return best_x;
  }
};

struct Profile {
  std::vector<Curve> curves;
  Matrix matrix{};
  explicit Profile(cmsHPROFILE profile) {
    if (cmsGetColorSpace(profile) == cmsSigGrayData) {
      curves.emplace_back(static_cast<const cmsToneCurve*>(cmsReadTag(profile, cmsSigGrayTRCTag)));
      matrix[0][0] = 0xf6d6 / 65536.0;
      matrix[1][0] = 1;
      matrix[2][0] = 0xd32d / 65536.0;
    } else {
      const std::array<cmsTagSignature, 3> curve_tags{cmsSigRedTRCTag, cmsSigGreenTRCTag, cmsSigBlueTRCTag};
      const std::array<cmsTagSignature, 3> matrix_tags{cmsSigRedColorantTag, cmsSigGreenColorantTag, cmsSigBlueColorantTag};
      for (size_t c = 0; c < 3; ++c) {
        curves.emplace_back(static_cast<const cmsToneCurve*>(cmsReadTag(profile, curve_tags[c])));
        const auto* xyz = static_cast<const cmsCIEXYZ*>(cmsReadTag(profile, matrix_tags[c]));
        if (!xyz) throw std::runtime_error("missing scalar colorants");
        matrix[0][c] = xyz->X; matrix[1][c] = xyz->Y; matrix[2][c] = xyz->Z;
      }
    }
  }
};

inline Matrix Invert(Matrix input) {
  Matrix output{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  for (size_t column = 0; column < 3; ++column) {
    size_t pivot = column;
    for (size_t r = column + 1; r < 3; ++r) if (std::abs(input[r][column]) > std::abs(input[pivot][column])) pivot = r;
    std::swap(input[pivot], input[column]); std::swap(output[pivot], output[column]);
    const double scale = input[column][column];
    if (scale == 0) throw std::runtime_error("singular scalar matrix");
    for (size_t c = 0; c < 3; ++c) { input[column][c] /= scale; output[column][c] /= scale; }
    for (size_t r = 0; r < 3; ++r) if (r != column) {
      const double factor = input[r][column];
      for (size_t c = 0; c < 3; ++c) { input[r][c] -= factor * input[column][c]; output[r][c] -= factor * output[column][c]; }
    }
  }
  return output;
}

struct References {
  std::vector<float> exact, lower, upper, native_lower, native_upper;
  std::vector<uint8_t> native_semantics;
};

inline References Convert(const Profile& source, const Profile& target, const std::vector<float>& input) {
  Matrix inverse{};
  if (target.curves.size() == 1) inverse[0][1] = 1;
  else inverse = Invert(target.matrix);
  Matrix matrix{};
  for (size_t r = 0; r < 3; ++r) for (size_t c = 0; c < 3; ++c) for (size_t k = 0; k < 3; ++k) matrix[r][c] += inverse[r][k] * source.matrix[k][c];
  if (source.matrix == target.matrix && source.curves.size() == target.curves.size()) {
    matrix = Matrix{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  }
  References result;
  for (size_t pixel = 0; pixel < input.size() / source.curves.size(); ++pixel) {
    std::array<double, 3> linear{};
    bool source_boundary = false;
    for (size_t c = 0; c < source.curves.size(); ++c) {
      const auto& curve = source.curves[c];
      const double x = input[pixel * source.curves.size() + c];
      linear[c] = curve.Forward(x);
      // cmsgamma.c case 3 uses zero when the power base is exactly zero, losing +c.
      source_boundary |= curve.type == 3 && x >= -curve.p[2] / curve.p[1] && curve.p[1] * x + curve.p[2] == 0.0;
    }
    for (size_t r = 0; r < target.curves.size(); ++r) {
      const auto& curve = target.curves[r];
      double value = 0, magnitude = 0, coefficient_sum = 0;
      for (size_t c = 0; c < source.curves.size(); ++c) {
        value += matrix[r][c] * linear[c];
        magnitude += std::abs(matrix[r][c] * linear[c]);
        coefficient_sum += std::abs(matrix[r][c]);
      }
      // Bound F32 curve arithmetic and matrix dot products before the inverse. The final
      // inverse may be arbitrarily steep; a fixed output-code tolerance is inappropriate.
      const double uncertainty = 4e-7 * (1.0 + magnitude + coefficient_sum);
      result.exact.push_back(static_cast<float>(curve.Inverse(value)));
      result.lower.push_back(static_cast<float>(curve.Inverse(value - uncertainty) - 2e-7));
      result.upper.push_back(static_cast<float>(curve.Inverse(value + uncertainty) + 2e-7));
      // Little CMS retains 16-bit sampled curves and a 4096-entry reverse approximation.
      const double native_uncertainty = uncertainty + (3.0 / 65535.0) * coefficient_sum;
      const double reverse_error = curve.type == 0 ? 2.0 / 4095.0 : 2e-7;
      result.native_lower.push_back(static_cast<float>(curve.Inverse(value - native_uncertainty) - reverse_error));
      result.native_upper.push_back(static_cast<float>(curve.Inverse(value + native_uncertainty) + reverse_error));
      result.native_semantics.push_back(static_cast<uint8_t>((source_boundary ? 1 : 0) | ((curve.type == 2 && value < 0) ? 2 : 0)));
    }
  }
  return result;
}
}  // namespace scalar
