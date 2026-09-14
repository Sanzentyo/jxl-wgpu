#pragma once
#include <icc/linear.hpp>

namespace intents {

// Only classifies a native-CMM boundary difference. The authoritative scalar
// evaluation remains Curve::Forward/Inverse, including ICC.1:2022 10.18's
// unit-domain/range clipping.
inline double UnclippedForward(const scalar::Curve &curve, double x) {
  x = std::clamp(x, 0.0, 1.0);
  const auto &p = curve.p;
  if (curve.type <= 1)
    return curve.Forward(x);
  if (curve.type <= 3)
    return (curve.type == 3 ? p[3] : 0.0) +
           (x < -p[2] / p[1] ? 0.0
                             : std::pow(std::max(0.0, p[1] * x + p[2]), p[0]));
  if (x < p[4])
    return p[3] * x + (curve.type == 5 ? p[6] : 0.0);
  return std::pow(std::max(0.0, p[1] * x + p[2]), p[0]) +
         (curve.type == 5 ? p[5] : 0.0);
}

inline std::array<double, 3> Black(const scalar::Profile &profile) {
  std::array<double, 3> value{};
  for (unsigned row = 0; row < 3; ++row)
    for (size_t channel = 0; channel < profile.curves.size(); ++channel)
      value[row] +=
          profile.matrix[row][channel] * profile.curves[channel].Forward(0);
  // ICC D50 Lab's piecewise functions. Native BPC clips lightness while
  // retaining chromatic components; evaluate only profile black metadata, never
  // image pixels.
  constexpr std::array<double, 3> white{0.9642, 1, 0.8249};
  const auto f = [](double t) {
    return t > 216.0 / 24389.0 ? std::cbrt(t) : (24389.0 / 27.0 * t + 16) / 116;
  };
  const auto inverse_f = [](double t) {
    return t > 6.0 / 29.0 ? t * t * t : (116 * t - 16) * 27.0 / 24389.0;
  };
  const double fy = f(value[1]);
  const double lightness = 116 * fy - 16;
  const double clipped = lightness > 95 ? 0 : std::clamp(lightness, 0.0, 50.0);
  if (clipped != lightness) {
    const double delta = (clipped + 16) / 116 - fy;
    for (unsigned c = 0; c < 3; ++c)
      value[c] = white[c] * inverse_f(f(value[c] / white[c]) + delta);
  }
  return value;
}

struct Endpoint {
  const scalar::Profile *profile;
  scalar::Matrix matrix;

  explicit Endpoint(const scalar::Profile &value)
      : profile(&value), matrix(value.matrix) {}
  explicit Endpoint(const connection::LinearSpace &space)
      : profile(nullptr), matrix(space.ToPcs()) {}
  size_t Channels() const { return profile ? profile->curves.size() : 3; }
  std::array<double, 3> BlackPoint() const {
    return profile ? Black(*profile) : std::array<double, 3>{};
  }
};

inline scalar::References Affine(const Endpoint &source,
                                 const Endpoint &target,
                                 const std::vector<float> &input,
                                 bool compensate) {
  const auto source_black =
      compensate ? source.BlackPoint() : std::array<double, 3>{};
  const auto target_black =
      compensate ? target.BlackPoint() : std::array<double, 3>{};
  const auto dropped_black_offset = [](const Endpoint &endpoint) {
    return endpoint.profile &&
           std::any_of(endpoint.profile->curves.begin(), endpoint.profile->curves.end(),
                       [](const scalar::Curve &curve) {
                         return curve.type == 3 && curve.p[2] == 0 &&
                                curve.p[3] != 0;
                       });
  };
  const bool black_boundary = compensate && (dropped_black_offset(source) ||
                                             dropped_black_offset(target));
  constexpr std::array<double, 3> white{0.9642, 1, 0.8249};
  std::array<double, 3> scale{}, pcs_offset{}, offset{};
  for (unsigned c = 0; c < 3; ++c) {
    scale[c] = (white[c] - target_black[c]) / (white[c] - source_black[c]);
    pcs_offset[c] = target_black[c] - scale[c] * source_black[c];
  }
  scalar::Matrix inverse{}, matrix{};
  if (target.Channels() == 1)
    inverse[0][1] = 1;
  else
    inverse = scalar::Invert(target.matrix);
  for (unsigned r = 0; r < 3; ++r) {
    for (unsigned k = 0; k < 3; ++k) {
      offset[r] += inverse[r][k] * pcs_offset[k];
      for (unsigned c = 0; c < 3; ++c)
        matrix[r][c] += inverse[r][k] * scale[k] * source.matrix[k][c];
    }
  }
  if (source.matrix == target.matrix &&
      source.Channels() == target.Channels() &&
      source_black == target_black)
    matrix = scalar::Matrix{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  scalar::References result;
  for (size_t pixel = 0; pixel < input.size() / source.Channels(); ++pixel) {
    std::array<double, 3> linear{};
    bool source_boundary = false;
    bool source_range = false;
    for (size_t c = 0; c < source.Channels(); ++c) {
      const double x = input[pixel * source.Channels() + c];
      if (!source.profile) {
        linear[c] = x;
        continue;
      }
      const auto &curve = source.profile->curves[c];
      linear[c] = curve.Forward(x);
      source_boundary |= curve.type == 3 && x >= -curve.p[2] / curve.p[1] &&
                         curve.p[1] * x + curve.p[2] == 0.0;
      const double raw = UnclippedForward(curve, x);
      source_range |= raw < 0 || raw > 1;
    }

    for (size_t r = 0; r < target.Channels(); ++r) {
      const auto *curve = target.profile ? &target.profile->curves[r] : nullptr;
      const auto output = [curve](double value) {
        return curve ? curve->Inverse(value) : value;
      };
      double value = offset[r], magnitude = std::abs(offset[r]), sum = 0;
      for (size_t c = 0; c < source.Channels(); ++c) {
        value += matrix[r][c] * linear[c];
        magnitude += std::abs(matrix[r][c] * linear[c]);
        sum += std::abs(matrix[r][c]);
      }
      const double error = 4e-7 * (1 + magnitude + sum);
      const double native_error = error + (3.0 / 65535.0) * sum;
      const double reverse_error = curve && curve->type == 0 ? 2.0 / 4095.0 : 2e-7;
      result.exact.push_back(static_cast<float>(output(value)));
      result.lower.push_back(
          static_cast<float>(output(value - error) - 2e-7));
      result.upper.push_back(
          static_cast<float>(output(value + error) + 2e-7));
      result.native_lower.push_back(static_cast<float>(
          output(value - native_error) - reverse_error));
      result.native_upper.push_back(static_cast<float>(
          output(value + native_error) + reverse_error));
      // Native arithmetic can cross a clipped endpoint within its existing
      // uncertainty. Its extrapolated inverse then differs even though the
      // bounded ICC inverse is flat.
      const bool inverse_range = curve &&
          ((value + native_error > 1 && UnclippedForward(*curve, 1) > 1) ||
           (value - native_error < 0 && UnclippedForward(*curve, 0) < 0));
      result.native_semantics.push_back(static_cast<uint8_t>(
          (source_boundary ? 1 : 0) | ((curve && curve->type == 2 && value < 0) ? 2 : 0) |
          (inverse_range ? 4 : 0) | (source_range ? 8 : 0) |
          (black_boundary ? 16 : 0)));
    }
  }
  return result;
}

inline std::array<double, 3> MediaWhite(cmsHPROFILE profile) {
  if (cmsGetEncodedICCversion(profile) < 0x04000000 &&
      cmsGetDeviceClass(profile) == cmsSigDisplayClass)
    return {0.9642, 1, 0.8249};
  const auto *white = static_cast<const cmsCIEXYZ *>(
      cmsReadTag(profile, cmsSigMediaWhitePointTag));
  if (white == nullptr)
    throw std::runtime_error("ICC media white tag");
  return {white->X, white->Y, white->Z};
}

inline scalar::References Convert(cmsHPROFILE source, cmsHPROFILE target,
                                  const std::vector<float> &input,
                                  unsigned intent) {
  scalar::Profile src(source), dst(target);
  if (intent == INTENT_ABSOLUTE_COLORIMETRIC) {
    const auto sw = MediaWhite(source), tw = MediaWhite(target);
    for (unsigned r = 0; r < 3; ++r)
      for (unsigned c = 0; c < 3; ++c)
        src.matrix[r][c] *= sw[r] / tw[r];
  }
  const bool compensate =
      (intent == INTENT_PERCEPTUAL || intent == INTENT_SATURATION) &&
      cmsGetEncodedICCversion(target) >= 0x04000000;
  return Affine(Endpoint(src), Endpoint(dst), input, compensate);
}

inline scalar::References LinearReference(cmsHPROFILE profile,
                                          const connection::LinearSpace &space,
                                          const std::vector<float> &input,
                                          bool to_linear, unsigned intent) {
  const scalar::Profile device(profile);
  Endpoint source = to_linear ? Endpoint(device) : Endpoint(space);
  const Endpoint target = to_linear ? Endpoint(space) : Endpoint(device);
  if (intent == INTENT_ABSOLUTE_COLORIMETRIC) {
    constexpr std::array<double, 3> white{0.9642, 1, 0.8249};
    const auto media = MediaWhite(profile);
    for (unsigned r = 0; r < 3; ++r)
      for (unsigned c = 0; c < 3; ++c)
        source.matrix[r][c] *= to_linear ? media[r] / white[r] : white[r] / media[r];
  }
  const bool compensate =
      (intent == INTENT_PERCEPTUAL || intent == INTENT_SATURATION) &&
      (to_linear || cmsGetEncodedICCversion(profile) >= 0x04000000);
  return Affine(source, target, input, compensate);
}
} // namespace intents
