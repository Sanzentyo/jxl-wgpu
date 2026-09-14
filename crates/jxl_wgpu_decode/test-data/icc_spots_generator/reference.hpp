// Independent interval colorimetry. Little CMS only decodes tags or supplies
// the separate native comparison; production Rust/WGSL code is never used to
// generate expectations.
#pragma once
#include <icc/linear.hpp>

namespace reference {
struct Range {
  double low = 0, high = 0;
  static Range Around(double center, double error) {
    return {center - error, center + error};
  }
};
using Color = std::array<Range, 3>;
inline Range operator+(Range a, Range b) {
  return {a.low + b.low, a.high + b.high};
}
inline Range operator*(Range a, Range b) {
  const std::array<double, 4> products{a.low * b.low, a.low * b.high,
                                       a.high * b.low, a.high * b.high};
  return {*std::min_element(products.begin(), products.end()),
          *std::max_element(products.begin(), products.end())};
}
inline double Linear(double value) {
  const double a = std::abs(value);
  return std::copysign(
      a <= 0.04045 ? a / 12.92 : std::pow((a + 0.055) / 1.055, 2.4), value);
}
struct Source {
  const scalar::Profile *device;
  const connection::LinearSpace &space;
  bool linear;
};
struct Connection {
  Source source;
  const scalar::Profile *target;
  bool identity;
  scalar::Matrix matrix;
  Connection(Source source, const scalar::Profile *target,
             bool identity = false)
      : source(source), target(target), identity(identity) {
    scalar::Matrix inverse{};
    if (target && target->curves.size() == 1)
      inverse[0][1] = 1;
    else
      inverse = scalar::Invert(target ? target->matrix
                                      : connection::kSpaces[0].ToPcs());
    matrix = connection::Multiply(
        inverse, source.device ? source.device->matrix : source.space.ToPcs());
  }
  Color Convert(Color input) const {
    if (identity)
      return input;
    const size_t inputs = source.device ? source.device->curves.size() : 3;
    for (size_t c = 0; c < inputs; ++c) {
      if (source.device)
        input[c] = {source.device->curves[c].Forward(input[c].low),
                    source.device->curves[c].Forward(input[c].high)};
      else if (!source.linear)
        input[c] = {Linear(input[c].low), Linear(input[c].high)};
    }
    Color output{};
    for (size_t c = 0; c < (target ? target->curves.size() : 3); ++c) {
      double magnitude = 0, coefficients = 0;
      for (size_t k = 0; k < inputs; ++k) {
        output[c] = output[c] + input[k] * Range{matrix[c][k], matrix[c][k]};
        magnitude += std::abs(matrix[c][k]) *
                     std::max(std::abs(input[k].low), std::abs(input[k].high));
        coefficients += std::abs(matrix[c][k]);
      }
      // The existing matrix/TRC contract's F32 dot-product uncertainty, before
      // an arbitrarily steep target inverse. Never replace it with a fixed
      // output-code bound.
      const double error = 4e-7 * (1 + magnitude + coefficients);
      output[c].low -= error;
      output[c].high += error;
      if (target)
        output[c] = {target->curves[c].Inverse(output[c].low),
                     target->curves[c].Inverse(output[c].high)};
      output[c].low -= 2e-7;
      output[c].high += 2e-7;
    }
    return output;
  }
};

inline std::vector<float> Native(const Source &source,
                                 cmsHPROFILE source_profile,
                                 cmsHPROFILE target_profile,
                                 std::vector<float> values, bool identity,
                                 size_t &validated) {
  if (identity)
    return values;
  // ICC device curves have the unit-domain contract. Little CMS's float API may
  // extend analytic curves, so apply that contract at its device boundary. A
  // same- profile bypass and enumerated linear samples deliberately retain
  // their range.
  if (source.device)
    for (auto &value : values)
      value = std::clamp(value, 0.0f, 1.0f);
  std::vector<float> output;
  scalar::References bounds;
  if (source.device && target_profile) {
    bounds = scalar::Convert(*source.device, scalar::Profile(target_profile),
                             values);
    const unsigned count =
        cmsGetColorSpace(target_profile) == cmsSigGrayData ? 1 : 3;
    auto transform = cmsCreateTransform(
        source_profile,
        source.device->curves.size() == 1 ? TYPE_GRAY_FLT : TYPE_RGB_FLT,
        target_profile, count == 1 ? TYPE_GRAY_FLT : TYPE_RGB_FLT,
        INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
    if (!transform)
      throw std::runtime_error("native profile connection");
    const size_t pixels = values.size() / source.device->curves.size();
    output.resize(pixels * count);
    cmsDoTransform(transform, values.data(), output.data(), pixels);
    cmsDeleteTransform(transform);
    for (auto &value : output)
      value = std::clamp(value, 0.0f, 1.0f);
  } else if (source.device) {
    bounds = connection::Reference(*source.device, connection::kSpaces[0],
                                   values, true);
    output = connection::Native(source_profile, connection::kSpaces[0], values,
                                true);
  } else if (target_profile) {
    auto linear = values;
    if (!source.linear)
      for (auto &value : linear)
        value = float(Linear(value));
    bounds = connection::Reference(scalar::Profile(target_profile),
                                   source.space, linear, false);
    output = connection::Native(target_profile, source.space, linear, false);
  } else {
    // No ICC method is involved in an enumerated-to-linear reference.
    const auto matrix = connection::Multiply(
        scalar::Invert(connection::kSpaces[0].ToPcs()), source.space.ToPcs());
    for (size_t pixel = 0; pixel < values.size() / 3; ++pixel) {
      connection::Vector linear{};
      for (size_t c = 0; c < 3; ++c)
        linear[c] = source.linear ? values[pixel * 3 + c]
                                  : Linear(values[pixel * 3 + c]);
      const auto result = connection::Apply(matrix, linear);
      for (auto value : result)
        output.push_back(float(value));
    }
    return output;
  }
  for (size_t i = 0; i < output.size(); ++i) {
    if (bounds.native_semantics[i] != 0 || !std::isfinite(output[i]) ||
        output[i] < bounds.native_lower[i] ||
        output[i] > bounds.native_upper[i]) {
      std::fprintf(stderr,
                   "CMS component %zu: %g outside [%g,%g], semantics %u, input "
                   "range [%g,%g]\n",
                   i, output[i], bounds.native_lower[i], bounds.native_upper[i],
                   unsigned(bounds.native_semantics[i]),
                   *std::min_element(values.begin(), values.end()),
                   *std::max_element(values.begin(), values.end()));
      throw std::runtime_error(
          "native CMS outside independently derived method interval");
    }
    ++validated;
  }
  return output;
}
} // namespace reference
