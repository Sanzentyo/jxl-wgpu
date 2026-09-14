// Independent CIE/Bradford connection to unbounded linear RGB. Native ICC evaluation uses
// Little CMS's XYZ double interface; no quantized synthetic RGB profile is substituted.
#pragma once

#include "scalar.hpp"
#include <memory>
#include <string>

namespace connection {
using scalar::Matrix;
using Vector = std::array<double, 3>;

inline Matrix Multiply(const Matrix& a, const Matrix& b) {
  Matrix result{};
  for (size_t r = 0; r < 3; ++r) for (size_t c = 0; c < 3; ++c)
    for (size_t k = 0; k < 3; ++k) result[r][c] += a[r][k] * b[k][c];
  return result;
}

inline Vector Apply(const Matrix& m, const Vector& v) {
  Vector result{};
  for (size_t r = 0; r < 3; ++r) for (size_t c = 0; c < 3; ++c) result[r] += m[r][c] * v[c];
  return result;
}

struct LinearSpace {
  std::string name;
  std::array<double, 2> white;
  std::array<std::array<double, 2>, 3> primaries;

  Matrix ToPcs() const {
    // Normalize each primary to Y=1, independently of production's homogeneous columns.
    Matrix xyz{};
    for (size_t c = 0; c < 3; ++c) {
      const auto xy = primaries[c];
      xyz[0][c] = xy[0] / xy[1]; xyz[1][c] = 1; xyz[2][c] = (1 - xy[0] - xy[1]) / xy[1];
    }
    const Vector source_white{white[0] / white[1], 1, (1 - white[0] - white[1]) / white[1]};
    const auto scale = Apply(scalar::Invert(xyz), source_white);
    for (size_t r = 0; r < 3; ++r) for (size_t c = 0; c < 3; ++c) xyz[r][c] *= scale[c];
    const Matrix bradford{{{0.8951, 0.2664, -0.1614}, {-0.7502, 1.7135, 0.0367}, {0.0389, -0.0685, 1.0296}}};
    const auto source = Apply(bradford, source_white);
    const auto target = Apply(bradford, {0xf6d6 / 65536.0, 1, 0xd32d / 65536.0});
    Matrix scaled = bradford;
    for (size_t r = 0; r < 3; ++r) for (size_t c = 0; c < 3; ++c) scaled[r][c] *= target[r] / source[r];
    return Multiply(Multiply(scalar::Invert(bradford), scaled), xyz);
  }
};

inline const std::array<LinearSpace, 5> kSpaces{{
  {"bt709", {0.3127, 0.3290}, {{{0.64, 0.33}, {0.30, 0.60}, {0.15, 0.06}}}},
  {"bt2020", {0.3127, 0.3290}, {{{0.708, 0.292}, {0.170, 0.797}, {0.131, 0.046}}}},
  {"display_p3", {0.3127, 0.3290}, {{{0.680, 0.320}, {0.265, 0.690}, {0.150, 0.060}}}},
  {"equal_white", {1.0 / 3.0, 1.0 / 3.0}, {{{0.64, 0.33}, {0.30, 0.60}, {0.15, 0.06}}}},
  {"native_rgb", {0.3127, 0.3290}, {{{0.639998686, 0.330010138}, {0.300003784, 0.600003357}, {0.150002046, 0.059997204}}}},
}};

inline scalar::References Reference(const scalar::Profile& profile, const LinearSpace& space,
                                    const std::vector<float>& input, bool to_linear) {
  Matrix inverse{};
  if (profile.curves.size() == 1) inverse[0][1] = 1;
  else inverse = scalar::Invert(profile.matrix);
  const Matrix matrix = to_linear ? Multiply(scalar::Invert(space.ToPcs()), profile.matrix)
                                  : Multiply(inverse, space.ToPcs());
  const size_t inputs = to_linear ? profile.curves.size() : 3;
  const size_t outputs = to_linear ? 3 : profile.curves.size();
  scalar::References result;
  for (size_t pixel = 0; pixel < input.size() / inputs; ++pixel) {
    Vector linear{};
    bool source_boundary = false;
    for (size_t c = 0; c < inputs; ++c) {
      const double x = input[pixel * inputs + c];
      linear[c] = to_linear ? profile.curves[c].Forward(x) : x;
      if (to_linear) {
        const auto& curve = profile.curves[c];
        source_boundary |= curve.type == 3 && x >= -curve.p[2] / curve.p[1] && curve.p[1] * x + curve.p[2] == 0;
      }
    }
    for (size_t r = 0; r < outputs; ++r) {
      double value = 0, magnitude = 0, coefficients = 0;
      for (size_t c = 0; c < inputs; ++c) {
        value += matrix[r][c] * linear[c];
        magnitude += std::abs(matrix[r][c] * linear[c]);
        coefficients += std::abs(matrix[r][c]);
      }
      const double uncertainty = 4e-7 * (1 + magnitude + coefficients);
      const auto output = [&](double v) { return to_linear ? v : profile.curves[r].Inverse(v); };
      const double native_uncertainty = uncertainty + 3.0 / 65535.0 * coefficients;
      const double reverse_error = !to_linear && profile.curves[r].type == 0 ? 2.0 / 4095.0 : 2e-7;
      result.exact.push_back(static_cast<float>(output(value)));
      result.lower.push_back(static_cast<float>(output(value - uncertainty) - 2e-7));
      result.upper.push_back(static_cast<float>(output(value + uncertainty) + 2e-7));
      result.native_lower.push_back(static_cast<float>(output(value - native_uncertainty) - reverse_error));
      result.native_upper.push_back(static_cast<float>(output(value + native_uncertainty) + reverse_error));
      result.native_semantics.push_back(static_cast<uint8_t>((source_boundary ? 1 : 0)
          | ((!to_linear && profile.curves[r].type == 2 && value < 0) ? 2 : 0)));
    }
  }
  return result;
}

inline std::vector<float> Native(cmsHPROFILE profile, const LinearSpace& space,
                                  const std::vector<float>& input, bool to_linear,
                                  unsigned intent = INTENT_RELATIVE_COLORIMETRIC) {
  const size_t channels = cmsGetColorSpace(profile) == cmsSigGrayData ? 1 : 3;
  const size_t pixels = input.size() / (to_linear ? channels : 3);
  const cmsUInt32Number format = channels == 1 ? TYPE_GRAY_FLT : TYPE_RGB_FLT;
  const std::unique_ptr<void, decltype(&cmsCloseProfile)> xyz(cmsCreateXYZProfile(), cmsCloseProfile);
  if (!xyz) throw std::runtime_error("create native XYZ endpoint");
  const std::unique_ptr<void, decltype(&cmsDeleteTransform)> transform(
      cmsCreateTransform(to_linear ? profile : xyz.get(), to_linear ? format : TYPE_XYZ_DBL,
                         to_linear ? xyz.get() : profile, to_linear ? TYPE_XYZ_DBL : format,
                         intent, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE), cmsDeleteTransform);
  if (!transform) throw std::runtime_error("create native linear connection");
  const Matrix matrix = to_linear ? scalar::Invert(space.ToPcs()) : space.ToPcs();
  std::vector<double> intermediate(pixels * 3);
  std::vector<float> output(pixels * (to_linear ? 3 : channels));
  if (to_linear) {
    cmsDoTransform(transform.get(), input.data(), intermediate.data(), static_cast<cmsUInt32Number>(pixels));
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
      const auto rgb = Apply(matrix, {intermediate[3 * pixel], intermediate[3 * pixel + 1], intermediate[3 * pixel + 2]});
      for (size_t c = 0; c < 3; ++c) output[pixel * 3 + c] = static_cast<float>(rgb[c]);
    }
  } else {
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
      const auto pcs = Apply(matrix, {input[3 * pixel], input[3 * pixel + 1], input[3 * pixel + 2]});
      for (size_t c = 0; c < 3; ++c) intermediate[pixel * 3 + c] = pcs[c];
    }
    cmsDoTransform(transform.get(), intermediate.data(), output.data(), static_cast<cmsUInt32Number>(pixels));
    for (auto& value : output) value = std::clamp(value, 0.0f, 1.0f);
  }
  for (float value : output) if (!std::isfinite(value)) throw std::runtime_error("nonfinite native linear output");
  return output;
}
}  // namespace connection
