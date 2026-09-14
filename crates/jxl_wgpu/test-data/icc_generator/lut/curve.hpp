#pragma once
#include "types.hpp"

namespace lut {
struct Curve {
  Bytes bytes;
  std::vector<uint16_t> table;
  std::function<double(double)> evaluate;
  std::function<double(double)> native_evaluate;
  std::vector<double> breaks;

  Value Apply(Value input, bool native) const {
    const auto primary = [&](double x) {
      return std::clamp(evaluate(std::clamp(x, 0.0, 1.0)), 0.0, 1.0);
    };
    const auto eval = [&](double x) {
      return native ? native_evaluate(x) : primary(x);
    };
    const double value = eval(input.x);
    const double quantization = native && !table.empty() ? quantum : 0;
    const double uncertainty = input.radius + quantization;
    const double lo = input.x - uncertainty, hi = input.x + uncertainty;
    double radius =
        std::max(std::abs(eval(lo) - value), std::abs(eval(hi) - value));
    for (double boundary : breaks)
      if (boundary >= lo && boundary <= hi) {
        radius = std::max(radius, std::abs(eval(boundary) - value));
        radius = std::max(
            radius,
            std::abs(eval(std::nextafter(
                         boundary, -std::numeric_limits<double>::infinity())) -
                     value));
      }
    const auto semantics =
        input.semantics |
        (native && value != primary(input.x) ? native_unbounded : 0);
    return {value, radius + 8 * epsilon * (1 + std::abs(value)) + quantization,
            semantics};
  }
};

inline Curve Table(unsigned count, unsigned precision,
                   const std::function<double(double)> &function) {
  const unsigned maximum = precision == 1 ? 255 : 65535;
  std::vector<uint16_t> values;
  for (unsigned n = 0; n < count; ++n)
    values.push_back(static_cast<uint16_t>(
        std::lround(std::clamp(function(static_cast<double>(n) / (count - 1)),
                               0.0, 1.0) *
                    maximum) *
        (precision == 1 ? 257 : 1)));
  auto bytes = Header("curv");
  U32(bytes, count);
  for (auto value : values)
    U16(bytes, value);
  Pad(bytes);
  std::vector<double> breaks;
  for (unsigned n = 1; n + 1 < count; ++n)
    breaks.push_back(static_cast<double>(n) / (count - 1));
  auto evaluate = [=](double x) {
    const double position = std::clamp(x, 0.0, 1.0) * (count - 1);
    const auto left = std::min(static_cast<unsigned>(position), count - 2);
    return (values[left] +
            (position - left) *
                (static_cast<double>(values[left + 1]) - values[left])) /
           65535;
  };
  return {bytes, values, evaluate, evaluate, breaks};
}

inline Curve Shape(unsigned index) {
  index %= 8;
  if (index == 0)
    return {Header("curv"),
            {},
            [](double x) { return x; },
            [](double x) { return x; },
            {}};
  if (index == 1) {
    auto bytes = Header("curv");
    U32(bytes, 1);
    U16(bytes, 320);
    Pad(bytes);
    auto evaluate = [](double x) { return std::pow(std::max(x, 0.0), 1.25); };
    return {bytes, {}, evaluate, evaluate, {0}};
  }
  if (index == 7)
    return Table(7, 2,
                 [](double x) { return .03125 + .90625 * std::pow(x, 1.5); });
  const unsigned function = index - 2;
  const std::array<unsigned, 5> count{1, 3, 4, 5, 7};
  const std::array<double, 7> p{
      1.75, .75, function == 1 ? -.125 : .125, .03125, .25, .03125, .015625};
  auto bytes = Header("para");
  U32(bytes, function << 16);
  for (unsigned n = 0; n < count[function]; ++n)
    S15(bytes, p[n]);
  const double boundary = function >= 3  ? p[4]
                          : function > 0 ? -p[2] / p[1]
                                         : -1;
  auto evaluate = [=](double x) {
    double value;
    if (function == 0)
      value = std::pow(x, p[0]);
    else if (function <= 2)
      value = (x < -p[2] / p[1] ? 0 : std::pow(p[1] * x + p[2], p[0])) +
              (function == 2 ? p[3] : 0);
    else
      value = x < p[4] ? p[3] * x + (function == 4 ? p[6] : 0)
                       : std::pow(p[1] * x + p[2], p[0]) +
                             (function == 4 ? p[5] : 0);
    return value;
  };
  // Native analytical curves do not clamp their output. Its ICC function 2
  // additionally chooses the constant branch below max(-b/a, 0), including
  // negative matrix outputs which ICC requires clipping before this curve.
  auto native = [=](double x) {
    if (function == 0)
      return std::pow(std::max(x, 0.0), p[0]);
    const double base = p[1] * x + p[2];
    if (function == 1)
      return x < -p[2] / p[1] || base <= 0 ? 0 : std::pow(base, p[0]);
    if (function == 2)
      return x < std::max(-p[2] / p[1], 0.0) ? p[3]
             : base > 0                      ? std::pow(base, p[0]) + p[3]
                                             : 0;
    return x < p[4] ? p[3] * x + (function == 4 ? p[6] : 0)
                    : std::pow(std::max(base, 0.0), p[0]) +
                          (function == 4 ? p[5] : 0);
  };
  return {bytes, {}, evaluate, native, {boundary, 0}};
}

inline Stage Curves(std::vector<Curve> curves) {
  Bytes bytes;
  for (auto &curve : curves) {
    if (curve.bytes.size() == 8)
      U32(curve.bytes, 0);
    bytes.insert(bytes.end(), curve.bytes.begin(), curve.bytes.end());
    Pad(bytes);
  }
  return {bytes, [=](const Values &input, bool native) {
            Check(input.size() == curves.size(), "curve channel count");
            Values output;
            for (size_t c = 0; c < curves.size(); ++c)
              output.push_back(curves[c].Apply(input[c], native));
            return output;
          }};
}
} // namespace lut
