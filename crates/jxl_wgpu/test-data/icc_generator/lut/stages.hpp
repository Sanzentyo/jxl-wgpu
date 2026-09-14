#pragma once
#include "types.hpp"

namespace lut {
inline Stage Matrix(std::array<double, 9> coefficients,
                    std::array<double, 3> offset, bool clamp,
                    bool stored = false) {
  Bytes bytes;
  if (stored) {
    for (double &value : coefficients) {
      value = Fixed(value);
      S15(bytes, value);
    }
    for (double &value : offset) {
      value = Fixed(value);
      S15(bytes, value);
    }
  }
  return {bytes, [=](const Values &input, bool native) {
            Check(input.size() == 3, "matrix channels");
            Values result;
            for (unsigned r = 0; r < 3; ++r) {
              double value = offset[r], radius = 0,
                     magnitude = 1 + std::abs(value);
              uint32_t semantics = 0;
              for (unsigned c = 0; c < 3; ++c) {
                const double a = coefficients[r * 3 + c];
                value += a * input[c].x;
                radius += std::abs(a) * input[c].radius;
                magnitude += std::abs(a * input[c].x);
                if (a != 0)
                  semantics |= input[c].semantics;
              }
              // Little CMS also leaves the virtual PCS normalization matrices
              // unclipped.
              const bool clip = clamp && !native;
              if (native && clamp && (value < 0 || value > 1))
                semantics |= native_unbounded;
              result.push_back({clip ? std::clamp(value, 0.0, 1.0) : value,
                                radius + epsilon * magnitude, semantics});
            }
            return result;
          }};
}
inline Stage Diagonal(std::array<double, 3> scale, std::array<double, 3> offset,
                      bool clamp = false) {
  return Matrix({scale[0], 0, 0, 0, scale[1], 0, 0, 0, scale[2]}, offset,
                clamp);
}
inline Stage Clut(const std::vector<unsigned> &grid, unsigned outputs,
                  unsigned precision, bool multilinear) {
  const unsigned n = static_cast<unsigned>(grid.size());
  std::vector<size_t> stride(n);
  size_t count = outputs;
  for (unsigned i = n; i-- > 0;) {
    stride[i] = count;
    count *= grid[i];
  }
  const unsigned maximum = precision == 1 ? 255 : 65535;
  Bytes bytes;
  std::vector<double> values(count), gradient(n * outputs);
  for (unsigned points : grid)
    bytes.push_back(static_cast<uint8_t>(points));
  bytes.resize(16);
  bytes.push_back(static_cast<uint8_t>(precision));
  bytes.resize(20);
  for (size_t point = 0; point < count / outputs; ++point) {
    std::vector<double> x(n);
    for (unsigned axis = 0; axis < n; ++axis)
      x[axis] =
          static_cast<double>((point * outputs / stride[axis]) % grid[axis]) /
          (grid[axis] - 1);
    for (unsigned c = 0; c < outputs; ++c) {
      double value = .0625 + .03125 * c + .27 * x[0] * x[n - 1];
      for (double coordinate : x)
        value += (.18 + .03 * c) * coordinate * coordinate / n;
      const auto encoded = static_cast<unsigned>(
          std::lround(std::clamp(value, 0.0, 1.0) * maximum));
      values[point * outputs + c] = static_cast<double>(encoded) / maximum;
      if (precision == 1)
        bytes.push_back(static_cast<uint8_t>(encoded));
      else
        U16(bytes, static_cast<uint16_t>(encoded));
    }
  }
  for (size_t i = 0; i < count; ++i)
    for (unsigned axis = 0; axis < n; ++axis)
      if ((i / stride[axis]) % grid[axis] + 1 < grid[axis]) {
        gradient[axis * outputs + i % outputs] = std::max(
            gradient[axis * outputs + i % outputs],
            std::abs(values[i + stride[axis]] - values[i]) * (grid[axis] - 1));
      }
  return {bytes, [=](const Values &input, bool native) {
            Check(input.size() == n, "CLUT channels");
            std::vector<double> t(n);
            size_t base = 0;
            for (unsigned axis = 0; axis < n; ++axis) {
              const double position =
                  std::clamp(input[axis].x, 0.0, 1.0) * (grid[axis] - 1);
              const auto left =
                  std::min(static_cast<unsigned>(position), grid[axis] - 2);
              t[axis] = position - left;
              base += left * stride[axis];
            }
            std::function<double(unsigned, size_t, unsigned)> interpolate =
                [&](unsigned axis, size_t at, unsigned c) -> double {
              if (axis == n)
                return values[at + c];
              if (!multilinear && n - axis == 3) {
                std::array<unsigned, 3> order{axis, axis + 1, axis + 2};
                std::stable_sort(
                    order.begin(), order.end(),
                    [&](unsigned a, unsigned b) { return t[a] > t[b]; });
                double value = (1 - t[order[0]]) * values[at + c];
                for (unsigned j = 0; j < 3; ++j) {
                  at += stride[order[j]];
                  value += (t[order[j]] - (j < 2 ? t[order[j + 1]] : 0)) *
                           values[at + c];
                }
                return value;
              }
              return (1 - t[axis]) * interpolate(axis + 1, at, c) +
                     t[axis] * interpolate(axis + 1, at + stride[axis], c);
            };
            Values output;
            uint32_t semantics = 0;
            for (const auto &value : input)
              semantics |= value.semantics;
            for (unsigned c = 0; c < outputs; ++c) {
              double radius = 8 * epsilon * n * 2 + (native ? n * quantum : 0);
              for (unsigned axis = 0; axis < n; ++axis)
                radius += gradient[axis * outputs + c] *
                          (input[axis].radius + (native ? quantum : 0));
              output.push_back({interpolate(0, base, c), radius, semantics});
            }
            return output;
          }};
}

inline std::array<double, 3> LabValue(std::array<double, 3> input,
                                      bool to_xyz) {
  if (to_xyz) {
    auto f = [](double x) {
      return x > 6.0 / 29 ? x * x * x : (116 * x - 16) * 27 / 24389;
    };
    const double y = (input[0] + 16) / 116;
    return {.9642 * f(y + input[1] / 500), f(y), .8249 * f(y - input[2] / 200)};
  }
  auto f = [](double x) {
    return x > 216.0 / 24389 ? std::cbrt(x) : (24389.0 / 27 * x + 16) / 116;
  };
  const double y = f(input[1]);
  return {116 * y - 16, 500 * (f(input[0] / .9642) - y),
          200 * (y - f(input[2] / .8249))};
}
inline Stage Lab(bool to_xyz) {
  return {{}, [=](const Values &input, bool) {
            Check(input.size() == 3, "Lab channels");
            const auto center =
                LabValue({input[0].x, input[1].x, input[2].x}, to_xyz);
            uint32_t semantics = 0;
            for (const auto &value : input)
              semantics |= value.semantics;
            Values output;
            for (double value : center)
              output.push_back({value, 0, semantics});
            for (unsigned mask = 0; mask < 8; ++mask) {
              std::array<double, 3> corner;
              for (unsigned c = 0; c < 3; ++c)
                corner[c] =
                    input[c].x + ((mask & (1 << c)) ? 1 : -1) * input[c].radius;
              const auto value = LabValue(corner, to_xyz);
              for (unsigned c = 0; c < 3; ++c)
                output[c].radius =
                    std::max(output[c].radius, std::abs(value[c] - center[c]));
            }
            std::array<double, 3> magnitude;
            if (to_xyz) {
              for (unsigned c = 0; c < 3; ++c)
                magnitude[c] = 1 + std::abs(center[c]);
            } else {
              auto f = [](double x) {
                return x > 216.0 / 24389 ? std::cbrt(x)
                                         : (24389.0 / 27 * x + 16) / 116;
              };
              const double fx = std::abs(f(input[0].x / .9642)),
                           fy = std::abs(f(input[1].x)),
                           fz = std::abs(f(input[2].x / .8249));
              magnitude = {1 + 116 * fy + 16, 1 + 500 * (fx + fy),
                           1 + 200 * (fy + fz)};
            }
            for (unsigned c = 0; c < 3; ++c)
              output[c].radius += 8 * epsilon * magnitude[c];
            return output;
          }};
}

inline std::vector<Stage> Pcs(bool lab, bool legacy, bool reverse) {
  if (!lab) {
    const double scale = reverse ? 32768.0 / 65535 : 65535.0 / 32768;
    return {Diagonal({scale, scale, scale}, {0, 0, 0}, reverse)};
  }
  if (!legacy)
    return reverse ? std::vector<Stage>{Lab(false),
                                        Diagonal({.01, 1.0 / 255, 1.0 / 255},
                                                 {0, 128.0 / 255, 128.0 / 255},
                                                 true)}
                   : std::vector<Stage>{
                         Diagonal({100, 255, 255}, {0, -128, -128}), Lab(true)};
  return reverse
             ? std::vector<Stage>{Lab(false),
                                  Diagonal(
                                      {.01, 256.0 / 65535, 256.0 / 65535},
                                      {0, 32768.0 / 65535, 32768.0 / 65535},
                                      true),
                                  Diagonal({65280.0 / 65535, 1, 1}, {0, 0, 0})}
             : std::vector<Stage>{
                   Diagonal({65535.0 / 65280, 1, 1}, {0, 0, 0}, true),
                   Diagonal({100, 65535.0 / 256, 65535.0 / 256},
                            {0, -128, -128}),
                   Lab(true)};
}
} // namespace lut
