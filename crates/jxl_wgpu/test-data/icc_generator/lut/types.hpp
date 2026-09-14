#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <lcms2.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lut {
using Bytes = std::vector<uint8_t>;
struct Value {
  double x, radius;
  uint32_t semantics = 0;
};
using Values = std::vector<Value>;
struct Stage {
  Bytes bytes;
  std::function<Values(const Values &, bool)> apply;
};
constexpr double epsilon = 4e-7;
constexpr double quantum = 0.5 / 65535;
// Little CMS 2.19 extends stored LUT matrices/analytical curves outside the
// normative unit domain. Both models retain independently propagated intervals.
constexpr uint32_t native_unbounded = 1u << 5;
inline void Check(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
inline uint32_t Bits(float value) {
  uint32_t word;
  std::memcpy(&word, &value, 4);
  return word;
}
inline void U16(Bytes &out, uint16_t value) {
  out.push_back(static_cast<uint8_t>(value >> 8));
  out.push_back(static_cast<uint8_t>(value));
}
inline void U32(Bytes &out, uint32_t value) {
  for (int n = 3; n >= 0; --n)
    out.push_back(static_cast<uint8_t>(value >> (n * 8)));
}
inline void Put(Bytes &out, size_t at, uint32_t value) {
  for (int n = 3; n >= 0; --n)
    out[at++] = static_cast<uint8_t>(value >> (n * 8));
}
inline void LE(Bytes &out, float value) {
  const auto word = Bits(value);
  for (unsigned n = 0; n < 4; ++n)
    out.push_back(static_cast<uint8_t>(word >> (n * 8)));
}
inline void Pad(Bytes &out) {
  while (out.size() % 4)
    out.push_back(0);
}
inline Bytes Header(const char *signature) {
  Bytes out(signature, signature + 4);
  U32(out, 0);
  return out;
}
inline double Fixed(double value) { return std::round(value * 65536) / 65536; }
inline void S15(Bytes &out, double value) {
  U32(out,
      static_cast<uint32_t>(static_cast<int32_t>(std::llround(value * 65536))));
}
inline void Save(const std::filesystem::path &path, const Bytes &bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  Check(bool(file), "write LUT corpus");
}
inline void Record(Bytes &out, float native, Value primary,
                   Value native_bound) {
  auto valid = [](Value value) {
    return std::isfinite(value.x) && std::isfinite(value.radius) &&
           value.radius >= 0;
  };
  Check(std::isfinite(native) && valid(primary) && valid(native_bound),
        "finite LUT reference and nonnegative radii");
  auto lower = [](Value v) {
    return std::nextafter(static_cast<float>(v.x - v.radius),
                          -std::numeric_limits<float>::infinity());
  };
  auto upper = [](Value v) {
    return std::nextafter(static_cast<float>(v.x + v.radius),
                          std::numeric_limits<float>::infinity());
  };
  if (!(native >= lower(native_bound) && native <= upper(native_bound))) {
    std::cerr << "native=" << native << " exact=" << primary.x
              << " primary radius=" << primary.radius
              << " native center=" << native_bound.x
              << " native radius=" << native_bound.radius << '\n';
    throw std::runtime_error("native outside independent LUT interval");
  }
  for (float value :
       {native, static_cast<float>(primary.x), lower(primary), upper(primary),
        lower(native_bound), upper(native_bound)}) {
    Check(std::isfinite(value), "finite serialized LUT reference");
    LE(out, value);
  }
  for (unsigned n = 0; n < 4; ++n)
    out.push_back(static_cast<uint8_t>(native_bound.semantics >> (n * 8)));
}
} // namespace lut
