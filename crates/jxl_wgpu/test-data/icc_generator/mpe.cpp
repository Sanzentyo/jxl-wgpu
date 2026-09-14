// Independent ICC.1:2022 MPE equations and Little CMS 2.19 execution.
// This offline generator never reads production Rust/WGSL or GPU-produced pixels.
#include <lcms2.h>
#include <icc/intents.hpp>
#include <cctype>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Bytes = std::vector<uint8_t>;
constexpr double epsilon = 4e-7;
struct Value { double x, radius; };
using Values = std::vector<Value>;
struct Stage { Bytes bytes; std::function<Values(const Values&)> apply; };
struct Case { std::string name; bool lab; std::vector<Stage> stages; };
void Check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
uint32_t Bits(float x) { uint32_t bits; std::memcpy(&bits, &x, 4); return bits; }
void U32(Bytes& b, uint32_t v) { for (int i = 3; i >= 0; --i) b.push_back(static_cast<uint8_t>(v >> (i * 8))); }
void Put(Bytes& b, size_t at, uint32_t v) { for (int i = 3; i >= 0; --i) b[at++] = static_cast<uint8_t>(v >> (i * 8)); }
void Float(Bytes& b, double v) { U32(b, Bits(static_cast<float>(v))); }
Bytes Header(const char* sig, unsigned p = 0, unsigned q = 0) {
  Bytes b(sig, sig + 4); U32(b, 0); if (p) U32(b, (p << 16) | q); return b;
}
Bytes Positioned(const char* sig, unsigned p, unsigned q, const std::vector<Bytes>& elements) {
  auto b = Header(sig, p, q);
  if (std::string(sig, 4) == "mpet") U32(b, static_cast<uint32_t>(elements.size()));
  size_t table = b.size(); b.resize(table + elements.size() * 8);
  // Reverse physical storage so every real profile tests position-table execution order.
  for (size_t r = elements.size(); r-- > 0;) {
    Put(b, table + r * 8, static_cast<uint32_t>(b.size())); Put(b, table + r * 8 + 4, static_cast<uint32_t>(elements[r].size()));
    b.insert(b.end(), elements[r].begin(), elements[r].end()); while (b.size() % 4) b.push_back(0);
  }
  return b;
}
Stage Matrix(unsigned p, unsigned q, std::vector<double> a, std::vector<double> offsets) {
  Check(a.size() == p * q && offsets.size() == q, "matrix shape");
  for (auto& v : a) v = static_cast<float>(v);
  for (auto& v : offsets) v = static_cast<float>(v);
  auto b = Header("matf", p, q); for (double v : a) Float(b, v); for (double v : offsets) Float(b, v);
  return {b, [=](const Values& input) {
    Check(input.size() == p, "matrix input"); Values result;
    for (unsigned r = 0; r < q; ++r) {
      double x = offsets[r], radius = 0, magnitude = 1 + std::abs(x);
      for (unsigned c = 0; c < p; ++c) { x += a[r * p + c] * input[c].x; radius += std::abs(a[r * p + c]) * input[c].radius; magnitude += std::abs(a[r * p + c] * input[c].x); }
      result.push_back({x, radius + epsilon * std::max(1.0, p / 3.0) * magnitude});
    }
    return result;
  }};
}
Stage Identity(unsigned n) {
  std::vector<double> a(n * n); for (unsigned c = 0; c < n; ++c) a[c * n + c] = 1;
  return Matrix(n, n, a, std::vector<double>(n));
}
struct Piece { double upper; int kind; std::vector<double> p; };
using Curve = std::vector<Piece>;
double Formula(const Piece& piece, double x, double lower, double initial) {
  const auto& p = piece.p;
  if (piece.kind == 0) return std::pow(p[1] * x + p[2], p[0]) + p[3];
  if (piece.kind == 1) return p[1] * std::log10(p[2] * std::pow(x, p[0]) + p[3]) + p[4];
  if (piece.kind == 2) return p[0] * std::pow(p[1], p[2] * x + p[3]) + p[4];
  const double position = std::clamp((x - lower) / (piece.upper - lower), 0.0, 1.0) * p.size();
  const size_t left = std::min(static_cast<size_t>(position), p.size() - 1);
  const double a = left == 0 ? initial : p[left - 1], b = p[left];
  return a + (position - left) * (b - a);
}
double CurveValue(const Curve& curve, double x) {
  double lower = -std::numeric_limits<double>::infinity(), initial = 0;
  for (const auto& piece : curve) {
    if (x <= piece.upper) return Formula(piece, x, lower, initial);
    initial = Formula(piece, piece.upper, lower, initial); lower = piece.upper;
  }
  throw std::runtime_error("curve domain");
}
Stage Curves(std::vector<Curve> curves) {
  std::vector<Bytes> records;
  for (auto& curve : curves) {
    auto b = Header("curf"); U32(b, static_cast<uint32_t>(curve.size()) << 16);
    for (size_t i = 0; i + 1 < curve.size(); ++i) { curve[i].upper = static_cast<float>(curve[i].upper); Float(b, curve[i].upper); }
    for (auto& piece : curve) {
      auto segment = Header(piece.kind == 3 ? "samf" : "parf");
      U32(segment, piece.kind == 3 ? static_cast<uint32_t>(piece.p.size()) : static_cast<uint32_t>(piece.kind) << 16);
      for (auto& v : piece.p) { v = static_cast<float>(v); Float(segment, v); }
      b.insert(b.end(), segment.begin(), segment.end());
    }
    records.push_back(b);
  }
  return {Positioned("cvst", static_cast<unsigned>(curves.size()), static_cast<unsigned>(curves.size()), records), [=](const Values& input) {
    Values output;
    for (size_t c = 0; c < curves.size(); ++c) {
      const auto& curve = curves[c]; const auto v = input[c]; const double center = CurveValue(curve, v.x);
      double radius = std::max(std::abs(CurveValue(curve, v.x - v.radius) - center), std::abs(CurveValue(curve, v.x + v.radius) - center));
      for (const auto& piece : curve) if (piece.upper >= v.x - v.radius && piece.upper <= v.x + v.radius) {
        for (double x : {piece.upper, std::nextafter(piece.upper, std::numeric_limits<double>::infinity())}) radius = std::max(radius, std::abs(CurveValue(curve, x) - center));
      }
      output.push_back({center, radius + 8 * epsilon * (1 + std::abs(center))});
    }
    return output;
  }};
}
Stage Clut(std::vector<unsigned> grid) {
  const unsigned n = static_cast<unsigned>(grid.size());
  std::vector<size_t> stride(n); size_t count = 3;
  for (size_t i = n; i-- > 0;) { stride[i] = count; count *= grid[i]; }
  std::vector<double> table(count), gradient(n * 3); double magnitude = 0;
  for (size_t point = 0; point < count / 3; ++point) {
    std::vector<double> x(n); for (unsigned i = 0; i < n; ++i) x[i] = static_cast<double>((point * 3 / stride[i]) % grid[i]) / (grid[i] - 1);
    for (unsigned c = 0; c < 3; ++c) {
      double value = -0.125 * (c + 1);
      for (unsigned axis = 0; axis < n; ++axis) value += (0.3 + 0.1 * c) * x[axis] * x[axis] / n;
      if (n > 1) value += (1.3 + 0.1 * c) * x[0] * x[n - 1]; else value += 1.5 * x[0];
      table[point * 3 + c] = static_cast<float>(value); magnitude = std::max(magnitude, std::abs(table[point * 3 + c]));
    }
  }
  for (size_t i = 0; i < count; ++i) for (unsigned axis = 0; axis < n; ++axis) if ((i / stride[axis]) % grid[axis] + 1 < grid[axis]) {
    gradient[axis * 3 + i % 3] = std::max(gradient[axis * 3 + i % 3], std::abs(table[i + stride[axis]] - table[i]) * (grid[axis] - 1));
  }
  auto b = Header("clut", n, 3); for (unsigned g : grid) b.push_back(static_cast<uint8_t>(g)); b.resize(28); for (double v : table) Float(b, v);
  return {b, [=](const Values& input) {
    std::vector<double> t(n); size_t base = 0;
    for (unsigned i = 0; i < n; ++i) { double v = std::clamp(input[i].x, 0.0, 1.0) * (grid[i] - 1); auto left = std::min(static_cast<unsigned>(v), grid[i] - 2); base += left * stride[i]; t[i] = v - left; }
    std::function<double(unsigned, size_t, unsigned)> interpolate = [&](unsigned axis, size_t offset, unsigned channel) -> double {
      if (axis == n) return table[offset + channel];
      if (n - axis == 3) {
        std::array<unsigned, 3> order{axis, axis + 1, axis + 2}; std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) { return t[a] > t[b]; });
        double result = (1 - t[order[0]]) * table[offset + channel];
        for (unsigned i = 0; i < 3; ++i) { offset += stride[order[i]]; result += (t[order[i]] - (i < 2 ? t[order[i + 1]] : 0)) * table[offset + channel]; }
        return result;
      }
      return (1 - t[axis]) * interpolate(axis + 1, offset, channel) + t[axis] * interpolate(axis + 1, offset + stride[axis], channel);
    };
    Values output;
    for (unsigned c = 0; c < 3; ++c) {
      double radius = 8 * epsilon * n * (1 + magnitude);
      for (unsigned axis = 0; axis < n; ++axis) radius += gradient[axis * 3 + c] * input[axis].radius;
      output.push_back({interpolate(0, base, c), radius});
    }
    return output;
  }};
}
Values Lab(const Values& input, bool to_xyz) {
  auto convert = [=](std::array<double, 3> x) {
    if (to_xyz) {
      auto f = [](double t) { return t > 6.0 / 29 ? t * t * t : (116 * t - 16) * 27 / 24389; };
      double y = (x[0] + 16) / 116;
      return std::array<double, 3>{0.9642 * f(y + x[1] / 500), f(y), 0.8249 * f(y - x[2] / 200)};
    }
    auto f = [](double t) { return t > 216.0 / 24389 ? std::cbrt(t) : (24389.0 / 27 * t + 16) / 116; };
    double y = f(x[1]); return std::array<double, 3>{116 * y - 16, 500 * (f(x[0] / 0.9642) - y), 200 * (y - f(x[2] / 0.8249))};
  };
  auto center = convert({input[0].x, input[1].x, input[2].x}); Values output;
  for (double x : center) output.push_back({x, 0});
  for (unsigned mask = 0; mask < 8; ++mask) {
    std::array<double, 3> corner; for (unsigned c = 0; c < 3; ++c) corner[c] = input[c].x + ((mask & (1 << c)) ? 1 : -1) * input[c].radius;
    auto value = convert(corner); for (unsigned c = 0; c < 3; ++c) output[c].radius = std::max(output[c].radius, std::abs(value[c] - center[c]));
  }
  std::array<double, 3> magnitude;
  if (to_xyz) {
    for (unsigned c = 0; c < 3; ++c) magnitude[c] = 1 + std::abs(output[c].x);
  } else {
    auto f = [](double t) { return t > 216.0 / 24389 ? std::cbrt(t) : (24389.0 / 27 * t + 16) / 116; };
    const double fx = std::abs(f(input[0].x / .9642)), fy = std::abs(f(input[1].x)), fz = std::abs(f(input[2].x / .8249));
    // Subtracting similar f(X), f(Y), f(Z) loses relative accuracy in a*/b*.
    // Bound both operands before cancellation, then propagate through later matrices.
    magnitude = {1 + 116 * fy + 16, 1 + 500 * (fx + fy), 1 + 200 * (fy + fz)};
  }
  for (unsigned c = 0; c < 3; ++c) output[c].radius += 8 * epsilon * magnitude[c];

  return output;
}
Stage Intent(unsigned intent) {
  const double d = (static_cast<double>(intent) - 1) / 64;
  return Matrix(3, 3, {1,0,0, 0,1,0, 0,0,1}, {d, d * 2, d * 3});
}
Bytes Profile(const Case& c, bool identity = false) {
  std::vector<std::pair<std::string, Bytes>> tags;
  auto white = Header("XYZ "); U32(white, 0xf6d6); U32(white, 65536); U32(white, 0xd32d); tags.push_back({"wtpt", white});
  for (unsigned intent = 0; intent < 4; ++intent) {
    std::vector<Bytes> elements; for (const auto& stage : c.stages) elements.push_back(stage.bytes);
    if (!identity) elements.push_back(Intent(intent).bytes);
    auto mpe = Positioned("mpet", 3, 3, elements);
    tags.push_back({"D2B" + std::to_string(intent), mpe}); tags.push_back({"B2D" + std::to_string(intent), mpe});
  }
  Bytes b(132 + 12 * tags.size()); Put(b, 8, 0x04400000); std::copy_n("mntr", 4, b.begin() + 12); std::copy_n("RGB ", 4, b.begin() + 16); std::copy_n(c.lab ? "Lab " : "XYZ ", 4, b.begin() + 20);
  std::copy_n("acsp", 4, b.begin() + 36); Put(b, 64, 1); Put(b, 68, 0xf6d6); Put(b, 72, 65536); Put(b, 76, 0xd32d); Put(b, 128, static_cast<uint32_t>(tags.size()));
  for (size_t i = 0; i < tags.size(); ++i) { auto [tag, data] = tags[i]; std::copy_n(tag.begin(), 4, b.begin() + 132 + i * 12); Put(b, 136 + i * 12, static_cast<uint32_t>(b.size())); Put(b, 140 + i * 12, static_cast<uint32_t>(data.size())); b.insert(b.end(), data.begin(), data.end()); while (b.size() % 4) b.push_back(0); }
  Put(b, 0, static_cast<uint32_t>(b.size())); return b;
}
void Save(const fs::path& path, const Bytes& bytes) { std::ofstream f(path, std::ios::binary); f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())); Check(bool(f), "write corpus"); }
void LE(Bytes& b, uint32_t word) { for (unsigned i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>(word >> (i * 8))); }
void Record(Bytes& b, float native, Value value, double native_radius = -1) {
  Check(std::isfinite(native) && std::isfinite(value.x) && std::isfinite(value.radius), "nonfinite reference");
  const float lower = std::nextafter(static_cast<float>(value.x - value.radius), -std::numeric_limits<float>::infinity());
  const float upper = std::nextafter(static_cast<float>(value.x + value.radius), std::numeric_limits<float>::infinity());
  if (native_radius < 0) native_radius = value.radius;
  const float native_lower = std::nextafter(static_cast<float>(value.x - native_radius), -std::numeric_limits<float>::infinity());
  const float native_upper = std::nextafter(static_cast<float>(value.x + native_radius), std::numeric_limits<float>::infinity());
  if (!(native >= native_lower && native <= native_upper)) { std::cerr << "native=" << native << " exact=" << value.x << " radius=" << value.radius << '\n'; throw std::runtime_error("native outside independent interval"); }
  for (float v : {native, static_cast<float>(value.x), lower, upper, native_lower, native_upper}) LE(b, Bits(v)); LE(b, 0);
}
Bytes Read(const fs::path& path) {
  std::ifstream f(path, std::ios::binary); Check(bool(f), "read native source");
  return Bytes(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
std::vector<float> HexFloats(const fs::path& path) {
  Bytes bytes; int high = -1;
  for (unsigned char ch : Read(path)) {
    if (std::isspace(ch)) continue;
    int digit = ch >= '0' && ch <= '9' ? ch - '0' : ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 : -1;
    Check(digit >= 0, "hex source digit");
    if (high < 0) high = digit; else { bytes.push_back(static_cast<uint8_t>(high * 16 + digit)); high = -1; }
  }
  Check(high < 0 && bytes.size() % 4 == 0, "complete float source");
  std::vector<float> values;
  for (size_t i = 0; i < bytes.size(); i += 4) {
    uint32_t bits = 0; for (unsigned j = 0; j < 4; ++j) bits |= uint32_t{bytes[i + j]} << (j * 8);
    float value; std::memcpy(&value, &bits, 4); Check(std::isfinite(value), "finite native source"); values.push_back(value);
  }
  return values;
}
void Decoder(const std::vector<Case>& cases, const fs::path& base, const fs::path& out) {
  fs::create_directories(out);
  constexpr std::array<double, 3> d50{.9642,1,.8249}, reference_black{.00336,.0034731,.0028646};
  for (bool gray : {false, true}) for (bool modular : {false, true}) {
    const std::string color = gray ? "gray" : "rgb";
    const auto name = color + (modular ? "_modular_original" : "_vardct_original");
    const auto bytes = Read(base / (color + ".icc"));
    auto source = cmsOpenProfileFromMem(bytes.data(), static_cast<cmsUInt32Number>(bytes.size())); Check(source, "source ICC profile");
    const scalar::Profile metadata(source); const auto black = intents::Black(metadata), white = intents::MediaWhite(source);
    const unsigned channels = gray ? 1 : 3;
    const auto rgba = HexFloats(base / (name + ".native.f32.hex")); Check(rgba.size() == 153 * (channels + 1), "decoder dimensions");
    std::vector<float> input; for (unsigned pixel = 0; pixel < 153; ++pixel) for (unsigned c = 0; c < channels; ++c) input.push_back(rgba[pixel * (channels + 1) + c]);
    for (const auto& target : cases) {
      if (target.name != "segments" && target.name != "clut4" && target.name != "lab") continue;
      auto data = Profile(target); auto profile = cmsOpenProfileFromMem(data.data(), static_cast<cmsUInt32Number>(data.size())); Check(profile, "decoder target profile");
      for (unsigned intent = 0; intent < 4; ++intent) {
        auto transform = cmsCreateTransform(source, gray ? TYPE_GRAY_FLT : TYPE_RGB_FLT, profile, TYPE_RGB_FLT, intent, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE); Check(transform, "decoder native transform");
        std::vector<float> native(153 * 3); cmsDoTransform(transform, input.data(), native.data(), 153); cmsDeleteTransform(transform);
        Bytes records;
        for (unsigned pixel = 0; pixel < 153; ++pixel) {
          Values linear;
          for (unsigned c = 0; c < channels; ++c) {
            const double sample = input[pixel * channels + c], error = modular ? 0 : 2e-5;
            const auto& curve = metadata.curves[c]; const double value = curve.Forward(sample);
            const double radius = std::max(std::abs(curve.Forward(sample - error) - value), std::abs(curve.Forward(sample + error) - value));
            linear.push_back({value, radius + epsilon * (1 + std::abs(value))});
          }
          std::vector<double> matrix; for (const auto& row : metadata.matrix) for (unsigned c = 0; c < channels; ++c) matrix.push_back(row[c]);
          // Source colorants retain their exact s15Fixed16 values; do not serialize them again.
          Values pcs;
          for (unsigned r = 0; r < 3; ++r) {
            double value = 0, radius = 0, magnitude = 1;
            for (unsigned c = 0; c < channels; ++c) { const double a = matrix[r * channels + c]; value += a * linear[c].x; radius += std::abs(a) * linear[c].radius; magnitude += std::abs(a * linear[c].x); }
            pcs.push_back({value, radius + epsilon * magnitude});
          }
          for (unsigned c = 0; c < 3; ++c) {
            double scale = 1, offset = 0;
            if (intent == 3) scale = white[c] / d50[c]; // BToD3 already consumes absolute PCS.
            if (intent == 0 || intent == 2) { scale = (d50[c] - reference_black[c]) / (d50[c] - black[c]); offset = reference_black[c] - scale * black[c]; }
            pcs[c] = {pcs[c].x * scale + offset, pcs[c].radius * std::abs(scale) + epsilon * (1 + std::abs(pcs[c].x * scale) + std::abs(offset))};
          }
          Values native_bound = pcs;
          // Native CMM fixed-PCS/white and curve interpolation uncertainty is separate
          // from the primary GPU interval, as in the earlier matrix/TRC corpus.
          for (auto& v : native_bound) v.radius += (3.0 / 65535) * (1 + std::abs(v.x));
          auto finish = [&](Values values) {
            if (target.lab) values = Lab(values, false);
            for (const auto& stage : target.stages) values = stage.apply(values);
            return Intent(intent).apply(values);
          };
          const auto value = finish(pcs), native_interval = finish(native_bound);
          for (unsigned c = 0; c < 3; ++c) Record(records, native[pixel * 3 + c], value[c], native_interval[c].radius);
        }
        Save(out / (name + "_to_" + target.name + "_" + std::to_string(intent) + ".reference"), records);
      }
      cmsCloseProfile(profile);
    }
    cmsCloseProfile(source);
  }
}

// Full-range scalar probes are separate from the native corpus. Little CMS 2.19
// substitutes +/-1e22 for unbounded segment endpoints (cmstypes.c), so it cannot
// supply a full-F32-range oracle. These equations use host f64 log1p/sqrt/exp2;
// they share neither the production scaled arithmetic nor a pixel evaluator.
struct ScalarProbe { std::string name; Curve curve; std::function<Value(double)> expected; };
void ScalarCorpus(const fs::path& out, const std::vector<ScalarProbe>& probes, const std::vector<float>& values) {
  fs::create_directories(out);
  Bytes input;
  for (size_t pixel = 0; pixel < values.size(); ++pixel) for (size_t c = 0; c < 3; ++c) LE(input,Bits(values[(pixel + c*17) % values.size()]));
  Save(out / "input.f32le",input);
  std::ofstream manifest(out / "manifest.json");
  manifest << "{\"width\":" << values.size() << ",\"height\":1,\"profiles\":[";
  for (size_t i = 0; i < probes.size(); ++i) {
    const auto& p = probes[i];
    const Case c{p.name,false,{Curves({p.curve,p.curve,p.curve})}};
    Save(out / (p.name + ".icc"),Profile(c,true));
    manifest << (i ? "," : "") << "{\"name\":\"" << p.name << "\",\"channels\":3}";
    Bytes reference;
    for (size_t pixel = 0; pixel < values.size(); ++pixel) for (size_t channel = 0; channel < 3; ++channel) {
      const auto value = p.expected(values[(pixel + channel*17) % values.size()]);
      Check(std::isfinite(value.x) && std::abs(value.x) <= std::numeric_limits<float>::max() && std::isfinite(value.radius),"finite range reference");
      for (double v : {value.x,value.radius}) {
        uint64_t word; std::memcpy(&word,&v,8); for (unsigned b = 0; b < 8; ++b) reference.push_back(static_cast<uint8_t>(word >> (b*8)));
      }
    }
    Save(out / (p.name + ".reference"),reference);
  }
  manifest << "]}\n"; Check(bool(manifest),"write range manifest");
  std::cout << out.filename() << " independent MPE components: " << probes.size()*values.size()*3 << '\n';
}

std::vector<float> RangeInputs() {
  const double largest = std::numeric_limits<float>::max(), smallest = std::numeric_limits<float>::min();
  std::vector<float> values{0,-0.0f,std::numeric_limits<float>::denorm_min(),-std::numeric_limits<float>::denorm_min(),
    std::nextafter(static_cast<float>(smallest),0.0f),static_cast<float>(smallest),-static_cast<float>(smallest),
    static_cast<float>(largest),-static_cast<float>(largest)};
  for (unsigned word = 2; word <= 17; ++word) {
    const float value = std::ldexp(static_cast<float>(word),-149);
    values.push_back(value); values.push_back(-value);
  }
  for (int exponent = -126; exponent <= 127; exponent += 5) {
    const float x = std::ldexp(.75f,exponent);
    for (float v : {x,-x,std::nextafter(x,0.0f),std::nextafter(x,std::numeric_limits<float>::infinity())}) values.push_back(v);
  }
  for (float x : {-2.f,-1.f,-.5f,0.f,.5f,1.f,2.f,3.f,1e-30f,1e-20f,1e-10f,1e-5f}) {
    values.push_back(std::nextafter(x,-std::numeric_limits<float>::infinity()));
    values.push_back(x); values.push_back(std::nextafter(x,std::numeric_limits<float>::infinity()));
  }
  return values;
}

void ScalarRange(const fs::path& out) {
  const double inf = std::numeric_limits<double>::infinity();
  const double largest = std::numeric_limits<float>::max(), smallest = std::numeric_limits<float>::min();
  auto bounded = [](double value, double magnitude = 0) {
    return Value{value, 16 * epsilon * (1 + std::max(std::abs(value), magnitude))};
  };
  const std::vector<ScalarProbe> probes{
    {"identity", {{inf,0,{1,1,0,0}}}, [](double x) { return Value{x,0}; }},
    {"half", {{inf,0,{1,.5,0,0}}}, [](double x) { return Value{x/2,static_cast<double>(std::numeric_limits<float>::denorm_min())/2}; }},
    {"log_square", {{inf,1,{2,1,1,1,0}}}, [=](double x) { return bounded(std::log1p(x*x) / std::log(10.0)); }},
    {"log_small_increment", {{inf,1,{2,0x1p100,1,1,0}}}, [=](double x) { return bounded(0x1p100 * std::log1p(x*x) / std::log(10.0)); }},
    {"log_huge_power", {{inf,1,{largest,smallest,1,1,0}}}, [=](double x) {
      if (x == 0) return bounded(0);
      const double exponent = largest * std::log(std::abs(x));
      const double logarithm = std::max(0.0, exponent) + std::log1p(std::exp(-std::abs(exponent)));
      return bounded(smallest * logarithm / std::log(10.0));
    }},
    {"log_difference", {{-1,0,{1,0,0,.25}}, {0,1,{1,0x1p100,1,1,0}}, {std::nextafter(1.f,0.f),1,{1,0x1p100,-1,1,0}}, {inf,0,{1,0,0,.5}}}, [=](double x) {
      return bounded(x <= -1 ? .25 : x > std::nextafter(1.f,0.f) ? .5 : 0x1p100 * std::log1p(-std::abs(x)) / std::log(10.0));
    }},
    {"log_zero_scale", {{inf,1,{2,0,1,1,.5}}}, [=](double) { return bounded(.5); }},
    {"root_large_product", {{0,0,{1,0,0,0}}, {inf,0,{.5,largest,0,0}}}, [=](double x) { return bounded(x <= 0 ? 0 : std::sqrt(largest * x)); }},
    {"exponential", {{-1,0,{1,0,0,.25}}, {1,2,{smallest,2,1,140,0}}, {inf,0,{1,0,0,.5}}}, [=](double x) { return bounded(x <= -1 ? .25 : x <= 1 ? std::exp2(14+x) : .5); }},
    {"exponential_zero_scale", {{inf,2,{0,2,largest,largest,.125}}}, [=](double) { return bounded(.125); }},
    {"sampled_wide", {{-largest,0,{1,0,0,-largest/2}}, {largest,3,{largest/2}}, {inf,0,{1,0,0,largest/2}}}, [=](double x) { return bounded(x/2, largest); }},
    {"sampled_narrow", {{-smallest,0,{1,0,0,-.5}}, {smallest,3,{.5}}, {inf,0,{1,0,0,.5}}}, [=](double x) { return bounded(std::clamp(x / (2*smallest), -.5, .5)); }},
    {"sampled_empty", {{0,0,{1,0,0,.25}}, {0,3,{7,8}}, {1,3,{9}}, {inf,0,{1,0,0,9}}}, [=](double x) { return bounded(x <= 0 ? .25 : x <= 1 ? 8+x : 9); }},
  };
  ScalarCorpus(out,probes,RangeInputs());
}

void PowerRange(const fs::path& out) {
  const double inf = std::numeric_limits<double>::infinity();
  const double largest = std::numeric_limits<float>::max(), smallest = std::numeric_limits<float>::min();
  const double step = 0x1p-23;
  // Relative error plus half a subnormal ULP makes cancellation probes non-vacuous.
  // These bounds are fixed independently of the GPU and never use the cancelled operands.
  auto tight = [](double value) {
    return Value{value,16 * epsilon * std::abs(value) + static_cast<double>(std::numeric_limits<float>::denorm_min()) / 2};
  };
  std::vector<ScalarProbe> probes;
  auto add = [&](const std::string& name, Curve curve, std::function<double(double)> equation) {
    probes.push_back({name,std::move(curve),[=](double x) { return tight(equation(x)); }});
  };
  auto bounded = [&](const std::string& name, std::vector<double> parameters, double lower, double upper, std::function<double(double)> equation) {
    add(name,{{lower,0,{1,0,0,.25}}, {upper,0,std::move(parameters)}, {inf,0,{1,0,0,.5}}},
      [=](double x) { return x <= lower ? .25 : x > upper ? .5 : equation(x); });
  };
  for (bool negative_gamma : {false,true}) for (bool negative_slope : {false,true}) {
    const double gamma = negative_gamma ? -largest : largest, a = negative_slope ? -smallest : smallest;
    bounded(std::string("unit_") + (negative_gamma ? "negative" : "positive") + "_" + (negative_slope ? "decrease" : "increase"),
      {gamma,a,1,0},-1,1,[=](double x) { return std::exp(gamma * std::log1p(a*x)); });
  }
  bounded("positive_offset",{2,smallest,1,-1},-1,1,[=](double x) { return std::expm1(2 * std::log1p(smallest*x)); });
  bounded("negative_odd_offset",{3,smallest,-1,1},-1,1,[=](double x) { return -std::expm1(3 * std::log1p(-smallest*x)); });
  bounded("negative_even_offset",{2,smallest,-1,-1},-1,1,[=](double x) { return std::expm1(2 * std::log1p(-smallest*x)); });
  bounded("negative_reciprocal_offset",{-1,smallest,-1,1},-1,1,[=](double x) { return -std::expm1(-std::log1p(-smallest*x)); });
  for (bool negative : {false,true}) {
    const double gamma = negative ? -smallest : smallest;
    add(negative ? "tiny_negative_gamma" : "tiny_positive_gamma",{{0,0,{1,0,0,.25}}, {inf,0,{gamma,1,0,-1}}},
      [=](double x) { return x <= 0 ? .25 : std::expm1(gamma * std::log(x)); });
  }
  // Each f64 product below is exact: both multiplicands are F32. At the upper
  // breakpoint it is 1 - 2^-46, whose discarded remainder determines exp(-1).
  bounded("product_below_one",{0x1p46,1+step,0,0},.5,1-step,[=](double x) { return std::exp(0x1p46 * std::log1p((1+step)*x-1)); });
  bounded("product_above_one",{0x1p46,1-step,0,0},.5,1+step,[=](double x) { return std::exp(0x1p46 * std::log1p((1-step)*x-1)); });
  bounded("product_negative_odd",{16777215,-(1+step),0,0},.5,1-step,[=](double x) { return -std::exp(16777215 * std::log1p((1+step)*x-1)); });
  bounded("product_squared_offset",{2,1+step,0,-1},.5,1-step,[=](double x) { const double delta = (1+step)*x-1; return delta*(2+delta); });
  bounded("product_affine_offset",{1,1+step,-1,0},-1,1,[=](double x) { return (1+step)*x-1; });
  bounded("product_outer_offset",{1,1+step,0,-1},-1,1,[=](double x) { return (1+step)*x-1; });
  add("affine_double_offset",{{inf,0,{1,smallest,1,-1}}},[=](double x) { return smallest*x; });
  add("zero_gamma",{{inf,0,{0,largest,largest,-1}}},[](double) { return 0; });
  auto sampled = [&](const std::string& name, std::vector<double> parameters, double end, std::function<double(double)> equation) {
    const double initial = static_cast<float>(equation(1));
    add(name,{{-1,0,{1,0,0,.25}}, {1,0,std::move(parameters)}, {2,3,{end}}, {inf,0,{1,0,0,end}}},
      [=](double x) { return x <= -1 ? .25 : x <= 1 ? equation(x) : x <= 2 ? initial+(x-1)*(end-initial) : end; });
  };
  sampled("sampled_increase",{largest,smallest,1,0},.5,[=](double x) { return std::exp(largest * std::log1p(smallest*x)); });
  sampled("sampled_decrease",{largest,-smallest,1,0},.5,[=](double x) { return std::exp(largest * std::log1p(-smallest*x)); });
  sampled("sampled_positive_offset",{2,smallest,1,-1},4*smallest,[=](double x) { return std::expm1(2 * std::log1p(smallest*x)); });
  sampled("sampled_negative_odd_offset",{3,smallest,-1,1},6*smallest,[=](double x) { return -std::expm1(3 * std::log1p(-smallest*x)); });
  auto values = RangeInputs();
  for (int n = -256; n <= 512; ++n) values.push_back(static_cast<float>(n) / 256);
  for (float center : {static_cast<float>(1-step),static_cast<float>(1+step),-.5f,.5f,1.f,2.f}) {
    float below = center, above = center;
    values.push_back(center);
    for (unsigned n = 0; n < 4; ++n) {
      below = std::nextafter(below,-std::numeric_limits<float>::infinity());
      above = std::nextafter(above,std::numeric_limits<float>::infinity());
      values.push_back(below); values.push_back(above);
    }
  }
  ScalarCorpus(out,probes,values);
}

int main(int argc, char** argv) try {
  Check(argc == 3 && !fs::exists(argv[2]), "usage: mpe EMBEDDED_ICC_DIRECTORY NEW_OUTPUT_DIRECTORY"); Check(cmsGetEncodedCMMversion() == 2190, "requires Little CMS 2.19");
  const fs::path out(argv[2]); fs::create_directories(out);
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<Case> cases{
    {"matrix", false, {Matrix(3, 3, {1.25,-.125,.0625, .03125,.75,.125, -.125,.25,1.125}, {-.125,.0625,.25})}},
    {"segments", false, {Curves({{{0,0,{1,1.25,-.0625,.125}}, {1,0,{2,1.25,0,.0625}}, {inf,0,{1,1.25,.375,0}}}, {{0,0,{1,.5,0,0}}, {1,3,{.08,.5,.75,1.125}}, {inf,0,{1,2,-.875,0}}}, {{-.5,0,{1,1,0,0}}, {0,1,{1,.8,1,1,.2}}, {0,0,{1,2,0,7}}, {.5,2,{.5,2,1,0,-.5}}, {inf,0,{1,.75,.0625,0}}}})}},
    {"lab", true, {Matrix(3,3,{100,0,0, 0,192,0, 0,0,192}, {0,-96,-96})}},
  };
  for (unsigned n = 1; n <= 5; ++n) {
    std::vector<double> a(n * 3), offset(n); for (unsigned i = 0; i < n; ++i) { a[i * 3 + i % 3] = 0.75 + i / 8.0; offset[i] = -static_cast<double>(i) / 32.0; }
    std::vector<unsigned> grid(n); for (unsigned i = 0; i < n; ++i) grid[i] = 2 + i % 3;
    cases.push_back({"clut" + std::to_string(n), false, {Matrix(3,n,a,offset), Clut(grid)}});
  }
  std::vector<double> expand(15 * 3), contract(3 * 15); std::vector<Curve> middle;
  for (unsigned i = 0; i < 15; ++i) {
    expand[i * 3 + i % 3] = 0.125 * (i + 1); contract[(i % 3) * 15 + i] = 1.0 / 16;
    middle.push_back({{inf,0,{1,1 + (i % 3) / 8.0,0,-static_cast<double>(i) / 64}}});
  }
  cases.push_back({"channels15", false, {Matrix(3,15,expand,std::vector<double>(15)), Curves(middle), Matrix(15,3,contract,{-.25,.125,.5})}});
  Case identity{"identity", false, {Identity(3)}}; auto identity_bytes = Profile(identity, true); Save(out / "identity.icc", identity_bytes);
  auto identity_profile = cmsOpenProfileFromMem(identity_bytes.data(), static_cast<cmsUInt32Number>(identity_bytes.size())); Check(identity_profile, "open identity profile");
  std::vector<float> input; const std::vector<float> edges{-1.25f,-.5f,std::nextafter(-.5f,0.f),0,-0.0f,std::nextafter(0.f,-1.f),std::nextafter(0.f,1.f),.25f,1.f/3,.5f,std::nextafter(.5f,1.f),2.f/3,.75f,1,std::nextafter(1.f,2.f),1.25f,2};
  for (unsigned pixel = 0; pixel < 629; ++pixel) for (unsigned c = 0; c < 3; ++c) input.push_back(pixel < edges.size() ? edges[(pixel + c * 5) % edges.size()] : static_cast<float>(((pixel * (17 + c * 8) + c * 13) % 101) / 64.0 - .25));
  Bytes input_bytes; for (float x : input) LE(input_bytes, Bits(x)); Save(out / "input.f32le", input_bytes);
  std::ofstream manifest(out / "manifest.json"); manifest << "{\"width\":37,\"height\":17,\"profiles\":[";
  size_t components = 0;
  for (size_t index = 0; index < cases.size(); ++index) {
    const auto& c = cases[index]; auto bytes = Profile(c); Save(out / (c.name + ".icc"), bytes);
    auto profile = cmsOpenProfileFromMem(bytes.data(), static_cast<cmsUInt32Number>(bytes.size())); Check(profile, "open test profile");
    manifest << (index ? "," : "") << "{\"name\":\"" << c.name << "\",\"channels\":3}";
    for (unsigned intent = 0; intent < 4; ++intent) for (bool reverse : {false, true}) {
      std::cerr << c.name << " intent " << intent << " reverse " << reverse << '\n';
      auto transform = cmsCreateTransform(reverse ? identity_profile : profile, TYPE_RGB_FLT, reverse ? profile : identity_profile, TYPE_RGB_FLT, intent, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE); Check(transform, "create native transform");
      std::vector<float> native(input.size()); cmsDoTransform(transform, input.data(), native.data(), 629); cmsDeleteTransform(transform);
      Bytes records;
      for (size_t pixel = 0; pixel < 629; ++pixel) {
        Values value; for (unsigned c = 0; c < 3; ++c) value.push_back({input[pixel * 3 + c], 0});
        if (reverse && c.lab) value = Lab(value, false);
        for (const auto& stage : c.stages) value = stage.apply(value);
        value = Intent(intent).apply(value);
        if (!reverse && c.lab) value = Lab(value, true);
        for (unsigned ch = 0; ch < 3; ++ch) { Record(records, native[pixel * 3 + ch], value[ch]); ++components; }
      }
      Save(out / (c.name + (reverse ? "_reverse_" : "_forward_") + std::to_string(intent) + ".reference"), records);
    }
    cmsCloseProfile(profile);
  }
  manifest << "]}\n"; Check(bool(manifest), "write manifest"); cmsCloseProfile(identity_profile);
  Decoder(cases, argv[1], out / "decoder");
  ScalarRange(out / "range");
  PowerRange(out / "power");
  std::cout << "independent and native MPE components: " << components << '\n';
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
