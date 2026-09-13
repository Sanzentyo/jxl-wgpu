// Offline Little CMS oracle. Production code has no dependency on this executable or lcms2.
#include <lcms2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "scalar.hpp"
#include "linear.hpp"

namespace {
using Profile = std::unique_ptr<void, decltype(&cmsCloseProfile)>;
using Curve = std::unique_ptr<cmsToneCurve, decltype(&cmsFreeToneCurve)>;
constexpr size_t kWidth = 37;
constexpr size_t kHeight = 17;

void Require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

void Write(const std::filesystem::path& path, const void* data, size_t size) {
  std::ofstream out(path, std::ios::binary);
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  Require(out.good(), "write " + path.string());
}

void WriteFloats(const std::filesystem::path& path, const std::vector<float>& values) {
  std::vector<uint8_t> bytes;
  for (float value : values) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    for (size_t i = 0; i < 4; ++i) bytes.push_back(static_cast<uint8_t>(bits >> (8 * i)));
  }
  Write(path, bytes.data(), bytes.size());
}

void WriteReferences(const std::filesystem::path& path, const std::vector<float>& native,
                     const scalar::References& reference) {
  std::vector<uint8_t> bytes;
  auto word = [&](uint32_t bits) {
    for (size_t i = 0; i < 4; ++i) bytes.push_back(static_cast<uint8_t>(bits >> (8 * i)));
  };
  for (size_t i = 0; i < native.size(); ++i) {
    for (float value : {native[i], reference.exact[i], reference.lower[i], reference.upper[i],
                       reference.native_lower[i], reference.native_upper[i]}) {
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      word(bits);
    }
    word(reference.native_semantics[i]);
  }
  Write(path, bytes.data(), bytes.size());
}

Curve BuildCurve(size_t profile, size_t channel) {
  std::array<double, 7> p{};
  int type = 1;
  if (profile == 1 || profile == 2 || profile >= 8) {
    p[0] = std::array<double, 3>{1.75, 2.1875, 2.5}[channel];
  } else if (profile == 3) {
    type = 2; p = {2.0, 1.25, -0.25, 0, 0, 0, 0};
  } else if (profile == 4) {
    type = 3; p = {2.0, 0.75, 0.0, 0.4375, 0, 0, 0};
  } else if (profile == 5) {
    type = 4; p = {2.0, 1.0, 0.0, 0.25, 0.25, 0, 0};
  } else if (profile == 6) {
    type = 5; p = {2.0, 0.75, 0.125, 0.25, 0.5, 0.234375, 0.359375};
  } else {
    const size_t count = 257 + 256 * channel;
    std::vector<uint16_t> values(count);
    for (size_t i = 0; i < count; ++i) {
      const double x = static_cast<double>(i) / static_cast<double>(count - 1);
      values[i] = static_cast<uint16_t>(std::lround(std::pow(x, 1.6 + 0.3 * channel) * 65535.0));
    }
    return Curve(cmsBuildTabulatedToneCurve16(nullptr, static_cast<cmsUInt32Number>(count), values.data()), cmsFreeToneCurve);
  }
  return Curve(cmsBuildParametricToneCurve(nullptr, type, p.data()), cmsFreeToneCurve);
}

Profile BuildProfile(size_t index) {
  if (index == 0) return Profile(cmsCreate_sRGBProfile(), cmsCloseProfile);
  std::array<Curve, 3> curves{BuildCurve(index, 0), BuildCurve(index, 1), BuildCurve(index, 2)};
  for (const auto& curve : curves) Require(curve != nullptr, "create curve");
  cmsCIExyY white{0.3127, 0.3290, 1.0};
  cmsCIExyYTRIPLE primaries{{0.64, 0.33, 1.0}, {0.30, 0.60, 1.0}, {0.15, 0.06, 1.0}};
  if (index == 8) return Profile(cmsCreateGrayProfile(cmsD50_xyY(), curves[0].get()), cmsCloseProfile);
  if (index == 9) {
    white = {0.3457, 0.3585, 1.0};
    primaries.Green = {0.21, 0.71, 1.0};
  }
  std::array<cmsToneCurve*, 3> raw{curves[0].get(), curves[1].get(), curves[2].get()};
  return Profile(cmsCreateRGBProfile(&white, &primaries, raw.data()), cmsCloseProfile);
}

std::vector<uint8_t> Serialize(cmsHPROFILE profile, size_t index) {
  cmsSetProfileVersion(profile, index == 1 ? 2.4 : 4.3);
  cmsSetHeaderRenderingIntent(profile, INTENT_RELATIVE_COLORIMETRIC);
  cmsUInt32Number size = 0;
  Require(cmsSaveProfileToMem(profile, nullptr, &size) != 0, "measure profile");
  std::vector<uint8_t> bytes(size);
  Require(cmsSaveProfileToMem(profile, bytes.data(), &size) != 0, "serialize profile");
  // A fixed timestamp and an unspecified profile ID make offline regeneration reproducible.
  const std::array<uint8_t, 12> date{0x07, 0xea, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0};
  std::copy(date.begin(), date.end(), bytes.begin() + 24);
  std::fill(bytes.begin() + 84, bytes.begin() + 100, 0);
  return bytes;
}

std::vector<float> Input(size_t channels) {
  std::vector<float> values(kWidth * kHeight * channels);
  const std::array<float, 14> boundary{0.0f, 1.0f, 0.00001f, 0.0001f, 0.0031308f,
      0.04044f, 0.04045f, 0.04046f, 0.19999f, 0.2f, 0.20001f, 0.24999f, 0.25f, 0.25001f};
  for (size_t pixel = 0; pixel < kWidth * kHeight; ++pixel) {
    for (size_t channel = 0; channel < channels; ++channel) {
      values[pixel * channels + channel] = pixel < boundary.size()
          ? boundary[pixel]
          : static_cast<float>((pixel * (37 + channel * 12) + channel * 101) % 1025) / 1024.0f;
    }
  }
  return values;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 2, "usage: icc_generator OUTPUT_DIRECTORY");
    Require(cmsGetEncodedCMMversion() == 2190, "oracle requires Little CMS 2.19");
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    const std::array<std::string, 10> names{"srgb", "gamma_v2", "gamma_v4", "threshold", "offset", "piecewise", "affine", "sampled", "gray", "wide"};
    std::vector<Profile> profiles;
    std::ofstream manifest(directory / "manifest.json");
    manifest << "{\n  \"oracle\": \"Little CMS 2.19, relative, NOOPTIMIZE | NOCACHE, unit output\",\n  \"width\": " << kWidth << ", \"height\": " << kHeight << ",\n  \"profiles\": [\n";
    for (size_t i = 0; i < names.size(); ++i) {
      auto profile = BuildProfile(i);
      Require(profile != nullptr, "create profile");
      const auto bytes = Serialize(profile.get(), i);
      Write(directory / (names[i] + ".icc"), bytes.data(), bytes.size());
      // Oracles always use the exact serialized fixed-point profile that the GPU will parse.
      profiles.emplace_back(cmsOpenProfileFromMem(bytes.data(), static_cast<cmsUInt32Number>(bytes.size())), cmsCloseProfile);
      Require(profiles.back() != nullptr, "reopen serialized profile");
      const size_t channels = i == 8 ? 1 : 3;
      manifest << "    {\"name\": \"" << names[i] << "\", \"channels\": " << channels << "}" << (i + 1 == names.size() ? "\n" : ",\n");
      WriteFloats(directory / (names[i] + "_input.f32le"), Input(channels));
    }
    manifest << "  ]\n}\n";
    Require(manifest.good(), "write manifest");
    for (size_t source = 0; source < names.size(); ++source) {
      const size_t source_channels = source == 8 ? 1 : 3;
      const auto input = Input(source_channels);
      for (size_t target = 0; target < names.size(); ++target) {
        const size_t target_channels = target == 8 ? 1 : 3;
        const auto transform = cmsCreateTransform(profiles[source].get(), source_channels == 1 ? TYPE_GRAY_FLT : TYPE_RGB_FLT,
            profiles[target].get(), target_channels == 1 ? TYPE_GRAY_FLT : TYPE_RGB_FLT,
            INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
        Require(transform != nullptr, "create transform");
        std::vector<float> output(kWidth * kHeight * target_channels);
        cmsDoTransform(transform, input.data(), output.data(), static_cast<cmsUInt32Number>(kWidth * kHeight));
        cmsDeleteTransform(transform);
        for (auto& value : output) {
          Require(std::isfinite(value), "non-finite native output");
          value = std::clamp(value, 0.0f, 1.0f);
        }
        const auto reference = scalar::Convert(scalar::Profile(profiles[source].get()), scalar::Profile(profiles[target].get()), input);
        const std::string prefix = names[source] + "_to_" + names[target];
        WriteReferences(directory / (prefix + ".reference"), output, reference);
      }
    }
    const auto linear_directory = directory / "linear";
    std::filesystem::create_directories(linear_directory);
    auto linear_input = Input(3);
    for (auto& value : linear_input) value = value * 1.5f - 0.25f;
    const std::array<float, 9> probes{0, 1, -0.25f, 1.25f, -0.1f, 0.2f, 0.0001f, -0.0001f, 0.5f};
    std::copy(probes.begin(), probes.end(), linear_input.begin());
    WriteFloats(linear_directory / "input.f32le", linear_input);
    std::ofstream linear_manifest(linear_directory / "manifest.json");
    linear_manifest.precision(17);
    linear_manifest << "{\"spaces\": [\n";
    for (size_t s = 0; s < connection::kSpaces.size(); ++s) {
      const auto& space = connection::kSpaces[s];
      linear_manifest << "{\"name\":\"" << space.name << "\",\"white\":[" << space.white[0] << ',' << space.white[1] << "],\"primaries\":[";
      for (size_t c = 0; c < 3; ++c) linear_manifest << '[' << space.primaries[c][0] << ',' << space.primaries[c][1] << ']' << (c == 2 ? "" : ",");
      linear_manifest << "]}" << (s + 1 == connection::kSpaces.size() ? "\n" : ",\n");
      for (size_t p = 0; p < profiles.size(); ++p) for (bool to_linear : {true, false}) {
        const auto input = to_linear ? Input(p == 8 ? 1 : 3) : linear_input;
        const auto reference = connection::Reference(scalar::Profile(profiles[p].get()), space, input, to_linear);
        const auto native = connection::Native(profiles[p].get(), space, input, to_linear);
        const std::string name = to_linear ? names[p] + "_to_" + space.name : space.name + "_to_" + names[p];
        WriteReferences(linear_directory / (name + ".reference"), native, reference);
      }
    }
    linear_manifest << "]}\n";
    Require(linear_manifest.good(), "write linear manifest");
    std::cout << "10 exact profiles, 100 profile pairs and 100 linear RGB connections, 629 samples per transform\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
