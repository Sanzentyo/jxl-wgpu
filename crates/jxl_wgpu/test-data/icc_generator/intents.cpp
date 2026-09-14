// Offline reference generation. Neither Little CMS nor this evaluator is a
// production dependency.
#include <icc/intents.hpp>

#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>

namespace {
using Bytes = std::vector<unsigned char>;
using Handle = std::unique_ptr<void, decltype(&cmsCloseProfile)>;
using Curve = std::unique_ptr<cmsToneCurve, decltype(&cmsFreeToneCurve)>;
constexpr unsigned kWidth = 17, kHeight = 9, kVariants = 13;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

Bytes Read(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "read " + path.string());
  return Bytes(std::istreambuf_iterator<char>(input), {});
}

void Write(const std::filesystem::path &path, const Bytes &bytes) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  Check(output.good(), "write " + path.string());
}

void Word(Bytes &bytes, uint32_t value) {
  for (unsigned i = 0; i < 4; ++i)
    bytes.push_back(static_cast<unsigned char>(value >> (8 * i)));
}

void Float(Bytes &bytes, float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  Word(bytes, bits);
}

Bytes Serialize(cmsHPROFILE profile) {
  cmsUInt32Number size = 0;
  Check(cmsSaveProfileToMem(profile, nullptr, &size), "profile size");
  Bytes bytes(size);
  Check(cmsSaveProfileToMem(profile, bytes.data(), &size),
        "profile serialization");
  const std::array<unsigned char, 12> date{0x07, 0xea, 0, 1, 0, 1,
                                           0,    0,    0, 0, 0, 0};
  std::copy(date.begin(), date.end(), bytes.begin() + 24);
  std::fill(bytes.begin() + 84, bytes.begin() + 100, 0);
  return bytes;
}

struct Profile {
  std::string name;
  Bytes bytes;
  Handle handle;
  unsigned channels;
  std::vector<float> input;

  Profile(std::string name, Bytes bytes, bool gray)
      : name(std::move(name)), bytes(std::move(bytes)),
        handle(cmsOpenProfileFromMem(this->bytes.data(), this->bytes.size()),
               cmsCloseProfile),
        channels(gray ? 1 : 3) {
    Check(handle != nullptr, "reopen serialized profile");
    for (unsigned pixel = 0; pixel < kWidth * kHeight; ++pixel)
      for (unsigned c = 0; c < channels; ++c)
        input.push_back(pixel < 2 ? static_cast<float>(pixel)
                                  : ((pixel * 37 + c * 61) % 257) / 256.0f);
  }
};

Curve BuildCurve(unsigned variant, unsigned channel) {
  if (variant == 12) {
    const std::array<double, 4> parameters{2, 0.75, 0, 0.4375};
    return Curve(cmsBuildParametricToneCurve(nullptr, 3, parameters.data()),
                 cmsFreeToneCurve);
  }
  if (variant >= 10) {
    const double offset = (channel + 1) / 64.0;
    const std::array<double, 7> parameters =
        variant == 10 ? std::array<double, 7>{2, 1, 0, 0.5, 0.5, offset, offset}
                      : std::array<double, 7>{2, 0.75, 0.125, 0.125, 0, 0, 0};
    return Curve(cmsBuildParametricToneCurve(nullptr, variant == 10 ? 5 : 3,
                                             parameters.data()),
                 cmsFreeToneCurve);
  }
  std::array<cmsUInt16Number, 9> table{768,   1800,  4400,  8500, 14400,
                                       23200, 34900, 49100, 65535};
  if (variant >= 6) {
    const double black = variant == 6   ? 0.25
                         : variant == 7 ? 0.90
                         : variant == 8 ? 0.06 + 0.085 * channel
                                        : 0.38 + 0.085 * channel;
    for (size_t i = 0; i < table.size(); ++i)
      table[i] = static_cast<cmsUInt16Number>(
          std::round(65535 * (black + (1 - black) * i / (table.size() - 1))));
  }
  return Curve(
      cmsBuildTabulatedToneCurve16(nullptr, table.size(), table.data()),
      cmsFreeToneCurve);
}

void CheckAnalyticalEndpoints() {
  // The inverse of min(x*x + offset, 1) reaches its final plateau at
  // sqrt(1-offset), even if re-evaluating that root rounds below one.
  for (unsigned numerator = 1; numerator < 32; ++numerator) {
    const double offset = numerator / 64.0;
    const std::array<double, 7> parameters{2, 1, 0, .5, .5, offset, offset};
    Curve native(cmsBuildParametricToneCurve(nullptr, 5, parameters.data()),
                 cmsFreeToneCurve);
    Check(native != nullptr, "analytical endpoint curve");
    const scalar::Curve curve(native.get());
    const double endpoint = std::sqrt(1 - offset);
    Check(curve.Inverse(1) == endpoint && curve.Inverse(2) == endpoint,
          "first analytical point of the clipped terminal plateau");
    const double below = std::nextafter(1.0, 0.0);
    Check(std::abs(curve.Inverse(below) - std::sqrt(below - offset)) < 1e-15,
          "analytical root just before the terminal plateau");
  }
}

std::vector<Profile> Profiles(const std::filesystem::path &directory) {
  std::vector<Profile> profiles;
  for (bool gray : {false, true}) {
    const Bytes original = Read(directory / (gray ? "gray.icc" : "rgb.icc"));
    for (unsigned variant = 0; variant < kVariants; ++variant) {
      Handle profile(cmsOpenProfileFromMem(original.data(), original.size()),
                     cmsCloseProfile);
      Check(profile != nullptr, "open base profile");
      cmsSetProfileVersion(profile.get(),
                           variant == 1 || variant == 5 ? 2.4 : 4.4);
      cmsSetDeviceClass(profile.get(), variant == 2 || variant == 3
                                           ? cmsSigInputClass
                                           : cmsSigDisplayClass);
      if (variant == 2 || variant == 3 || variant == 5) {
        const cmsCIEXYZ white = variant == 3   ? cmsCIEXYZ{0.72, 0.81, 0.59}
                                : variant == 5 ? cmsCIEXYZ{0.95047, 1, 1.08883}
                                               : cmsCIEXYZ{0.81, 0.92, 0.67};
        Check(cmsWriteTag(profile.get(), cmsSigMediaWhitePointTag, &white),
              "media white");
      }
      if (variant >= 4) {
        const std::array<cmsTagSignature, 3> rgb{
            cmsSigRedTRCTag, cmsSigGreenTRCTag, cmsSigBlueTRCTag};
        for (unsigned c = 0; c < (gray ? 1u : 3u); ++c) {
          Curve curve = BuildCurve(variant, c);
          Check(curve != nullptr &&
                    cmsWriteTag(profile.get(), gray ? cmsSigGrayTRCTag : rgb[c],
                                curve.get()),
                "curve");
        }
      }
      profiles.emplace_back(std::string(gray ? "gray" : "rgb") + "_" +
                                std::to_string(variant),
                            Serialize(profile.get()), gray);
    }
  }
  return profiles;
}

void WriteReferences(const std::filesystem::path &directory, const std::string &name,
                     std::vector<float> native, const scalar::References &reference,
                     bool unit_output) {
  Check(native.size() == reference.exact.size(), "reference dimensions");
  Bytes bytes;
  for (size_t i = 0; i < native.size(); ++i) {
    Check(std::isfinite(native[i]), "finite native output");
    if (unit_output)
      native[i] = std::clamp(native[i], 0.0f, 1.0f);
    if (reference.native_semantics[i] == 0) {
      if (native[i] < reference.native_lower[i] || native[i] > reference.native_upper[i])
        std::cerr << std::setprecision(17) << name << " component " << i
                  << " native " << native[i] << " exact " << reference.exact[i]
                  << " interval [" << reference.native_lower[i] << ", "
                  << reference.native_upper[i] << "]\n";
      Check(native[i] >= reference.native_lower[i] && native[i] <= reference.native_upper[i],
            name + " outside native interval at component " + std::to_string(i));
    }
    for (float value :
         {native[i], reference.exact[i], reference.lower[i], reference.upper[i],
          reference.native_lower[i], reference.native_upper[i]})
      Float(bytes, value);
    Word(bytes, reference.native_semantics[i]);
  }
  Write(directory / (name + ".reference"), bytes);
}

std::vector<float> Native(const Profile &source, const Profile &target, unsigned intent) {
  std::vector<float> native(kWidth * kHeight * target.channels);
  std::unique_ptr<void, decltype(&cmsDeleteTransform)> transform(
      cmsCreateTransform(source.handle.get(),
                         source.channels == 1 ? TYPE_GRAY_FLT : TYPE_RGB_FLT,
                         target.handle.get(),
                         target.channels == 1 ? TYPE_GRAY_FLT : TYPE_RGB_FLT,
                         intent, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE),
      cmsDeleteTransform);
  Check(transform != nullptr, "native transform");
  cmsDoTransform(transform.get(), source.input.data(), native.data(), kWidth * kHeight);
  return native;
}

void Convert(const Profile &source, const Profile &target, unsigned intent,
             const std::filesystem::path &directory) {
  const auto reference = intents::Convert(
      source.handle.get(), target.handle.get(), source.input, intent);
  const std::string name = source.name + "_to_" + target.name + "_" + std::to_string(intent);
  WriteReferences(directory, name, Native(source, target, intent), reference, true);
}

void LinearReferences(const std::vector<Profile> &profiles,
                      const std::filesystem::path &directory) {
  std::filesystem::create_directory(directory);
  std::vector<float> signed_input;
  Bytes bytes;
  for (unsigned pixel = 0; pixel < kWidth * kHeight; ++pixel)
    for (unsigned c = 0; c < 3; ++c) {
      const float value = ((pixel * 37 + c * 61) % 257) / 128.0f - 0.5f;
      signed_input.push_back(value);
      Float(bytes, value);
    }
  Write(directory / "input.f32le", bytes);
  for (const auto &space : connection::kSpaces)
    for (const auto &profile : profiles)
      for (bool to_linear : {false, true})
        for (unsigned intent = 0; intent < 4; ++intent) {
          const auto &input = to_linear ? profile.input : signed_input;
          const auto reference = intents::LinearReference(
              profile.handle.get(), space, input, to_linear, intent);
          const auto native = connection::Native(
              profile.handle.get(), space, input, to_linear, intent);
          const std::string name = (to_linear ? profile.name + "_to_" + space.name
                                              : space.name + "_to_" + profile.name) +
                                   "_" + std::to_string(intent);
          WriteReferences(directory, name, native, reference, !to_linear);
        }
}

std::vector<float> Floats(const Bytes &bytes) {
  Check(bytes.size() % 4 == 0, "complete F32 reference");
  std::vector<float> values;
  for (size_t i = 0; i < bytes.size(); i += 4) {
    uint32_t bits = 0;
    for (unsigned c = 0; c < 4; ++c) bits |= uint32_t{bytes[i + c]} << (8 * c);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    Check(std::isfinite(value), "finite source reference");
    values.push_back(value);
  }
  return values;
}

std::vector<float> ReadHexFloats(const std::filesystem::path &path) {
  Bytes bytes;
  int high = -1;
  for (unsigned char c : Read(path)) {
    if (std::isspace(c)) continue;
    const int digit = c >= '0' && c <= '9' ? c - '0'
                      : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
    Check(digit >= 0, "hexadecimal reference");
    if (high < 0) high = digit;
    else { bytes.push_back(static_cast<unsigned char>(high * 16 + digit)); high = -1; }
  }
  Check(high == -1, "complete hexadecimal reference");
  return Floats(bytes);
}

void DecoderReferences(const std::vector<Profile> &profiles,
                       const std::filesystem::path &base,
                       const std::filesystem::path &output) {
  std::filesystem::create_directory(output);
  for (bool gray : {false, true}) for (bool modular : {false, true}) for (bool xyb : {false, true}) {
    const std::string color = gray ? "gray" : "rgb";
    const std::string name = color + (modular ? "_modular" : "_vardct") + (xyb ? "_xyb" : "_original");
    Profile source(name, Read(base / (color + ".icc")), gray);
    const auto rgba = xyb
        ? Floats(Read(base.parent_path() / "embedded_icc_xyb" / (name + ".linear.native.f32le")))
        : ReadHexFloats(base / (name + ".native.f32.hex"));
    Check(rgba.size() == kWidth * kHeight * (source.channels + 1), "source dimensions");
    source.input.clear();
    const unsigned inputs = xyb ? 3 : source.channels;
    for (unsigned pixel = 0; pixel < kWidth * kHeight; ++pixel)
      for (unsigned c = 0; c < inputs; ++c)
        source.input.push_back(rgba[pixel * (source.channels + 1) + (gray ? 0 : c)]);
    Check(source.input.size() == kWidth * kHeight * inputs, "physical source dimensions");
    std::vector<float> corners;
    const unsigned count = 1u << inputs;
    for (unsigned pixel = 0; pixel < kWidth * kHeight; ++pixel) {
      for (unsigned mask = 0; mask < count; ++mask) for (unsigned c = 0; c < inputs; ++c) {
        const float value = source.input[pixel * inputs + c];
        const float sign = mask & (1u << c) ? 1 : -1;
        // Keep the established original-device bound. Lossless Modular is exact;
        // VarDCT permits 2e-5. XYB keeps its normalized native reconstruction bound.
        // Propagate these through every monotone curve and affine row.
        const double error = xyb ? (1 + std::abs(double{value})) / 1024 : modular ? 0 : 2e-5;
        corners.push_back(error == 0 ? value : std::nextafter(
            static_cast<float>(value + double{sign} * error),
            sign * std::numeric_limits<float>::infinity()));
      }
    }
    for (const auto &target : profiles) {
      if (target.name != "rgb_2" && target.name != "gray_2" &&
          target.name != "rgb_5" && target.name != "gray_5" &&
          target.name != "rgb_8" && target.name != "gray_8") continue;
      for (unsigned intent = 0; intent < 4; ++intent) {
        const auto convert = [&](const std::vector<float> &input) {
          return xyb ? intents::LinearReference(target.handle.get(), connection::kSpaces[0], input, false, intent)
                     : intents::Convert(source.handle.get(), target.handle.get(), input, intent);
        };
        auto reference = convert(source.input);
        const auto bounds = convert(corners);
        for (unsigned pixel = 0; pixel < kWidth * kHeight; ++pixel) for (unsigned c = 0; c < target.channels; ++c) {
          float lower = std::numeric_limits<float>::infinity(), upper = -lower;
          for (unsigned mask = 0; mask < count; ++mask) {
            const unsigned index = (pixel * count + mask) * target.channels + c;
            lower = std::min(lower, bounds.lower[index]);
            upper = std::max(upper, bounds.upper[index]);
          }
          const unsigned index = pixel * target.channels + c;
          reference.lower[index] = lower;
          reference.upper[index] = upper;
          Check(lower <= reference.exact[index] && reference.exact[index] <= upper, "decoder center bound");
        }
        const auto native = xyb ? connection::Native(target.handle.get(), connection::kSpaces[0], source.input, false, intent)
                                : Native(source, target, intent);
        WriteReferences(output, name + "_to_" + target.name + "_" + std::to_string(intent), native, reference, true);
      }
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  Check(argc == 3, "usage: intents BASE_PROFILE_DIRECTORY OUTPUT_DIRECTORY");
  Check(cmsGetEncodedCMMversion() == 2190, "Little CMS 2.19 required");
  CheckAnalyticalEndpoints();
  const auto profiles = Profiles(argv[1]);
  const std::filesystem::path output(argv[2]);
  Check(!std::filesystem::exists(output), "output directory already exists");
  std::filesystem::create_directories(output);
  std::ofstream manifest(output / "manifest.json");
  manifest << "{\"width\":" << kWidth << ",\"height\":" << kHeight
           << ",\"profiles\":[";
  for (size_t i = 0; i < profiles.size(); ++i) {
    const auto &profile = profiles[i];
    Write(output / (profile.name + ".icc"), profile.bytes);
    Bytes input;
    for (float value : profile.input)
      Float(input, value);
    Write(output / (profile.name + "_input.f32le"), input);
    if (i != 0)
      manifest << ',';
    manifest << "{\"name\":\"" << profile.name
             << "\",\"channels\":" << profile.channels << '}';
  }
  manifest << "],\"intents\":[0,1,2,3]}\n";
  Check(manifest.good(), "manifest");
  for (const auto &source : profiles)
    for (const auto &target : profiles)
      for (unsigned intent = 0; intent < 4; ++intent)
        Convert(source, target, intent, output);
  LinearReferences(profiles, output / "linear");
  DecoderReferences(profiles, argv[1], output / "decoder");
  std::cout << "Generated " << profiles.size() << " profiles and "
            << profiles.size() * profiles.size() * 4
            << " profile connections and 1040 linear connections\n";
}
