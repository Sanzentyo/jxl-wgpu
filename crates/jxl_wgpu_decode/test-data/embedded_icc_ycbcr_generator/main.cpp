// Offline ICC presentation references from the existing independent YCbCr device corpus.
#include <icc/linear.hpp>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>

using Bytes = std::vector<unsigned char>;
using Profile = std::unique_ptr<void, decltype(&cmsCloseProfile)>;

void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

Profile Open(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  Check(in.good(), "profile file");
  const Bytes bytes(std::istreambuf_iterator<char>(in), {});
  Profile profile(cmsOpenProfileFromMem(bytes.data(), bytes.size()), cmsCloseProfile);
  Check(profile != nullptr, "profile parse");
  return profile;
}

std::vector<float> ReadWords(const std::filesystem::path& path) {
  std::ifstream in(path);
  Check(in.good(), "reference file");
  std::vector<float> result;
  std::string token;
  while (in >> token) {
    Check(token.size() == 8, "expected one hexadecimal F32 word per token");
    size_t consumed = 0;
    const auto integer = std::stoul(token, &consumed, 16);
    Check(consumed == token.size() && integer <= UINT32_MAX, "reference word");
    const auto word = static_cast<uint32_t>(integer);
    float value;
    std::memcpy(&value, &word, sizeof(value));
    Check(std::isfinite(value), "finite reference");
    result.push_back(value);
  }
  Check(in.eof(), "reference read");
  return result;
}

void WriteWords(const std::filesystem::path& path, const std::vector<float>& values) {
  std::ofstream out(path);
  out << std::hex << std::setfill('0');
  for (float value : values) {
    Check(std::isfinite(value), "finite generated reference");
    uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    out << std::setw(8) << word << '\n';
  }
  Check(out.good(), "reference write");
}

double Srgb(double value) {
  const double magnitude = std::abs(value);
  return std::copysign(magnitude <= 0.0031308 ? 12.92 * magnitude
      : 1.055 * std::pow(magnitude, 1.0 / 2.4) - 0.055, value);
}

scalar::References Reference(const scalar::Profile& source, const scalar::Profile& target,
                             const std::vector<float>& input, const std::string& kind) {
  return kind == "other" ? scalar::Convert(source, target, input)
      : connection::Reference(source, connection::kSpaces[0], input, true);
}

std::vector<float> Native(cmsHPROFILE source, cmsHPROFILE target,
                          const std::vector<float>& input, bool gray, const std::string& kind) {
  // Little CMS's floating API extrapolates device curves outside [0,1]. The ICC
  // primitive's specified unit-domain input and independent Forward both clamp there.
  // Apply that contract explicitly, including every out-of-gamut codec sample.
  std::vector<float> device = input;
  for (auto& value : device) value = std::clamp(value, 0.0f, 1.0f);
  if (kind != "other") return connection::Native(source, connection::kSpaces[0], device, true);
  const size_t pixels = input.size() / (gray ? 1 : 3);
  const size_t channels = gray ? 3 : 1;
  const std::unique_ptr<void, decltype(&cmsDeleteTransform)> transform(
      cmsCreateTransform(source, gray ? TYPE_GRAY_FLT : TYPE_RGB_FLT,
          target, gray ? TYPE_RGB_DBL : TYPE_GRAY_DBL,
          INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE),
      cmsDeleteTransform);
  Check(transform != nullptr, "native ICC transform");
  std::vector<double> output(pixels * channels);
  cmsDoTransform(transform.get(), device.data(), output.data(), static_cast<cmsUInt32Number>(pixels));
  return std::vector<float>(output.begin(), output.end());
}

struct Case {
  std::string directory, name;
  size_t pixels;
  bool gray, vardct;

  double ColorError(float value) const {
    // Exactly the existing corpus's reconstruction bound, before any ICC calculation.
    return vardct ? (1.0 / 1024.0) * (1.0 + std::abs(static_cast<double>(value)))
                  : static_cast<double>(2e-6f);
  }
  double AlphaError(float value) const {
    return static_cast<double>(2e-6f) * (vardct ? 1.0 + std::abs(static_cast<double>(value)) : 1.0);
  }
};

int main(int argc, char** argv) {
  Check(argc == 3, "usage: ycbcr-convert decoder_test_data_directory output_directory");
  Check(cmsGetEncodedCMMversion() == 2190, "Little CMS 2.19 required");
  const std::filesystem::path root = argv[1], output = argv[2];
  std::filesystem::create_directories(output);
  const std::array<Case, 5> cases{{
      {"modular_ycbcr", "sampling_123", 37 * 19, false, false},
      {"modular_ycbcr", "gray", 37 * 19, true, false},
      {"modular_ycbcr", "associated", 37 * 19, false, false},
      {"modular_ycbcr", "resampling_8", 53 * 35, false, false},
      {"original_color", "vardct_ycbcr_bt709_srgb_still", 37 * 19, false, true},
  }};
  for (const auto& item : cases) {
    const auto rgba = ReadWords(root / item.directory /
        (item.name + (item.vardct ? ".original.f32.hex" : ".f32.hex")));
    Check(rgba.size() >= item.pixels * 4, "RGBA reference dimensions");
    auto source = Open(root / "embedded_icc" / (item.gray ? "gray.icc" : "rgb.icc"));
    auto target = Open(root / "embedded_icc" / (item.gray ? "rgb.icc" : "gray.icc"));
    const scalar::Profile source_profile(source.get()), target_profile(target.get());
    for (const auto& profile : {source.get(), target.get()}) {
      const auto tags = cmsGetColorSpace(profile) == cmsSigGrayData
          ? std::vector<cmsTagSignature>{cmsSigGrayTRCTag}
          : std::vector<cmsTagSignature>{cmsSigRedTRCTag, cmsSigGreenTRCTag, cmsSigBlueTRCTag};
      for (const auto tag : tags) {
        const auto* curve = static_cast<const cmsToneCurve*>(cmsReadTag(profile, tag));
        Check(curve != nullptr && cmsIsToneCurveMonotonic(curve) &&
              !cmsIsToneCurveDescending(curve), "increasing source and target curves");
      }
    }
    const size_t inputs = item.gray ? 1 : 3;
    std::vector<float> colors;
    for (size_t pixel = 0; pixel < item.pixels; ++pixel)
      for (size_t c = 0; c < inputs; ++c) colors.push_back(rgba[pixel * 4 + c]);
    const size_t corners_per_pixel = size_t{1} << inputs;
    std::vector<float> corners;
    for (size_t pixel = 0; pixel < item.pixels; ++pixel) {
      for (size_t mask = 0; mask < corners_per_pixel; ++mask) {
        for (size_t c = 0; c < inputs; ++c) {
          const auto value = colors[pixel * inputs + c];
          const double sign = ((mask >> c) & 1) ? 1 : -1;
          // Outward rounding includes every representable GPU value within the source bound.
          corners.push_back(std::nextafter(static_cast<float>(value + sign * item.ColorError(value)),
              static_cast<float>(sign) * std::numeric_limits<float>::infinity()));
        }
      }
    }
    for (const std::string kind : {"linear", "srgb", "other"}) {
      const auto reference = Reference(source_profile, target_profile, colors, kind);
      const auto bounds = Reference(source_profile, target_profile, corners, kind);
      const auto native_colors = Native(source.get(), target.get(), colors, item.gray, kind);
      const size_t outputs = kind == "other" && !item.gray ? 1 : 3;
      Check(reference.exact.size() == item.pixels * outputs, "reference color count");
      Check(native_colors.size() == reference.exact.size(), "native color count");
      std::vector<float> exact, lower, upper, native;
      double maximum_native_error = 0, maximum_interval_width = 0;
      for (size_t pixel = 0; pixel < item.pixels; ++pixel) {
        for (size_t c = 0; c < outputs; ++c) {
          const size_t index = pixel * outputs + c;
          Check(reference.native_semantics[index] == 0, "native boundary semantics");
          Check(native_colors[index] >= reference.native_lower[index] &&
                native_colors[index] <= reference.native_upper[index], "native precision interval");
          double lo = std::numeric_limits<double>::infinity(), hi = -lo;
          for (size_t mask = 0; mask < corners_per_pixel; ++mask) {
            const size_t corner = (pixel * corners_per_pixel + mask) * outputs + c;
            lo = std::min(lo, static_cast<double>(bounds.lower[corner]));
            hi = std::max(hi, static_cast<double>(bounds.upper[corner]));
          }
          // Forward curves, matrix rows, and inverse curves are coordinate-wise monotone;
          // all extrema, including each arithmetic-error bound, occur at box corners.
          double value = reference.exact[index], native_value = native_colors[index];
          if (kind == "srgb") {
            value = Srgb(value); native_value = Srgb(native_value);
            lo = Srgb(lo); hi = Srgb(hi);
            // Preserve the shared output kernel's existing normalized F32 transfer bound.
            lo -= 2e-6 * (1 + std::abs(lo));
            hi += 2e-6 * (1 + std::abs(hi));
          }
          Check(value >= lo && value <= hi, "scalar interval contains its center");
          exact.push_back(static_cast<float>(value));
          native.push_back(static_cast<float>(native_value));
          lower.push_back(std::nextafter(static_cast<float>(lo), -std::numeric_limits<float>::infinity()));
          upper.push_back(std::nextafter(static_cast<float>(hi), std::numeric_limits<float>::infinity()));
          maximum_native_error = std::max(maximum_native_error, std::abs(value - native_value));
          maximum_interval_width = std::max(maximum_interval_width, hi - lo);
        }
        const auto alpha = rgba[pixel * 4 + 3];
        exact.push_back(alpha); native.push_back(alpha);
        lower.push_back(std::nextafter(static_cast<float>(alpha - item.AlphaError(alpha)),
            -std::numeric_limits<float>::infinity()));
        upper.push_back(std::nextafter(static_cast<float>(alpha + item.AlphaError(alpha)),
            std::numeric_limits<float>::infinity()));
      }
      const auto prefix = item.name + "." + kind;
      WriteWords(output / (prefix + ".scalar.f32.hex"), exact);
      WriteWords(output / (prefix + ".native.f32.hex"), native);
      WriteWords(output / (prefix + ".lower.f32.hex"), lower);
      WriteWords(output / (prefix + ".upper.f32.hex"), upper);
      std::cout << std::setprecision(17) << "{\"case\":\"" << item.name
          << "\",\"kind\":\"" << kind << "\",\"pixels\":" << item.pixels
          << ",\"source_channels\":" << inputs << ",\"target_channels\":" << outputs
          << ",\"clamped_native_input_values\":"
          << std::count_if(colors.begin(), colors.end(), [](float value) { return value < 0 || value > 1; })
          << ",\"maximum_native_error\":" << maximum_native_error
          << ",\"maximum_interval_width\":" << maximum_interval_width << "}\n";
    }
  }
}
