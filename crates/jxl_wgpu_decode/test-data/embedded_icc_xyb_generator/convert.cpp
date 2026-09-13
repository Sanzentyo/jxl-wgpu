#include <icc/linear.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

using Bytes = std::vector<uint8_t>;

void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

Bytes Read(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "input file");
  return Bytes(std::istreambuf_iterator<char>(input), {});
}

std::vector<float> ReadFloats(const std::string& path) {
  const auto bytes = Read(path);
  Check(bytes.size() % 4 == 0, "F32 length");
  std::vector<float> output(bytes.size() / 4);
  for (size_t i = 0; i < output.size(); ++i) {
    uint32_t bits = 0;
    for (unsigned c = 0; c < 4; ++c) bits |= static_cast<uint32_t>(bytes[4 * i + c]) << (8 * c);
    std::memcpy(&output[i], &bits, sizeof(bits));
    Check(std::isfinite(output[i]), "nonfinite source");
  }
  return output;
}

void WriteFloats(const std::string& path, const std::vector<float>& pixels) {
  std::ofstream output(path, std::ios::binary);
  for (const float pixel : pixels) {
    uint32_t bits;
    std::memcpy(&bits, &pixel, sizeof(bits));
    for (unsigned shift = 0; shift < 32; shift += 8) output.put(bits >> shift);
  }
  Check(output.good(), "output file");
}

int main(int argc, char** argv) {
  Check(argc == 4, "usage: convert probe_directory profiles_directory output_directory");
  Check(cmsGetEncodedCMMversion() == 2190, "Little CMS 2.19 required");
  std::filesystem::create_directories(argv[3]);
  for (bool gray : {false, true}) for (bool modular : {true, false}) {
    const auto name = std::string(gray ? "gray" : "rgb") + (modular ? "_modular_xyb" : "_vardct_xyb");
    const auto source = ReadFloats(std::string(argv[1]) + "/" + name + "_builtin_0.f32le");
    const unsigned source_channels = gray ? 2 : 4;
    Check(source.size() == 17 * 9 * source_channels, "source size");
    std::vector<float> rgb;
    for (size_t i = 0; i < source.size(); i += source_channels)
      for (size_t c = 0; c < 3; ++c) rgb.push_back(source[i + (gray ? 0 : c)]);
    for (bool target_gray : {false, true}) {
      const std::string target_name = target_gray ? "gray" : "rgb";
      const auto bytes = Read(std::string(argv[2]) + "/" + target_name + ".icc");
      std::unique_ptr<void, decltype(&cmsCloseProfile)> profile(
          cmsOpenProfileFromMem(bytes.data(), bytes.size()), cmsCloseProfile);
      Check(profile != nullptr, "target profile");
      const auto native = connection::Native(profile.get(), connection::kSpaces[0], rgb, false);
      const auto scalar = connection::Reference(scalar::Profile(profile.get()), connection::kSpaces[0], rgb, false);
      Check(native.size() == scalar.exact.size(), "reference sizes");
      const unsigned target_channels = target_gray ? 1 : 3;
      std::vector<float> native_alpha, scalar_alpha;
      double max_error = 0;
      for (size_t i = 0; i < native.size(); ++i) {
        Check(scalar.native_semantics[i] == 0, "unexpected native boundary semantics");
        Check(native[i] >= scalar.native_lower[i] && native[i] <= scalar.native_upper[i], "native precision interval");
        native_alpha.push_back(native[i]);
        scalar_alpha.push_back(scalar.exact[i]);
        max_error = std::max(max_error, std::abs(static_cast<double>(native[i]) - scalar.exact[i]));
        if (i % target_channels + 1 == target_channels) {
          const auto alpha = source[(i / target_channels + 1) * source_channels - 1];
          native_alpha.push_back(alpha);
          scalar_alpha.push_back(alpha);
        }
      }
      const auto prefix = std::string(argv[3]) + "/" + name + "_to_" + target_name;
      WriteFloats(prefix + ".native.f32le", native_alpha);
      WriteFloats(prefix + ".scalar.f32le", scalar_alpha);
      std::printf("{\"source\":\"%s\",\"target\":\"%s\",\"native_scalar_max_error\":%.17g,\"samples\":%zu}\n",
          name.c_str(), target_name.c_str(), max_error, native_alpha.size());
    }
  }
}
