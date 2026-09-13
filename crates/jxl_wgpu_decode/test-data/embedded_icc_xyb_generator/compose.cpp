#include "bounds.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

using Bytes = std::vector<uint8_t>;
void Check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
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
    Check(std::isfinite(output[i]), "finite source");
  }
  return output;
}
void WriteFloats(const std::string& path, const std::vector<float>& pixels) {
  std::ofstream output(path, std::ios::binary);
  for (const float pixel : pixels) {
    uint32_t bits; std::memcpy(&bits, &pixel, sizeof(bits));
    for (unsigned shift = 0; shift < 32; shift += 8) output.put(bits >> shift);
  }
  Check(output.good(), "output file");
}

int main(int argc, char** argv) {
  Check(argc == 4, "usage: oracle layer_directory profiles_directory output_directory");
  Check(cmsGetEncodedCMMversion() == 2190, "Little CMS 2.19 required");
  std::filesystem::create_directories(argv[3]);
  for (bool gray : {false, true}) for (bool modular : {true, false}) {
    const auto name = std::string(gray ? "gray" : "rgb") + (modular ? "_modular" : "_vardct");
    const auto profile_bytes = Read(std::string(argv[2]) + (gray ? "/gray.icc" : "/rgb.icc"));
    std::unique_ptr<void, decltype(&cmsCloseProfile)> profile(
        cmsOpenProfileFromMem(profile_bytes.data(), profile_bytes.size()), cmsCloseProfile);
    Check(profile != nullptr, "original profile");
    const scalar::Profile model(profile.get());
    std::array<std::vector<float>, 2> linear, original;
    std::array<Bounds, 2> bounds;
    double native_scalar_max_error = 0;
    for (unsigned frame = 0; frame < 2; ++frame) {
      const auto input = ReadFloats(std::string(argv[1]) + "/" + name + "_layers_builtin.frame" + std::to_string(frame) + ".linear.f32le");
      Check(input.size() == 17 * 9 * (gray ? 1u : 3u), "layer geometry");
      for (size_t pixel = 0; pixel < 17 * 9; ++pixel)
        for (unsigned c = 0; c < 3; ++c) linear[frame].push_back(input[gray ? pixel : 3 * pixel + c]);
      const auto reference = connection::Reference(model, connection::kSpaces[0], linear[frame], false);
      const auto native = connection::Native(profile.get(), connection::kSpaces[0], linear[frame], false);
      for (size_t i = 0; i < native.size(); ++i) {
        Check(reference.native_semantics[i] == 0, "unexpected native boundary semantics");
        Check(native[i] >= reference.native_lower[i] && native[i] <= reference.native_upper[i], "native precision interval");
        native_scalar_max_error = std::max(native_scalar_max_error, std::abs(static_cast<double>(native[i]) - reference.exact[i]));
      }
      original[frame] = reference.exact;
      bounds[frame] = ReconstructionBounds(model, linear[frame]);
      const auto prefix = std::string(argv[3]) + "/" + name + ".frame" + std::to_string(frame);
      WriteFloats(prefix + ".device.scalar.f32le", reference.exact);
      WriteFloats(prefix + ".device.native.f32le", native);
      WriteFloats(prefix + ".device.lower.f32le", bounds[frame].lower);
      WriteFloats(prefix + ".device.upper.f32le", bounds[frame].upper);

    }
    std::vector<float> sum(original[0].size()), lower(sum.size()), upper(sum.size());
    float maximum_original = 0;
    for (size_t i = 0; i < sum.size(); ++i) {
      // Each reconstructed device surface is F32. Add in the original device domain;
      // do not apply a target-ICC unit curve to this same-profile presentation.
      sum[i] = static_cast<float>(static_cast<double>(original[0][i]) + original[1][i]);
      maximum_original = std::max(maximum_original, sum[i]);
      lower[i] = std::nextafter(static_cast<float>(static_cast<double>(bounds[0].lower[i]) + bounds[1].lower[i] - 2e-7), -std::numeric_limits<float>::infinity());
      upper[i] = std::nextafter(static_cast<float>(static_cast<double>(bounds[0].upper[i]) + bounds[1].upper[i] + 2e-7), std::numeric_limits<float>::infinity());

    }
    std::vector<float> wrong_linear_sum(linear[0].size());
    for (size_t i = 0; i < wrong_linear_sum.size(); ++i)
      wrong_linear_sum[i] = static_cast<float>(static_cast<double>(linear[0][i]) + linear[1][i]);
    const auto wrong_order = connection::Reference(model, connection::kSpaces[0], wrong_linear_sum, false);
    double wrong_order_max_error = 0;
    size_t distinguished = 0;
    for (size_t i = 0; i < sum.size(); ++i)
      wrong_order_max_error = std::max(wrong_order_max_error, std::abs(static_cast<double>(sum[i]) - wrong_order.exact[i]));
    for (size_t i = 0; i < sum.size(); ++i)
      distinguished += wrong_order.exact[i] < lower[i] || wrong_order.exact[i] > upper[i];
    Check(wrong_order_max_error > 0.1 && distinguished != 0,
          "propagated bounds must distinguish reconstruction/blending order");

    WriteFloats(std::string(argv[3]) + "/" + name + ".composed.device.scalar.f32le", sum);
    WriteFloats(std::string(argv[3]) + "/" + name + ".composed.device.lower.f32le", lower);
    WriteFloats(std::string(argv[3]) + "/" + name + ".composed.device.upper.f32le", upper);

    std::printf("{\"case\":\"%s\",\"native_layer_conversion_max_error\":%.17g,\"maximum_composed_device\":%.9g,\"wrong_linear_blend_max_error\":%.17g}\n",
        name.c_str(), native_scalar_max_error, maximum_original, wrong_order_max_error);
  }
}
