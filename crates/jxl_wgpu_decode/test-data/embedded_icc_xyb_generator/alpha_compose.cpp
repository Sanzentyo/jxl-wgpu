#include "bounds.hpp"
#include "lib/jxl/alpha.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

using Bytes = std::vector<uint8_t>;
void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
Bytes Read(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "input file");
  return Bytes(std::istreambuf_iterator<char>(input), {});
}
std::vector<float> ReadFloats(const std::string &path) {
  const auto bytes = Read(path);
  Check(bytes.size() % 4 == 0, "F32 length");
  std::vector<float> output(bytes.size() / 4);
  for (size_t i = 0; i < output.size(); ++i) {
    uint32_t bits = 0;
    for (unsigned c = 0; c < 4; ++c)
      bits |= static_cast<uint32_t>(bytes[4 * i + c]) << (8 * c);
    std::memcpy(&output[i], &bits, sizeof(bits));
    Check(std::isfinite(output[i]), "finite source");
  }
  return output;
}
void WriteFloats(const std::string &path, const std::vector<float> &pixels) {
  std::ofstream output(path, std::ios::binary);
  for (const float pixel : pixels) {
    uint32_t bits;
    std::memcpy(&bits, &pixel, sizeof(bits));
    for (unsigned shift = 0; shift < 32; shift += 8)
      output.put(bits >> shift);
  }
  Check(output.good(), "output file");
}

// Independent f64 coverage algebra. Native libjxl alpha.cc is linked below to
// check rounding/association separately from reconstruction and ICC curve
// evaluation.
double Blend(double bottom, double top, double ba, double fa, bool associated,
             unsigned mode) {
  if (mode == 0)
    return top;
  if (mode == 1)
    return bottom + top;
  if (mode == 2) {
    if (associated)
      return top + (1 - fa) * bottom;
    const double background_weight = ba * (1 - fa);
    const double coverage = fa + background_weight;
    return coverage > 0 ? (fa * top + background_weight * bottom) / coverage
                        : 0;
  }
  if (mode == 3)
    return bottom + fa * top;
  return bottom * std::clamp(top, 0.0, 1.0);
}
float NativeBlend(float bottom, float top, float ba, float fa, bool associated,
                  unsigned mode) {
  float output = top;
  if (mode == 1)
    output = bottom + top;
  if (mode == 2)
    jxl::PerformAlphaBlending(&bottom, &ba, &top, &fa, &output, 1, associated,
                              true);
  if (mode == 3)
    jxl::PerformAlphaWeightedAdd(&bottom, &top, &fa, &output, 1, true);
  if (mode == 4)
    jxl::PerformMulBlending(&bottom, &top, &output, 1, true);
  return output;
}
std::vector<float> WithAlpha(const std::vector<float> &color,
                             const std::vector<float> &alpha,
                             unsigned channels) {
  Check(color.size() == alpha.size() * channels, "color/alpha geometry");
  std::vector<float> output;
  for (size_t pixel = 0; pixel < alpha.size(); ++pixel) {
    for (unsigned c = 0; c < channels; ++c)
      output.push_back(color[pixel * channels + c]);
    output.push_back(alpha[pixel]);
  }
  return output;
}

int main(int argc, char **argv) {
  Check(argc == 4,
        "usage: oracle layer_directory profiles_directory output_directory");
  Check(cmsGetEncodedCMMversion() == 2190, "Little CMS 2.19 required");
  std::filesystem::create_directories(argv[3]);
  for (bool gray : {false, true})
    for (bool modular : {true, false})
      for (bool associated : {false, true})
        for (unsigned mode = 0; mode <= 4; ++mode)
          for (bool alpha_reference : {false, true}) {
            if (alpha_reference && mode != 2)
              continue;
            const auto name = std::string(gray ? "gray" : "rgb") +
                              (modular ? "_modular" : "_vardct") +
                              (associated ? "_associated" : "_straight") +
                              "_m" + std::to_string(mode) +
                              (alpha_reference ? "_alpha_ref1" : "");
            const auto profile_bytes =
                Read(std::string(argv[2]) + (gray ? "/gray.icc" : "/rgb.icc"));
            std::unique_ptr<void, decltype(&cmsCloseProfile)> profile(
                cmsOpenProfileFromMem(profile_bytes.data(),
                                      profile_bytes.size()),
                cmsCloseProfile);
            Check(profile != nullptr, "original profile");
            const scalar::Profile model(profile.get());
            const unsigned colors = gray ? 1 : 3;
            std::array<std::vector<float>, 2> linear, alpha, native;
            std::array<scalar::References, 2> reference;
            std::array<Bounds, 2> bounds;
            double native_layer_error = 0, native_blend_error = 0;
            size_t native_semantic_components = 0;
            for (unsigned frame = 0; frame < 2; ++frame) {
              const auto input = ReadFloats(
                  std::string(argv[1]) + "/" + name + "_layers_builtin.frame" +
                  std::to_string(frame) + ".linear.f32le");
              Check(input.size() == 153 * (colors + 1), "layer geometry");
              for (size_t pixel = 0; pixel < 153; ++pixel) {
                for (unsigned c = 0; c < 3; ++c)
                  linear[frame].push_back(
                      input[pixel * (colors + 1) + (gray ? 0 : c)]);
                alpha[frame].push_back(input[pixel * (colors + 1) + colors]);
              }
              reference[frame] = connection::Reference(
                  model, connection::kSpaces[0], linear[frame], false);
              bounds[frame] = ReconstructionBounds(model, linear[frame]);
              native[frame] = connection::Native(
                  profile.get(), connection::kSpaces[0], linear[frame], false);
              for (size_t i = 0; i < native[frame].size(); ++i) {
                native_semantic_components +=
                    reference[frame].native_semantics[i] != 0;
                Check(native[frame][i] >= reference[frame].native_lower[i] &&
                          native[frame][i] <= reference[frame].native_upper[i],
                      "native layer precision interval");
                native_layer_error =
                    std::max(native_layer_error,
                             std::abs(static_cast<double>(native[frame][i]) -
                                      reference[frame].exact[i]));
              }
              const auto prefix = std::string(argv[3]) + "/" + name + ".frame" +
                                  std::to_string(frame);
              WriteFloats(
                  prefix + ".device.scalar.f32le",
                  WithAlpha(reference[frame].exact, alpha[frame], colors));
              WriteFloats(prefix + ".device.native.f32le",
                          WithAlpha(native[frame], alpha[frame], colors));
              WriteFloats(prefix + ".device.lower.f32le",
                          WithAlpha(bounds[frame].lower, alpha[frame], colors));
              WriteFloats(prefix + ".device.upper.f32le",
                          WithAlpha(bounds[frame].upper, alpha[frame], colors));
            }
            std::vector<float> composed, native_composed, wrong_linear,
                composed_lower, composed_upper;
            double maximum = 0;
            for (size_t pixel = 0; pixel < 153; ++pixel) {
              // Full-frame Replace carries implicit source 0 for this extra.
              // Only the explicit alpha Blend variant selects saved
              // reference 1.
              const double ba = alpha_reference ? alpha[0][pixel] : 0.0;
              const double fa = alpha[1][pixel];

              for (unsigned c = 0; c < colors; ++c) {
                const size_t index = pixel * colors + c;
                const double exact =
                    Blend(reference[0].exact[index], reference[1].exact[index],
                          ba, fa, associated, mode);
                const float native_operation = NativeBlend(
                    reference[0].exact[index], reference[1].exact[index], ba,
                    fa, associated, mode);
                native_blend_error = std::max(
                    native_blend_error, std::abs(exact - native_operation));
                Check(std::abs(exact - native_operation) <= 2e-7,
                      "native blend rounding");
                const float native_result =
                    NativeBlend(native[0][index], native[1][index], ba, fa,
                                associated, mode);
                const double lower = Blend(
                    std::clamp(reference[0].native_lower[index], 0.0f, 1.0f),
                    std::clamp(reference[1].native_lower[index], 0.0f, 1.0f),
                    ba, fa, associated, mode);
                const double upper = Blend(
                    std::clamp(reference[0].native_upper[index], 0.0f, 1.0f),
                    std::clamp(reference[1].native_upper[index], 0.0f, 1.0f),
                    ba, fa, associated, mode);
                Check(native_result >= lower - 2e-7 &&
                          native_result <= upper + 2e-7,
                      "composed native precision interval");
                composed.push_back(static_cast<float>(exact));
                const double composed_lo =
                    Blend(std::clamp(bounds[0].lower[index], 0.0f, 1.0f),
                          std::clamp(bounds[1].lower[index], 0.0f, 1.0f), ba,
                          fa, associated, mode);
                const double composed_hi =
                    Blend(std::clamp(bounds[0].upper[index], 0.0f, 1.0f),
                          std::clamp(bounds[1].upper[index], 0.0f, 1.0f), ba,
                          fa, associated, mode);
                Check(exact >= composed_lo - 2e-7 &&
                          exact <= composed_hi + 2e-7,
                      "scalar inside propagated interval");
                composed_lower.push_back(
                    std::nextafter(static_cast<float>(composed_lo - 2e-7),
                                   -std::numeric_limits<float>::infinity()));
                composed_upper.push_back(
                    std::nextafter(static_cast<float>(composed_hi + 2e-7),
                                   std::numeric_limits<float>::infinity()));

                native_composed.push_back(native_result);
                maximum = std::max(maximum, exact);
              }
              const float merged_alpha =
                  mode == 2 ? static_cast<float>(fa + ba * (1 - fa))
                            : static_cast<float>(fa);
              float native_alpha = static_cast<float>(fa);
              if (mode == 2) {
                const float b = static_cast<float>(ba),
                            f = static_cast<float>(fa);
                jxl::PerformAlphaBlending(&b, &b, &f, &f, &native_alpha, 1,
                                          associated, true);
              }
              Check(native_alpha == merged_alpha, "native merged alpha");
              composed.push_back(merged_alpha);
              composed_lower.push_back(merged_alpha);
              composed_upper.push_back(merged_alpha);
              native_composed.push_back(native_alpha);
              for (unsigned c = 0; c < 3; ++c)
                wrong_linear.push_back(static_cast<float>(
                    Blend(linear[0][pixel * 3 + c], linear[1][pixel * 3 + c],
                          ba, fa, associated, mode)));
            }
            const auto wrong_order = connection::Reference(
                model, connection::kSpaces[0], wrong_linear, false);
            double wrong_order_error = 0;
            for (size_t pixel = 0; pixel < 153; ++pixel)
              for (unsigned c = 0; c < colors; ++c)
                wrong_order_error =
                    std::max(wrong_order_error,
                             std::abs(static_cast<double>(
                                          composed[pixel * (colors + 1) + c]) -
                                      wrong_order.exact[pixel * colors + c]));
            const auto prefix =
                std::string(argv[3]) + "/" + name + ".composed.device";
            WriteFloats(prefix + ".scalar.f32le", composed);
            WriteFloats(prefix + ".native.f32le", native_composed);
            WriteFloats(prefix + ".lower.f32le", composed_lower);
            WriteFloats(prefix + ".upper.f32le", composed_upper);

            bool excludes_wrong_order = false;
            for (size_t pixel = 0; pixel < 153; ++pixel) {
              for (unsigned c = 0; c < colors; ++c) {
                const size_t index = pixel * (colors + 1) + c;
                const float wrong = wrong_order.exact[pixel * colors + c];
                excludes_wrong_order |= wrong < composed_lower[index] ||
                                        wrong > composed_upper[index];
              }
            }
            if (mode == 1 || mode == 3 ||
                (mode == 2 && (associated || alpha_reference))) {
              Check(excludes_wrong_order,
                    "bounds distinguish original-device blending");
            }

            std::printf(
                "{\"case\":\"%s\",\"native_layer_error\":%.17g,\"native_blend_"
                "error\":%.17g,\"native_semantic_components\":%zu,\"maximum_"
                "composed_device\":%.17g,\"wrong_order_error\":%.17g,"
                "\"excludes_"
                "wrong_order\":%s}\n",
                name.c_str(), native_layer_error, native_blend_error,
                native_semantic_components, maximum, wrong_order_error,
                excludes_wrong_order ? "true" : "false");
          }
}
