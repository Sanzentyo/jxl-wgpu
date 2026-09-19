#pragma once
// Offline independent ICC equations and Little CMS; no GPU samples or
// production parser.
#include <icc/intents.hpp>
#include <iterator>
#include <lut/profile.hpp>
#include <memory>
#include <variant>

namespace rgb_icc {
using namespace lut;
using Handle = std::unique_ptr<void, decltype(&cmsCloseProfile)>;
constexpr std::array<double, 3> kWhite{.9642, 1, .8249};
constexpr std::array<double, 3> kBlack{.00336, .0034731, .0028646};
// Little CMS 2.19 lcms2.h rounds cmsPERCEPTUAL_BLACK_Z to .00287.
constexpr std::array<double, 3> kNativeBlack{.00336, .0034731, .00287};
constexpr uint32_t kNativeBlackSemantics = 1u << 6;
constexpr scalar::Matrix kIdentity{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};

struct MatrixTrc {};
struct IdentityMpe {};
struct TargetSpec {
  const char *name;
  std::variant<MatrixTrc, IdentityMpe, Recipe> method;
};
const std::array<TargetSpec, 9> kTargets{{
    {"gamma_v4", MatrixTrc{}},
    {"gray", MatrixTrc{}},
    {"intents/rgb_2", MatrixTrc{}},
    {"intents/rgb_8", MatrixTrc{}},
    {"mpe/identity", IdentityMpe{}},
    {"lut/lut8_xyz_3", Recipe{"lut8_xyz_3", Format::Eight, false, 3}},
    {"lut/lut16_lab_3", Recipe{"lut16_lab_3", Format::Sixteen, true, 3}},
    {"lut/ab_lab_1", Recipe{"ab_lab_1", Format::AB, true, 1}},
    {"lut/lut16_v2_xyz_3",
     Recipe{"lut16_v2_xyz_3", Format::Sixteen, false, 3, 2}},
}};

inline Bytes Read(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  Check(bool(file), "read source");
  return Bytes(std::istreambuf_iterator<char>(file), {});
}
inline std::vector<Values> Sources(const std::filesystem::path &path) {
  const auto bytes = Read(path);
  Check(bytes.size() % 48 == 0, "complete PCS source records");
  std::vector<Values> pixels;
  for (size_t i = 0; i < bytes.size(); i += 48) {
    Values pixel;
    for (unsigned c = 0; c < 3; ++c) {
      std::array<double, 2> value;
      for (unsigned part = 0; part < 2; ++part) {
        uint64_t word = 0;
        for (unsigned b = 0; b < 8; ++b)
          word |= uint64_t{bytes[i + c * 16 + part * 8 + b]} << (b * 8);
        std::memcpy(&value[part], &word, sizeof(word));
      }
      Check(std::isfinite(value[0]) && std::isfinite(value[1]) && value[1] >= 0,
            "finite PCS source and radius");
      pixel.push_back({value[0], value[1]});
    }
    pixels.push_back(std::move(pixel));
  }
  return pixels;
}

struct Target {
  std::string name;
  Bytes bytes;
  Handle handle;
  std::unique_ptr<scalar::Profile> scalar;
  std::unique_ptr<Profile> lut;
  unsigned channels;

  Target(const std::filesystem::path &root, const TargetSpec &spec)
      : name(spec.name), bytes(Read(root / (name + ".icc"))),
        handle(cmsOpenProfileFromMem(
                   bytes.data(), static_cast<cmsUInt32Number>(bytes.size())),
               cmsCloseProfile),
        channels(0) {
    Check(bool(handle), "open target profile");
    const auto count = cmsChannelsOfColorSpace(cmsGetColorSpace(handle.get()));
    Check(count > 0 && count <= 15, "ICC target device channel count");
    channels = static_cast<unsigned>(count);
    if (const auto *recipe = std::get_if<Recipe>(&spec.method)) {
      lut = std::make_unique<Profile>(Build(*recipe));
      Check(lut->bytes == bytes,
            "independent LUT recipe must reproduce exact target bytes");
    } else if (std::holds_alternative<MatrixTrc>(spec.method)) {
      scalar = std::make_unique<scalar::Profile>(handle.get());
    }
  }

  std::vector<float> Native(const std::vector<Values> &pixels,
                            unsigned intent) const {
    Handle xyz(cmsCreateXYZProfile(), cmsCloseProfile);
    Check(bool(xyz), "native XYZ endpoint");
    const auto format = cmsFormatterForColorspaceOfProfile(handle.get(), 4, TRUE);
    Check(format != 0, "native target device formatter");
    const std::unique_ptr<void, decltype(&cmsDeleteTransform)> transform(
        cmsCreateTransform(xyz.get(), TYPE_XYZ_DBL, handle.get(),
                           format, intent,
                           cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE),
        cmsDeleteTransform);
    Check(bool(transform), "native XYZ-to-target transform");
    std::vector<double> input;
    for (const auto &pixel : pixels)
      for (auto value : pixel)
        input.push_back(value.x);
    std::vector<float> output(pixels.size() * channels);
    cmsDoTransform(transform.get(), input.data(), output.data(),
                   static_cast<cmsUInt32Number>(pixels.size()));
    // Little CMS ink-space F32 formatters use percentages (CMYK and 5CLR..FCLR).
    // Our profile evaluator and GPU device API both use unit component values.
    if (channels >= 4)
      for (auto &value : output)
        value /= 100;
    // Matrix/TRC device output has the established unit-domain contract. The
    // independent native LUT model separately retains the CMM's unbounded
    // stages.
    if (scalar)
      for (auto &value : output)
        value = std::clamp(value, 0.0f, 1.0f);
    return output;
  }

  Values Evaluate(Values pixel, unsigned intent, bool native) const {
    if (scalar) {
      intents::Endpoint source(connection::kSpaces[0]), target(*scalar);
      source.matrix = kIdentity;
      if (intent == INTENT_ABSOLUTE_COLORIMETRIC) {
        const auto white = intents::MediaWhite(handle.get());
        for (unsigned c = 0; c < 3; ++c)
          source.matrix[c][c] = kWhite[c] / white[c];
      }
      const bool bpc = (intent == 0 || intent == 2) &&
                       cmsGetEncodedICCversion(handle.get()) >= 0x04000000;
      std::vector<float> input;
      for (auto value : pixel)
        input.push_back(static_cast<float>(value.x));
      auto result = intents::Affine(source, target, input, bpc);
      Values output;
      for (unsigned c = 0; c < channels; ++c) {
        const double center = result.exact[c];
        double low = native ? result.native_lower[c] : result.lower[c];
        double high = native ? result.native_upper[c] : result.upper[c];
        if (!native) {
          std::vector<float> corners;
          for (unsigned mask = 0; mask < 8; ++mask)
            for (unsigned axis = 0; axis < 3; ++axis) {
              const double sign = mask & (1u << axis) ? 1 : -1;
              corners.push_back(std::nextafter(
                  static_cast<float>(pixel[axis].x + sign * pixel[axis].radius),
                  static_cast<float>(sign) *
                      std::numeric_limits<float>::infinity()));
            }
          const auto bounds = intents::Affine(source, target, corners, bpc);
          for (unsigned mask = 0; mask < 8; ++mask) {
            low = std::min(low, double{bounds.lower[mask * channels + c]});
            high = std::max(high, double{bounds.upper[mask * channels + c]});
          }
        }
        Check(result.native_semantics[c] == 0,
              "selected scalar target has shared native semantics");
        output.push_back({center, std::max(center - low, high - center)});
      }
      return output;
    }
    std::array<double, 3> scale{1, 1, 1}, offset{};
    const bool v4 = cmsGetEncodedICCversion(handle.get()) >= 0x04000000;
    if ((intent == 0 || intent == 2) && v4) {
      const auto &black = native ? kNativeBlack : kBlack;
      for (unsigned c = 0; c < 3; ++c) {
        scale[c] = (kWhite[c] - black[c]) / kWhite[c];
        offset[c] = black[c];
        if (native)
          pixel[c].semantics |= kNativeBlackSemantics;
      }
    } else if (intent == 3 && lut) {
      const auto white = intents::MediaWhite(handle.get());
      for (unsigned c = 0; c < 3; ++c)
        scale[c] = kWhite[c] / white[c];
    }
    pixel = Diagonal(scale, offset).apply(pixel, native);
    if (lut)
      for (const auto &stage :
           lut->pipelines[(intent == 3 ? 1 : intent) * 2 + 1].stages)
        pixel = stage.apply(pixel, native);
    return pixel;
  }
};
} // namespace
