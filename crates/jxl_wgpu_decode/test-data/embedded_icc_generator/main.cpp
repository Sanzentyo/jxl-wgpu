#include <jxl/color_encoding.h>
#include <jxl/cms.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <lcms2.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <icc/scalar.hpp>
#include <icc/linear.hpp>

using Bytes = std::vector<unsigned char>;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
Bytes Read(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  Check(in.good(), "input file");
  return Bytes(std::istreambuf_iterator<char>(in), {});
}
void Write(const std::string& path, const Bytes& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  Check(out.good(), "output file");
}
Bytes Gray(const Bytes& source) {
  auto rgb = cmsOpenProfileFromMem(source.data(), source.size());
  Check(rgb != nullptr, "open sampled profile");
  auto* curve = static_cast<cmsToneCurve*>(cmsReadTag(rgb, cmsSigRedTRCTag));
  Check(curve != nullptr, "sampled red curve");
  auto gray = cmsCreateGrayProfile(cmsD50_xyY(), curve);
  Check(gray != nullptr, "create gray");
  cmsSetHeaderRenderingIntent(gray, INTENT_RELATIVE_COLORIMETRIC);
  cmsUInt32Number size = 0;
  Check(cmsSaveProfileToMem(gray, nullptr, &size), "gray size");
  Bytes bytes(size);
  Check(cmsSaveProfileToMem(gray, bytes.data(), &size), "gray bytes");
  cmsCloseProfile(gray);
  cmsCloseProfile(rgb);
  // Stable metadata; content is otherwise exactly the native profile.
  const std::array<unsigned char, 12> date = {7, 234, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0};
  std::copy(date.begin(), date.end(), bytes.begin() + 24);
  std::fill(bytes.begin() + 84, bytes.begin() + 100, 0);
  return bytes;
}
Bytes Encode(const Bytes& profile, bool gray, bool modular, bool original, const std::vector<float>& input) {
  auto* enc = JxlEncoderCreate(nullptr);
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = 17; info.ysize = 9;
  info.bits_per_sample = 32; info.exponent_bits_per_sample = 8;
  info.num_color_channels = gray ? 1 : 3;
  info.num_extra_channels = 1;
  info.alpha_bits = 32; info.alpha_exponent_bits = 8;
  info.uses_original_profile = original;
  Check(JxlEncoderSetBasicInfo(enc, &info) == JXL_ENC_SUCCESS, "basic info");
  Check(JxlEncoderSetICCProfile(enc, profile.data(), profile.size()) == JXL_ENC_SUCCESS, "set ICC");
  auto* settings = JxlEncoderFrameSettingsCreate(enc, nullptr);
  Check(JxlEncoderSetFrameDistance(settings, 1) == JXL_ENC_SUCCESS, "distance");
  if (modular && original)
    Check(JxlEncoderSetFrameLossless(settings, JXL_TRUE) == JXL_ENC_SUCCESS, "lossless");
  for (const auto [option, value] : std::array<std::pair<JxlEncoderFrameSettingId, int>, 9>{{
      {JXL_ENC_FRAME_SETTING_MODULAR, modular}, {JXL_ENC_FRAME_SETTING_EFFORT, 3},
      {JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM, original ? 1 : 0},
      {JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1}, {JXL_ENC_FRAME_SETTING_PATCHES, 0},
      {JXL_ENC_FRAME_SETTING_DOTS, 0}, {JXL_ENC_FRAME_SETTING_NOISE, 0},
      {JXL_ENC_FRAME_SETTING_GABORISH, 0}, {JXL_ENC_FRAME_SETTING_EPF, 0}}})
    Check(JxlEncoderFrameSettingsSetOption(settings, option, value) == JXL_ENC_SUCCESS, "option");
  const unsigned channels = gray ? 2 : 4;
  const JxlPixelFormat format = {channels, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  Check(JxlEncoderAddImageFrame(settings, &format, input.data(), input.size() * 4) == JXL_ENC_SUCCESS, "add frame");
  JxlEncoderCloseInput(enc);
  Bytes bytes(1 << 20);
  auto* next = bytes.data(); size_t available = bytes.size();
  Check(JxlEncoderProcessOutput(enc, &next, &available) == JXL_ENC_SUCCESS, "encode");
  bytes.resize(bytes.size() - available);
  JxlEncoderDestroy(enc);
  return bytes;
}

void WriteHex(const std::string& path, const Bytes& bytes) {
  std::ofstream out(path);
  const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < bytes.size(); ++i) {
    out << digits[bytes[i] >> 4] << digits[bytes[i] & 15];
    if (i % 32 == 31 || i + 1 == bytes.size()) out << '\n';
  }
  Check(out.good(), "hex output");
}

// No profile request: non-XYB data stays in its original device sample domain. The public
// native API rejects an explicit request for these per-channel / sampled profiles even when
// they match the embedded original. Verify the actual default profile and every sample bit.
std::vector<float> DecodeOriginal(const Bytes& bytes, const Bytes& profile, bool gray) {
  auto* dec = JxlDecoderCreate(nullptr);
  Check(JxlDecoderSubscribeEvents(dec, JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS, "events");
  Check(JxlDecoderSetInput(dec, bytes.data(), bytes.size()) == JXL_DEC_SUCCESS, "input");
  JxlDecoderCloseInput(dec);
  const JxlPixelFormat format = {gray ? 2u : 4u, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  std::vector<float> pixels(17 * 9 * (gray ? 2 : 4));
  size_t images = 0;
  for (;;) {
    const auto status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_COLOR_ENCODING) {
      for (const auto target : {JXL_COLOR_PROFILE_TARGET_ORIGINAL, JXL_COLOR_PROFILE_TARGET_DATA}) {
        size_t size = 0;
        Check(JxlDecoderGetICCProfileSize(dec, target, &size) == JXL_DEC_SUCCESS, "ICC size");
        Bytes actual(size);
        Check(JxlDecoderGetColorAsICCProfile(dec, target, actual.data(), size) == JXL_DEC_SUCCESS, "ICC bytes");
        Check(actual == profile, "original ICC changed");
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      Check(JxlDecoderSetImageOutBuffer(dec, &format, pixels.data(), pixels.size() * 4) == JXL_DEC_SUCCESS, "output");
    } else if (status == JXL_DEC_FULL_IMAGE) {
      ++images;
    } else if (status == JXL_DEC_SUCCESS) break;
    else Check(false, "decode failed");
  }
  Check(images == 1, "image count");
  JxlDecoderDestroy(dec);
  return pixels;
}

void WriteFloats(const std::string& path, const std::vector<float>& pixels) {
  Bytes words;
  for (const float sample : pixels) {
    uint32_t word;
    static_assert(sizeof(word) == sizeof(sample));
    std::memcpy(&word, &sample, sizeof(word));
    for (unsigned shift = 0; shift < 32; shift += 8) words.push_back(word >> shift);
  }
  WriteHex(path, words);
}

// The codec first produces original device samples. Little CMS independently evaluates those
// samples; it never supplies a requested-profile decoder label as a substitute for a transform.
double Srgb(double value) {
  const double magnitude = std::abs(value);
  return std::copysign(magnitude <= 0.0031308 ? 12.92 * magnitude : 1.055 * std::pow(magnitude, 1.0 / 2.4) - 0.055, value);
}

std::vector<float> Convert(const Bytes& profile, bool gray, const std::vector<float>& input,
                           cmsHPROFILE target, bool target_gray, const std::string& kind) {
  auto source = cmsOpenProfileFromMem(profile.data(), profile.size());
  Check(source != nullptr && target != nullptr, "open conversion profile");
  if (kind != "other") {
    const size_t channels = gray ? 2 : 4;
    std::vector<float> colors;
    for (size_t i = 0; i < input.size(); ++i) if (i % channels + 1 != channels) colors.push_back(input[i]);
    const auto rgb = connection::Native(source, connection::kSpaces[0], colors, true);
    std::vector<float> output;
    for (size_t i = 0; i < rgb.size(); ++i) {
      output.push_back(kind == "srgb" ? static_cast<float>(Srgb(rgb[i])) : rgb[i]);
      if (i % 3 == 2) output.push_back(input[(i / 3) * channels + channels - 1]);
    }
    cmsCloseProfile(source);
    return output;
  }
  auto transform = cmsCreateTransform(source, gray ? TYPE_GRAY_FLT : TYPE_RGB_FLT,
      target, target_gray ? TYPE_GRAY_DBL : TYPE_RGB_DBL,
      INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
  Check(transform != nullptr, "native ICC transform");
  const size_t source_channels = gray ? 2 : 4;
  const size_t target_channels = target_gray ? 1 : 3;
  std::vector<float> output;
  for (size_t i = 0; i < input.size(); i += source_channels) {
    std::array<double, 3> converted{};
    cmsDoTransform(transform, input.data() + i, converted.data(), 1);
    for (size_t c = 0; c < target_channels; ++c) output.push_back(static_cast<float>(converted[c]));
    output.push_back(input[i + source_channels - 1]);
  }
  cmsDeleteTransform(transform);
  cmsCloseProfile(source);
  return output;
}

std::vector<float> Scalar(const Bytes& profile, bool gray, const std::vector<float>& input,
                          cmsHPROFILE target, const std::string& kind, const std::vector<float>& native) {
  auto source = cmsOpenProfileFromMem(profile.data(), profile.size());
  Check(source != nullptr, "scalar source profile");
  const size_t channels = gray ? 2 : 4;
  std::vector<float> colors;
  for (size_t i = 0; i < input.size(); ++i) if (i % channels + 1 != channels) colors.push_back(input[i]);
  auto reference = kind == "other" ? scalar::Convert(scalar::Profile(source), scalar::Profile(target), colors)
      : connection::Reference(scalar::Profile(source), connection::kSpaces[0], colors, true);
  const size_t target_channels = kind == "other" && !gray ? 1 : 3;
  std::vector<float> output;
  for (size_t pixel = 0; pixel < input.size() / channels; ++pixel) {
    for (size_t c = 0; c < target_channels; ++c) {
      double value = reference.exact[pixel * target_channels + c];
      double lower = reference.native_lower[pixel * target_channels + c];
      double upper = reference.native_upper[pixel * target_channels + c];
      if (kind == "srgb") { value = Srgb(value); lower = Srgb(lower); upper = Srgb(upper); }
      const double observed = native[pixel * (target_channels + 1) + c];
      Check(reference.native_semantics[pixel * target_channels + c] == 0, "unexpected native boundary semantics");
      Check(observed >= lower && observed <= upper, "native precision interval");
      output.push_back(static_cast<float>(value));
    }
    output.push_back(input[pixel * channels + channels - 1]);
  }
  cmsCloseProfile(source);
  return output;
}

int main(int argc, char** argv) {
  Check(argc == 3, "usage: generator gpu_icc_profiles_directory output_directory");
  Check(JxlEncoderVersion() == 12000 && JxlDecoderVersion() == 12000, "libjxl 0.12.0 required");
  Check(cmsGetEncodedCMMversion() == 2190, "Little CMS 2.19 required");
  std::filesystem::create_directories(argv[2]);
  for (const bool gray : {false, true}) {
    const auto profile = gray ? Gray(Read(std::string(argv[1]) + "/sampled.icc"))
                              : Read(std::string(argv[1]) + "/gamma_v4.icc");
    const auto prefix = std::string(argv[2]) + "/" + (gray ? "gray" : "rgb");
    Write(prefix + ".icc", profile);
    std::vector<float> input(17 * 9 * (gray ? 2 : 4));
    for (size_t i = 0; i < input.size(); ++i) input[i] = (8 + ((i * 37) % 101)) / 128.0f;
    WriteFloats(prefix + ".input.f32.hex", input);
    for (const bool modular : {true, false}) for (const bool original : {true, false}) {
      const auto name = prefix + (modular ? "_modular" : "_vardct") + (original ? "_original" : "_xyb");
      const auto bytes = Encode(profile, gray, modular, original, input);
      if (original) {
        const auto decoded = DecodeOriginal(bytes, profile, gray);
        if (modular) Check(std::memcmp(decoded.data(), input.data(), input.size() * 4) == 0, "original sample bits changed");
        WriteFloats(name + ".native.f32.hex", decoded);
        const auto target_bytes = gray ? Read(std::string(argv[1]) + "/gamma_v4.icc")
                                       : Gray(Read(std::string(argv[1]) + "/sampled.icc"));
        auto target = cmsOpenProfileFromMem(target_bytes.data(), target_bytes.size());
        for (const std::string kind : {"linear", "srgb", "other"}) {
          const auto native = Convert(profile, gray, decoded, target, !gray, kind);
          WriteFloats(name + "." + kind + ".f32.hex", native);
          WriteFloats(name + "." + kind + ".scalar.f32.hex", Scalar(profile, gray, decoded, target, kind, native));
        }
        cmsCloseProfile(target);
      }
      WriteHex(name + ".jxl.hex", bytes);
      std::printf("%s: %zu bytes\n", name.c_str(), bytes.size());
    }
  }
}
