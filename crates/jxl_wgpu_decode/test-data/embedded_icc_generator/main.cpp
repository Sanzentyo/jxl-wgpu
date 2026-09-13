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
void CheckOriginal(const Bytes& bytes, const Bytes& profile, bool gray,
                   const std::vector<float>& input) {
  auto* dec = JxlDecoderCreate(nullptr);
  Check(JxlDecoderSubscribeEvents(dec, JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS, "events");
  Check(JxlDecoderSetInput(dec, bytes.data(), bytes.size()) == JXL_DEC_SUCCESS, "input");
  JxlDecoderCloseInput(dec);
  const JxlPixelFormat format = {gray ? 2u : 4u, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  std::vector<float> pixels(input.size());
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
      Check(std::memcmp(pixels.data(), input.data(), input.size() * 4) == 0, "original sample bits changed");
      ++images;
    } else if (status == JXL_DEC_SUCCESS) break;
    else Check(false, "decode failed");
  }
  Check(images == 1, "image count");
  JxlDecoderDestroy(dec);
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
    Bytes words;
    for (const float sample : input) {
      uint32_t word;
      static_assert(sizeof(word) == sizeof(sample));
      std::memcpy(&word, &sample, sizeof(word));
      for (unsigned shift = 0; shift < 32; shift += 8) words.push_back(word >> shift);
    }
    WriteHex(prefix + ".input.f32.hex", words);
    for (const bool modular : {true, false}) for (const bool original : {true, false}) {
      const auto name = prefix + (modular ? "_modular" : "_vardct") + (original ? "_original" : "_xyb");
      const auto bytes = Encode(profile, gray, modular, original, input);
      if (modular && original) CheckOriginal(bytes, profile, gray, input);
      WriteHex(name + ".jxl.hex", bytes);
      std::printf("%s: %zu bytes\n", name.c_str(), bytes.size());
    }
  }
}
