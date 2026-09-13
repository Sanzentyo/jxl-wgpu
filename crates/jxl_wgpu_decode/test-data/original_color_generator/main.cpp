// Offline libjxl interoperability corpus; never linked by the production codec.
#include <jxl/cms.h>
#include <jxl/color_encoding.h>
#include <jxl/decode.h>
#include <jxl/encode.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {
void Require(bool condition, const char* operation) {
  if (!condition) { std::fprintf(stderr, "%s failed\n", operation); std::exit(1); }
}
void Enc(JxlEncoderStatus status) { Require(status == JXL_ENC_SUCCESS, "encode"); }
void DecImpl(JxlDecoderStatus status, int line) {
  if (status != JXL_DEC_SUCCESS) std::fprintf(stderr, "native decoder status %d at line %d\n", status, line);
  Require(status == JXL_DEC_SUCCESS, "decode");
}
#define Dec(status) DecImpl(status, __LINE__)
struct Mode { const char* name; bool modular, original, ycbcr; };
constexpr Mode modes[] = {
    {"modular_rgb", true, true, false}, {"vardct_rgb", false, true, false},
    {"modular_xyb", true, false, false}, {"vardct_xyb", false, false, false},
    {"modular_ycbcr", true, true, true}, {"vardct_ycbcr", false, true, true},
};
struct Profile {
  const char* name; JxlPrimaries primaries; bool gray;
  JxlWhitePoint white = JXL_WHITE_POINT_D65;
  std::array<double, 2> white_xy = {.3127, .3290};
  std::array<double, 6> rgb_xy = {.64, .33, .30, .60, .15, .06};
};
constexpr Profile profiles[] = {
    {"bt709", JXL_PRIMARIES_SRGB, false}, {"bt2020", JXL_PRIMARIES_2100, false},
    {"p3", JXL_PRIMARIES_P3, false}, {"gray", JXL_PRIMARIES_SRGB, true},
};
struct Transfer { const char* name; JxlTransferFunction value; double gamma = 0; };
constexpr Transfer transfers[] = {
    {"linear", JXL_TRANSFER_FUNCTION_LINEAR}, {"srgb", JXL_TRANSFER_FUNCTION_SRGB},
    {"bt709", JXL_TRANSFER_FUNCTION_709},
};
struct Case {
  std::string name;
  Mode mode;
  Profile profile;
  Transfer transfer;
  bool sequence, floating;
};
std::vector<Case> Cases() {
  std::vector<Case> result;
  for (const auto& mode : modes) for (const auto& profile : profiles) {
    if (mode.ycbcr && profile.gray) continue;
    for (const auto& transfer : transfers) for (bool sequence : {false, true}) {
      std::string name = std::string(mode.name) + "_" + profile.name + "_" + transfer.name;
      result.push_back({name + (sequence ? "_sequence" : "_still"), mode, profile, transfer, sequence, false});
    }
  }
  for (size_t mode = 0; mode < 4; ++mode) for (bool sequence : {false, true}) {
    for (auto selection : {std::array<size_t, 2>{1, 0}, {2, 2}}) {
      const auto& profile = profiles[selection[0]];
      const auto& transfer = transfers[selection[1]];
      std::string name = std::string(modes[mode].name) + "_" + profile.name + "_" + transfer.name;
      result.push_back({name + (sequence ? "_float_sequence" : "_float_still"), modes[mode], profile, transfer, sequence, true});
    }
  }
  return result;
}
std::vector<Case> AnalyticCases() {
  const Profile profiles[] = {
      {"e_bt709", JXL_PRIMARIES_SRGB, false, JXL_WHITE_POINT_E},
      {"dci_p3", JXL_PRIMARIES_P3, false, JXL_WHITE_POINT_DCI},
      {"d50_adobe", JXL_PRIMARIES_CUSTOM, false, JXL_WHITE_POINT_CUSTOM,
          {.34567, .35850}, {.64, .33, .21, .71, .15, .06}},
      {"d65_custom", JXL_PRIMARIES_CUSTOM, false, JXL_WHITE_POINT_D65,
          {.3127, .3290}, {.7347, .2653, .1152, .8264, .1566, .0177}},
      {"e_gray", JXL_PRIMARIES_SRGB, true, JXL_WHITE_POINT_E},
      {"dci_gray", JXL_PRIMARIES_SRGB, true, JXL_WHITE_POINT_DCI},
  };
  const Transfer transfers[] = {
      {"srgb", JXL_TRANSFER_FUNCTION_SRGB}, {"dci", JXL_TRANSFER_FUNCTION_DCI},
      {"gamma22", JXL_TRANSFER_FUNCTION_GAMMA, .4545455},
      {"linear", JXL_TRANSFER_FUNCTION_LINEAR},
      {"gamma2", JXL_TRANSFER_FUNCTION_GAMMA, .5}, {"dci", JXL_TRANSFER_FUNCTION_DCI},
  };
  std::vector<Case> result;
  for (const auto& mode : modes) for (size_t profile = 0; profile < 6; ++profile) {
    if (mode.ycbcr && profile > 1) continue;
    for (bool floating : {false, true}) {
      if (floating && (mode.ycbcr || (profile != 1 && profile != 2 && profile != 4))) continue;
      for (bool sequence : {false, true}) {
        std::string name = std::string("analytic_") + mode.name + "_" + profiles[profile].name + "_" + transfers[profile].name;
        name += floating ? "_float" : "";
        name += sequence ? "_sequence" : "_still";
        result.push_back({name, mode, profiles[profile], transfers[profile], sequence, floating});
      }
    }
  }
  return result;
}
struct Layer {

  int x, y;
  uint32_t width, height, duration, save, source;
  JxlBlendMode blend;
};
std::vector<uint8_t> Encode(const Case& test) {
  constexpr uint32_t width = 37, height = 19;
  const Layer layers[] = {
      {0, 0, width, height, 1, 1, 0, JXL_BLEND_REPLACE},
      {-2, 3, 23, 13, 0, 2, 1, JXL_BLEND_BLEND},
      {26, -2, 23, 13, 2, 1, 2, JXL_BLEND_ADD},
      {0, 0, width, height, 1, 1, 1, JXL_BLEND_MUL},
      {-3, -2, width + 6, height + 4, 0, 2, 1, JXL_BLEND_MULADD},
      {0, 0, width, height, 2, 0, 2, JXL_BLEND_BLEND},
  };
  JxlEncoder* encoder = JxlEncoderCreate(nullptr);
  Require(encoder != nullptr, "encoder allocation");
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = width; info.ysize = height;
  info.bits_per_sample = test.floating ? 32 : test.mode.ycbcr ? 8 : 12;
  info.exponent_bits_per_sample = test.floating ? 8 : 0;
  info.num_color_channels = test.profile.gray ? 1 : 3;
  info.num_extra_channels = 1; info.alpha_bits = 10;
  info.uses_original_profile = test.mode.original;
  info.orientation = JXL_ORIENT_IDENTITY;
  info.have_animation = test.sequence;
  info.animation.tps_numerator = 10; info.animation.tps_denominator = 1;
  info.animation.num_loops = 2;
  Enc(JxlEncoderSetBasicInfo(encoder, &info));
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, test.profile.gray);
color.primaries = test.profile.primaries;
color.white_point = test.profile.white;
std::copy(test.profile.white_xy.begin(), test.profile.white_xy.end(), color.white_point_xy);
std::copy_n(test.profile.rgb_xy.begin(), 2, color.primaries_red_xy);
std::copy_n(test.profile.rgb_xy.begin() + 2, 2, color.primaries_green_xy);
std::copy_n(test.profile.rgb_xy.begin() + 4, 2, color.primaries_blue_xy);
color.gamma = test.transfer.gamma;
  color.transfer_function = test.transfer.value;
  color.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
  Enc(JxlEncoderSetColorEncoding(encoder, &color));
  const uint32_t channels = info.num_color_channels + 1;
  for (uint32_t index = 0; index < (test.sequence ? 6u : 1u); ++index) {
    const Layer& layer = layers[index];
    std::vector<float> pixels(size_t(layer.width) * layer.height * channels);
    for (uint32_t y = 0; y < layer.height; ++y) for (uint32_t x = 0; x < layer.width; ++x) {
      for (uint32_t c = 0; c < channels; ++c) {
        uint32_t code = (x * (113 + c * 57) + y * (61 + c * 97) + (x ^ y) * 29 + index * 173) % 4096;
        float value = float(code) / 4095.f;
        if (c == info.num_color_channels) value = float((x * 31 + y * 71 + index * 103) % 1024) / 1023.f;
        else if (test.floating) value = .05f + value * 1.35f;
        pixels[(size_t(y) * layer.width + x) * channels + c] = value;
      }
    }
    JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(encoder, nullptr);
    Enc(JxlEncoderSetFrameDistance(settings, 1.0f));
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 3));
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, test.mode.modular));
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1));
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PATCHES, 0));
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_DOTS, 0));
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_NOISE, 0));
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_GABORISH, 0));
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EPF, 0));
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, 0));
    if (test.mode.modular && test.mode.original) Enc(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
    // The public API overrides YCbCr to RGB. Emit explicitly named component sources;
    // the Rust fixture assembler changes the frame transform and native decoding verifies it.
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM,
        test.mode.original ? 1 : 0));
    if (!test.mode.modular) Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1));
    JxlFrameHeader header;
    JxlEncoderInitFrameHeader(&header);
    if (test.sequence) {
      header.duration = layer.duration;
      header.layer_info.have_crop = layer.x != 0 || layer.y != 0 || layer.width != width || layer.height != height;
      header.layer_info.crop_x0 = layer.x; header.layer_info.crop_y0 = layer.y;
      header.layer_info.xsize = layer.width; header.layer_info.ysize = layer.height;
      header.layer_info.save_as_reference = layer.save;
      header.layer_info.blend_info.blendmode = layer.blend;
      header.layer_info.blend_info.source = layer.source;
      header.layer_info.blend_info.alpha = 0;
      header.layer_info.blend_info.clamp = JXL_TRUE;
    }
    Enc(JxlEncoderSetFrameHeader(settings, &header));
    JxlBlendInfo alpha = header.layer_info.blend_info;
    alpha.blendmode = JXL_BLEND_REPLACE;
    Enc(JxlEncoderSetExtraChannelBlendInfo(settings, 0, &alpha));
    const JxlPixelFormat format = {channels, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
    Enc(JxlEncoderAddImageFrame(settings, &format, pixels.data(), pixels.size() * sizeof(float)));
  }
  JxlEncoderCloseInput(encoder);
  std::vector<uint8_t> encoded;
  for (;;) {
    std::array<uint8_t, 16384> buffer;
    uint8_t* next = buffer.data(); size_t available = buffer.size();
    auto status = JxlEncoderProcessOutput(encoder, &next, &available);
    if (status != JXL_ENC_SUCCESS && status != JXL_ENC_NEED_MORE_OUTPUT)
      std::fprintf(stderr, "native encoder status %d, error %d\n", status, JxlEncoderGetError(encoder));
    Require(status == JXL_ENC_SUCCESS || status == JXL_ENC_NEED_MORE_OUTPUT, "encoder output");
    encoded.insert(encoded.end(), buffer.data(), next);
    if (status == JXL_ENC_SUCCESS) break;
  }
  JxlEncoderDestroy(encoder);
  return encoded;
}
std::vector<float> DecodeOriginal(const std::vector<uint8_t>& encoded, const Case& test) {
  JxlDecoder* decoder = JxlDecoderCreate(nullptr);
  Require(decoder != nullptr, "decoder allocation");
  if (test.name.rfind("analytic_", 0) == 0) Dec(JxlDecoderSetCms(decoder, *JxlGetDefaultCms()));
  Dec(JxlDecoderSubscribeEvents(decoder, JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE));
  Dec(JxlDecoderSetUnpremultiplyAlpha(decoder, JXL_FALSE));
  Dec(JxlDecoderSetRenderSpotcolors(decoder, JXL_FALSE));
  Dec(JxlDecoderSetKeepOrientation(decoder, JXL_TRUE));
  Dec(JxlDecoderSetInput(decoder, encoded.data(), encoded.size()));
  JxlDecoderCloseInput(decoder);
  std::vector<float> frame, result;
  const JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  for (;;) {
    auto status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_COLOR_ENCODING) {
      JxlColorEncoding original;
      Dec(JxlDecoderGetColorAsEncodedProfile(decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &original));
      Require(original.transfer_function == test.transfer.value && original.white_point == test.profile.white,
          "original color metadata");
      Require(test.profile.gray || original.primaries == test.profile.primaries, "original primaries");
      // libjxl rejects explicit non-D65 Gray requests for non-XYB streams, even
      // when they already use exactly that original profile. Keep its original output.
      if (!(test.profile.gray && test.profile.white != JXL_WHITE_POINT_D65 && test.mode.original))
        Dec(JxlDecoderSetOutputColorProfile(decoder, &original, nullptr, 0));
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t bytes;
      Dec(JxlDecoderImageOutBufferSize(decoder, &format, &bytes));
      Require(bytes % sizeof(float) == 0, "F32 buffer");
      frame.resize(bytes / sizeof(float));
      Dec(JxlDecoderSetImageOutBuffer(decoder, &format, frame.data(), bytes));
    } else if (status == JXL_DEC_FULL_IMAGE) {
      result.insert(result.end(), frame.begin(), frame.end());
    } else if (status == JXL_DEC_SUCCESS) break;
    else Require(false, "native original decode");
  }
  JxlDecoderDestroy(decoder);
  Require(result.size() == 37 * 19 * 4 * (test.sequence ? 4 : 1), "native frame count");
  return result;
}
void Write(const std::filesystem::path& directory, const Case& test) {
  std::fprintf(stderr, "%s\n", test.name.c_str());
  auto encoded = Encode(test);
  const std::string source_name = test.name + (test.mode.ycbcr ? ".rgb-source" : "");
  FILE* stream = std::fopen((directory / (source_name + ".jxl.hex")).c_str(), "w");
  Require(stream != nullptr, "codestream file");
  for (size_t i = 0; i < encoded.size(); ++i) std::fprintf(stream, "%02x%s", encoded[i], (i + 1) % 32 == 0 ? "\n" : "");
  if (encoded.size() % 32) std::fputc('\n', stream);
  Require(std::fclose(stream) == 0, "codestream close");
  if (test.mode.ycbcr) return;
  auto reference = DecodeOriginal(encoded, test);
  stream = std::fopen((directory / (test.name + ".original.f32.hex")).c_str(), "w");
  Require(stream != nullptr, "reference file");
  for (size_t i = 0; i < reference.size(); ++i) {
    uint32_t word; std::memcpy(&word, &reference[i], sizeof(word));
    std::fprintf(stream, "%08x%s", word, (i + 1) % 8 == 0 || i + 1 == reference.size() ? "\n" : " ");
  }
  Require(std::fclose(stream) == 0, "reference close");
}
}  // namespace

int main(int argc, char** argv) {
  Require(argc == 2 || argc == 3, "OUTPUT_DIRECTORY [CASE]");
  Require(JxlEncoderVersion() == 12000 && JxlDecoderVersion() == 12000, "libjxl 0.12.0 version");
  std::filesystem::create_directories(argv[1]);
  size_t count = 0;
const bool analytic = argc == 3 && std::string(argv[2]) == "--analytic";
const auto cases = analytic ? AnalyticCases() : Cases();
for (const auto& test : cases) if (argc == 2 || analytic || test.name == argv[2]) { Write(argv[1], test); ++count; }
  Require(count != 0, "selected cases");
  std::fprintf(stderr, "Generated %zu cases\n", count);
}
