// Offline HDR interoperability corpus. No native codec is linked into production.
#include <jxl/color_encoding.h>
#include <jxl/encode.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "references.hpp"

namespace {
void Enc(JxlEncoderStatus status) { Require(status == JXL_ENC_SUCCESS, "encode"); }
struct Mode { const char* name; bool modular, original; };
constexpr Mode modes[] = {
    {"modular_rgb", true, true}, {"modular_xyb", true, false},
    {"vardct_rgb", false, true}, {"vardct_xyb", false, false},
};
struct Profile { const char* name; JxlPrimaries primaries; bool gray; };
constexpr Profile profiles[] = {
    {"bt2020", JXL_PRIMARIES_2100, false}, {"bt709", JXL_PRIMARIES_SRGB, false},
    {"p3", JXL_PRIMARIES_P3, false}, {"gray", JXL_PRIMARIES_SRGB, true},
};
struct Case {
  Mode mode; Profile profile; JxlTransferFunction transfer;
  uint32_t nits, width, height; bool sequence, floating;
  std::string Name() const {
    return std::string(mode.name) + "_" + profile.name +
        (transfer == JXL_TRANSFER_FUNCTION_PQ ? "_pq_" : "_hlg_") +
        std::to_string(nits) + "_" + std::to_string(width) + "x" + std::to_string(height) +
        (sequence ? "_sequence" : "_still") + (floating ? "_f32" : "_u16");
  }
};
std::vector<Case> Cases() {
  std::vector<Case> result;
  for (const auto& mode : modes) {
    for (auto transfer : {JXL_TRANSFER_FUNCTION_PQ, JXL_TRANSFER_FUNCTION_HLG}) {
      for (uint32_t nits : {100u, 255u, 1000u, 4000u}) {
        result.push_back({mode, profiles[0], transfer, nits, 17, 9, false,
            mode.modular && mode.original && (nits == 100 || nits == 1000)});
      }
      result.push_back({mode, profiles[0], transfer, 255, 37, 19, true, false});
      if (mode.original == mode.modular) {
        for (size_t profile = 1; profile < 4; ++profile)
          result.push_back({mode, profiles[profile], transfer, 1000, 17, 9, false, false});
        result.push_back({mode, profiles[0], transfer, 1000, 257, 17, false, false});
      }
    }
  }
  return result;
}
JxlColorEncoding Color(const Case& test) {
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, test.profile.gray);
  color.primaries = test.profile.primaries;
  color.transfer_function = test.transfer;
  color.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
  return color;
}
struct Layer {
  int x, y;
  uint32_t width, height, duration, save, source;
  JxlBlendMode blend;
};
std::vector<uint8_t> Encode(const Case& test) {
  const auto width = test.width, height = test.height;
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
  info.bits_per_sample = test.floating ? 32 : 16;
  info.exponent_bits_per_sample = test.floating ? 8 : 0;
  info.num_color_channels = test.profile.gray ? 1 : 3;
  info.num_extra_channels = 1; info.alpha_bits = 10;
  info.uses_original_profile = test.mode.original;
  info.intensity_target = float(test.nits);
  info.have_animation = test.sequence;
  info.animation.tps_numerator = 10; info.animation.tps_denominator = 1;
  Enc(JxlEncoderSetBasicInfo(encoder, &info));
  const auto color = Color(test);
  Enc(JxlEncoderSetColorEncoding(encoder, &color));
  const uint32_t channels = info.num_color_channels + 1;
  for (uint32_t index = 0; index < (test.sequence ? 6u : 1u); ++index) {
    const auto& layer = layers[index];
    std::vector<float> pixels(size_t(layer.width) * layer.height * channels);
    for (uint32_t y = 0; y < layer.height; ++y) for (uint32_t x = 0; x < layer.width; ++x) {
      for (uint32_t c = 0; c < channels; ++c) {
        const uint32_t code = (x * (13 + c * 7) + y * (11 + c * 17) + (x ^ y) * 3 + index * 23) % 256;
        float value = float(code) / 256.f;
        if (test.sequence) value = .125f + value * .225f;
        else if (x == 0 && y == 0) value = 0;
        if (c == info.num_color_channels)
          value = float((x * 31 + y * 71 + index * 103) % 1024) / 1023.f;
        pixels[(size_t(y) * layer.width + x) * channels + c] = value;
      }
    }
    auto* settings = JxlEncoderFrameSettingsCreate(encoder, nullptr);
    Enc(JxlEncoderSetFrameDistance(settings, 1.0f));
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 3));
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, test.mode.modular));
    Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1));
    for (auto option : {JXL_ENC_FRAME_SETTING_PATCHES, JXL_ENC_FRAME_SETTING_DOTS,
        JXL_ENC_FRAME_SETTING_NOISE, JXL_ENC_FRAME_SETTING_GABORISH,
        JXL_ENC_FRAME_SETTING_EPF, JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC})
      Enc(JxlEncoderFrameSettingsSetOption(settings, option, 0));
    if (test.mode.modular && test.mode.original) Enc(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
    if (!test.mode.modular)
      Enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1));
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
    auto alpha = header.layer_info.blend_info;
    alpha.blendmode = JXL_BLEND_REPLACE;
    Enc(JxlEncoderSetExtraChannelBlendInfo(settings, 0, &alpha));
    const JxlPixelFormat format = {channels, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
    Enc(JxlEncoderAddImageFrame(settings, &format, pixels.data(), pixels.size() * sizeof(float)));
  }
  JxlEncoderCloseInput(encoder);
  std::vector<uint8_t> encoded;
  for (;;) {
    std::array<uint8_t, 16384> buffer;
    auto* next = buffer.data(); size_t available = buffer.size();
    const auto status = JxlEncoderProcessOutput(encoder, &next, &available);
    Require(status == JXL_ENC_SUCCESS || status == JXL_ENC_NEED_MORE_OUTPUT, "encoder output");
    encoded.insert(encoded.end(), buffer.data(), next);
    if (status == JXL_ENC_SUCCESS) break;
  }
  JxlEncoderDestroy(encoder);
  return encoded;
}
}  // namespace

int main(int argc, char** argv) {
  Require(argc == 2, "OUTPUT_DIRECTORY");
  Require(JxlEncoderVersion() == 12000 && JxlDecoderVersion() == 12000, "libjxl 0.12.0");
  const std::filesystem::path directory = argv[1];
  std::filesystem::create_directories(directory);
  FILE* manifest = std::fopen((directory / "manifest.txt").c_str(), "w");
  Require(manifest != nullptr, "manifest");
  for (const auto& test : Cases()) {
    const auto name = test.Name();
    std::fprintf(stderr, "%s\n", name.c_str());
    const auto encoded = Encode(test);
    WriteBytes(directory / (name + ".jxl.hex"), encoded);
    for (bool linear : {false, true}) {
      // libjxl's XYB blending stage requires the original output profile. Keep
      // the native original reference; tests convert it independently in f64.
      if (linear && test.sequence) continue;
      const auto reference = Decode(encoded, Color(test), test.nits, linear);
      Require(reference.size() == size_t(test.width) * test.height * 4 * (test.sequence ? 4 : 1), "frame count");
      WriteFloats(directory / (name + (linear ? ".linear.f32.hex" : ".original.f32.hex")), reference);
    }
    std::fprintf(manifest, "%s %u %u %d %d %s %s %u %d %d\n", name.c_str(),
        test.width, test.height, test.mode.modular, !test.mode.original, test.profile.name,
        test.transfer == JXL_TRANSFER_FUNCTION_PQ ? "pq" : "hlg", test.nits, test.sequence, test.floating);
  }
  Require(std::fclose(manifest) == 0, "manifest close");
}
