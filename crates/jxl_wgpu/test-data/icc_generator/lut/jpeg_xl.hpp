#pragma once
#include "types.hpp"
#include <jxl/decode.h>
#include <jxl/encode.h>

namespace lut {
constexpr unsigned image_pixels = 17 * 9;

inline Bytes Encode(const Bytes &profile, unsigned colors, bool modular,
                    const std::vector<float> &input) {
  auto enc = JxlEncoderCreate(nullptr);
  Check(enc, "create JPEG XL encoder");
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = 17;
  info.ysize = 9;
  info.bits_per_sample = 32;
  info.exponent_bits_per_sample = 8;
  info.num_color_channels = colors;
  info.num_extra_channels = 1;
  info.alpha_bits = 32;
  info.alpha_exponent_bits = 8;
  info.uses_original_profile = JXL_TRUE;
  Check(JxlEncoderSetBasicInfo(enc, &info) == JXL_ENC_SUCCESS,
        "LUT image basic info");
  Check(JxlEncoderSetICCProfile(enc, profile.data(), profile.size()) ==
            JXL_ENC_SUCCESS,
        "embedded LUT profile");
  auto settings = JxlEncoderFrameSettingsCreate(enc, nullptr);
  Check(settings, "LUT frame settings");
  Check(JxlEncoderSetFrameDistance(settings, 1) == JXL_ENC_SUCCESS,
        "LUT frame distance");
  if (modular)
    Check(JxlEncoderSetFrameLossless(settings, JXL_TRUE) == JXL_ENC_SUCCESS,
          "LUT lossless frame");
  for (const auto &[option, value] :
       std::array<std::pair<JxlEncoderFrameSettingId, int>, 9>{
           {{JXL_ENC_FRAME_SETTING_MODULAR, modular},
            {JXL_ENC_FRAME_SETTING_EFFORT, 3},
            {JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM, 1},
            {JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1},
            {JXL_ENC_FRAME_SETTING_PATCHES, 0},
            {JXL_ENC_FRAME_SETTING_DOTS, 0},
            {JXL_ENC_FRAME_SETTING_NOISE, 0},
            {JXL_ENC_FRAME_SETTING_GABORISH, 0},
            {JXL_ENC_FRAME_SETTING_EPF, 0}}})
    Check(JxlEncoderFrameSettingsSetOption(settings, option, value) ==
              JXL_ENC_SUCCESS,
          "LUT frame option");
  const JxlPixelFormat format{colors + 1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  Check(JxlEncoderAddImageFrame(settings, &format, input.data(),
                                input.size() * sizeof(float)) ==
            JXL_ENC_SUCCESS,
        "LUT image samples");
  JxlEncoderCloseInput(enc);
  Bytes bytes(1 << 20);
  auto next = bytes.data();
  size_t available = bytes.size();
  Check(JxlEncoderProcessOutput(enc, &next, &available) == JXL_ENC_SUCCESS,
        "encode LUT image");
  bytes.resize(bytes.size() - available);
  JxlEncoderDestroy(enc);
  return bytes;
}

inline std::vector<float> Decode(const Bytes &bytes, const Bytes &profile,
                                 unsigned colors) {
  auto dec = JxlDecoderCreate(nullptr);
  Check(dec, "create JPEG XL decoder");
  Check(JxlDecoderSubscribeEvents(dec, JXL_DEC_COLOR_ENCODING |
                                           JXL_DEC_FULL_IMAGE) ==
            JXL_DEC_SUCCESS,
        "LUT decoder events");
  Check(JxlDecoderSetInput(dec, bytes.data(), bytes.size()) == JXL_DEC_SUCCESS,
        "LUT decoder input");
  JxlDecoderCloseInput(dec);
  const JxlPixelFormat format{colors + 1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  std::vector<float> pixels(image_pixels * (colors + 1));
  unsigned images = 0, profiles = 0;
  for (;;) {
    const auto status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_COLOR_ENCODING) {
      for (const auto target :
           {JXL_COLOR_PROFILE_TARGET_ORIGINAL, JXL_COLOR_PROFILE_TARGET_DATA}) {
        size_t size = 0;
        Check(JxlDecoderGetICCProfileSize(dec, target, &size) ==
                  JXL_DEC_SUCCESS,
              "LUT ICC size");
        Bytes actual(size);
        Check(JxlDecoderGetColorAsICCProfile(dec, target, actual.data(),
                                             size) == JXL_DEC_SUCCESS,
              "LUT ICC data");
        Check(actual == profile, "LUT source profile changed");
        ++profiles;
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      Check(JxlDecoderSetImageOutBuffer(dec, &format, pixels.data(),
                                        pixels.size() * sizeof(float)) ==
                JXL_DEC_SUCCESS,
            "LUT image buffer");
    } else if (status == JXL_DEC_FULL_IMAGE)
      ++images;
    else if (status == JXL_DEC_SUCCESS)
      break;
    else
      Check(false, "LUT original decode failed");
  }
  Check(images == 1 && profiles == 2, "LUT image/profile count");
  JxlDecoderDestroy(dec);
  return pixels;
}
} // namespace lut
