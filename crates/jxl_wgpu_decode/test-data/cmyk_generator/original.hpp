#pragma once
#include "../../../jxl_wgpu/test-data/icc_generator/lut/profile.hpp"
#include <jxl/decode.h>

namespace cmyk {
using namespace lut;
constexpr unsigned kPixels = 17 * 9;
constexpr std::array<float, 4> kInk{0.125f, 0.75f, 0.375f, 0.5f};
inline void Dec(JxlDecoderStatus status) {
  Check(status == JXL_DEC_SUCCESS, "CMYK decoder operation");
}
inline std::vector<float> Decode(const Bytes &bytes, const Bytes &profile) {
  auto *decoder = JxlDecoderCreate(nullptr);
  Check(decoder, "create CMYK decoder");
  Dec(JxlDecoderSubscribeEvents(decoder,
                                JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE));
  Dec(JxlDecoderSetRenderSpotcolors(decoder, JXL_FALSE));
  Dec(JxlDecoderSetUnpremultiplyAlpha(decoder, JXL_FALSE));
  Dec(JxlDecoderSetInput(decoder, bytes.data(), bytes.size()));
  JxlDecoderCloseInput(decoder);
  const JxlPixelFormat colors{3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  const JxlPixelFormat scalar{1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  std::vector<float> color(kPixels * 3), output;
  std::array<std::vector<float>, 3> extras;
  for (auto &extra : extras)
    extra.resize(kPixels);
  unsigned profiles = 0;
  for (;;) {
    const auto status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_COLOR_ENCODING) {
      for (auto target :
           {JXL_COLOR_PROFILE_TARGET_ORIGINAL, JXL_COLOR_PROFILE_TARGET_DATA}) {
        size_t size = 0;
        Dec(JxlDecoderGetICCProfileSize(decoder, target, &size));
        Bytes actual(size);
        Dec(JxlDecoderGetColorAsICCProfile(decoder, target, actual.data(),
                                           size));
        Check(actual == profile, "CMYK decoded original profile changed");
        ++profiles;
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      Dec(JxlDecoderSetImageOutBuffer(decoder, &colors, color.data(),
                                      color.size() * 4));
      for (unsigned index = 0; index < 3; ++index)
        Dec(JxlDecoderSetExtraChannelBuffer(
            decoder, &scalar, extras[index].data(), kPixels * 4, index));
    } else if (status == JXL_DEC_FULL_IMAGE) {
      for (unsigned p = 0; p < kPixels; ++p) {
        for (unsigned c = 0; c < 3; ++c)
          output.push_back(color[p * 3 + c]);
        for (const auto &extra : extras)
          output.push_back(extra[p]);
      }
    } else if (status == JXL_DEC_SUCCESS)
      break;
    else
      Check(false, "native CMYK decode failed");
  }
  JxlDecoderDestroy(decoder);
  Check(profiles == 2 && output.size() == 3 * kPixels * 6,
        "CMYK frame/profile count");
  return output;
}

inline Value Ink(double sample, double coverage, unsigned c, bool spots,
                 double error) {
  Value result{sample, error};
  if (spots && c < 3) {
    const double strength = coverage * kInk[3];
    result.x = sample * (1 - strength) + kInk[c] * strength;
    result.radius =
        std::abs(1 - strength) * error + 8 * epsilon * (1 + std::abs(result.x));
  }
  result.x = 1 - result.x;
  result.radius += epsilon * (1 + std::abs(result.x));
  return result;
}
} // namespace cmyk
