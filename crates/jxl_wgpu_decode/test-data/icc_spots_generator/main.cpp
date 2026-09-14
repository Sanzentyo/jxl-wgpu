#include "reference.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <jxl/cms.h>
#include <jxl/color_encoding.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Bytes = std::vector<uint8_t>;
constexpr uint32_t kWidth = 17, kHeight = 9, kPixels = kWidth * kHeight;
constexpr std::array<JxlExtraChannelType, 9> kTypes{
    JXL_CHANNEL_DEPTH,      JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_ALPHA,
    JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_THERMAL,
    JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_ALPHA};
constexpr std::array<unsigned, 9> kBits{32, 1, 7, 12, 32, 8, 6, 10, 15};
constexpr std::array<std::array<float, 4>, 9> kInks{{{},
                                                     {0.75f, 0.125f, 0.25f, 0},
                                                     {},
                                                     {0.25f, 0.5f, 0.75f, 0.5f},
                                                     {1, 0.125f, 0.5f, 1},
                                                     {},
                                                     {0.125f, 1, 0.375f, 0.5f},
                                                     {0.5f, 0.25f, 1, 1},
                                                     {}}};
void Require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Enc(JxlEncoderStatus status) {
  Require(status == JXL_ENC_SUCCESS, "encoder call");
}
void Dec(JxlDecoderStatus status) {
  Require(status == JXL_DEC_SUCCESS, "decoder call");
}
Bytes Read(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  Require(in.good(), "read profile");
  return Bytes(std::istreambuf_iterator<char>(in), {});
}
void Write(const std::filesystem::path &path, const Bytes &bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  Require(out.good(), "write bytes");
}
void Word(Bytes &bytes, uint64_t word, unsigned size) {
  for (unsigned i = 0; i < size; ++i)
    bytes.push_back(word >> (8 * i));
}
void WriteFloats(const std::filesystem::path &path,
                 const std::vector<float> &values) {
  Bytes bytes;
  for (float value : values) {
    uint32_t word;
    std::memcpy(&word, &value, 4);
    Word(bytes, word, 4);
  }
  Write(path, bytes);
}
struct Case {
  bool icc, gray, modular, original, sequence;
};
std::string Name(const Case &test) {
  return std::string(test.icc ? "icc_" : "enum_") +
         (test.gray ? "gray_" : "rgb_") +
         (test.modular ? "modular_" : "vardct_") +
         (test.original ? "original_" : "xyb_") +
         (test.sequence ? "sequence" : "still");
}
JxlColorEncoding Encoding(const Case &test, bool linear) {
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, test.gray);
  if (!test.gray)
    color.primaries = JXL_PRIMARIES_P3;
  if (linear)
    color.transfer_function = JXL_TRANSFER_FUNCTION_LINEAR;
  color.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
  return color;
}
float Sample(unsigned frame, unsigned pixel, unsigned channel) {
  if (kBits[channel] == 32)
    return float((pixel * (11 + channel * 3) + frame * 7) % 33) / 32;
  const unsigned maximum = (1u << kBits[channel]) - 1;
  return float((pixel * (11 + channel * 3) + frame * 7) % (maximum + 1)) /
         maximum;
}
Bytes Encode(const Case &test, const Bytes &profile) {
  auto *encoder = JxlEncoderCreate(nullptr);
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = kWidth;
  info.ysize = kHeight;
  info.bits_per_sample = 32;
  info.exponent_bits_per_sample = 8;
  info.num_color_channels = test.gray ? 1 : 3;
  info.num_extra_channels = kTypes.size();
  info.alpha_bits = 7;
  info.alpha_premultiplied = test.sequence;
  info.uses_original_profile = test.original;
  info.orientation =
      test.gray ? JXL_ORIENT_ROTATE_90_CCW : JXL_ORIENT_ROTATE_90_CW;
  info.have_animation = test.sequence;
  info.animation.tps_numerator = 10;
  info.animation.tps_denominator = 1;
  Enc(JxlEncoderSetBasicInfo(encoder, &info));
  if (test.icc)
    Enc(JxlEncoderSetICCProfile(encoder, profile.data(), profile.size()));
  else {
    const auto color = Encoding(test, false);
    Enc(JxlEncoderSetColorEncoding(encoder, &color));
  }
  for (unsigned index = 0; index < kTypes.size(); ++index) {
    JxlExtraChannelInfo extra;
    JxlEncoderInitExtraChannelInfo(kTypes[index], &extra);
    extra.bits_per_sample = kBits[index];
    extra.exponent_bits_per_sample = kBits[index] == 32 ? 8 : 0;
    extra.alpha_premultiplied = test.sequence && index == 2;
    std::copy(kInks[index].begin(), kInks[index].end(), extra.spot_color);
    Enc(JxlEncoderSetExtraChannelInfo(encoder, index, &extra));
  }
  const unsigned colors = info.num_color_channels;
  for (unsigned frame = 0; frame < (test.sequence ? 3u : 1u); ++frame) {
    auto *settings = JxlEncoderFrameSettingsCreate(encoder, nullptr);
    Enc(JxlEncoderSetFrameDistance(settings, 1));
    if (test.modular && test.original)
      Enc(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
    for (const auto [option, value] :
         std::array<std::pair<JxlEncoderFrameSettingId, int>, 10>{
             {{JXL_ENC_FRAME_SETTING_MODULAR, test.modular},
              {JXL_ENC_FRAME_SETTING_EFFORT, 3},
              {JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM, test.original ? 1 : 0},
              {JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1},
              {JXL_ENC_FRAME_SETTING_PATCHES, 0},
              {JXL_ENC_FRAME_SETTING_DOTS, 0},
              {JXL_ENC_FRAME_SETTING_NOISE, 0},
              {JXL_ENC_FRAME_SETTING_GABORISH, 0},
              {JXL_ENC_FRAME_SETTING_EPF, 0},
              {JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, !test.modular}}})
      Enc(JxlEncoderFrameSettingsSetOption(settings, option, value));
    JxlFrameHeader header;
    JxlEncoderInitFrameHeader(&header);
    if (test.sequence) {
      header.duration = frame + 1;
      header.layer_info.save_as_reference = frame < 2 ? frame + 1 : 0;
      header.layer_info.blend_info.blendmode =
          frame == 0 ? JXL_BLEND_REPLACE : JXL_BLEND_BLEND;
      header.layer_info.blend_info.source = frame;
      header.layer_info.blend_info.alpha = 2;
      header.layer_info.blend_info.clamp = JXL_TRUE;
    }
    Enc(JxlEncoderSetFrameHeader(settings, &header));
    for (unsigned index = 0; index < kTypes.size(); ++index) {
      JxlBlendInfo blend = header.layer_info.blend_info;
      blend.blendmode =
          index == 2 || frame == 0 ? JXL_BLEND_REPLACE : JXL_BLEND_BLEND;
      Enc(JxlEncoderSetExtraChannelBlendInfo(settings, index, &blend));
      Enc(JxlEncoderSetExtraChannelDistance(settings, index, 0));
    }
    std::vector<float> input(kPixels * (colors + 1));
    for (unsigned pixel = 0; pixel < kPixels; ++pixel) {
      const float alpha = Sample(frame, pixel, 2);
      for (unsigned c = 0; c < colors; ++c) {
        float value =
            0.125f + float((pixel * (13 + c * 3) + frame * 19) % 25) / 32;
        if (test.sequence)
          value *= alpha;
        input[pixel * (colors + 1) + c] = value;
      }
      input[pixel * (colors + 1) + colors] = alpha;
    }
    const JxlPixelFormat color_format{colors + 1, JXL_TYPE_FLOAT,
                                      JXL_NATIVE_ENDIAN, 0};
    Enc(JxlEncoderAddImageFrame(settings, &color_format, input.data(),
                                input.size() * 4));
    const JxlPixelFormat scalar_format{1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
    for (unsigned index = 0; index < kTypes.size(); ++index) {
      std::vector<float> samples(kPixels);
      for (unsigned p = 0; p < kPixels; ++p)
        samples[p] = Sample(frame, p, index);
      Enc(JxlEncoderSetExtraChannelBuffer(
          settings, &scalar_format, samples.data(), samples.size() * 4, index));
    }
  }
  JxlEncoderCloseInput(encoder);
  Bytes bytes;
  for (;;) {
    std::array<uint8_t, 16384> buffer;
    auto *next = buffer.data();
    size_t available = buffer.size();
    auto status = JxlEncoderProcessOutput(encoder, &next, &available);
    Require(status == JXL_ENC_SUCCESS || status == JXL_ENC_NEED_MORE_OUTPUT,
            "encode output");
    bytes.insert(bytes.end(), buffer.data(), next);
    if (status == JXL_ENC_SUCCESS)
      break;
  }
  JxlEncoderDestroy(encoder);
  return bytes;
}

std::vector<float> Decode(const Case &test, const Bytes &encoded,
                          const Bytes &profile, bool spots,
                          std::vector<float> *uncoalesced_linear = nullptr) {
  auto *decoder = JxlDecoderCreate(nullptr);
  // The native XYB blender cannot insert a general ICC inverse. Obtain
  // uncoalesced linear pixels, then independently convert and compose the
  // device samples.
  const bool compose = test.icc && !test.original && test.sequence;
  if (compose)
    Dec(JxlDecoderSetCoalescing(decoder, JXL_FALSE));
  Dec(JxlDecoderSetCms(decoder, *JxlGetDefaultCms()));
  Dec(JxlDecoderSubscribeEvents(decoder,
                                JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE));
  Dec(JxlDecoderSetUnpremultiplyAlpha(decoder, JXL_FALSE));
  Dec(JxlDecoderSetRenderSpotcolors(decoder, spots && !compose));
  Dec(JxlDecoderSetKeepOrientation(decoder, JXL_TRUE));
  Dec(JxlDecoderSetInput(decoder, encoded.data(), encoded.size()));
  JxlDecoderCloseInput(decoder);
  std::vector<float> output, frame(kPixels * 4);
  std::array<std::vector<float>, 9> extras;
  for (auto &extra : extras)
    extra.resize(kPixels);
  const JxlPixelFormat format{4, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  const JxlPixelFormat scalar{1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  for (;;) {
    auto status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_COLOR_ENCODING) {
      const bool linear = (!test.original && !test.sequence) || compose;
      // Non-XYB native output already has the exact original profile. libjxl
      // rejects an explicit request for some per-channel/sampled profiles even
      // when identical.
      if (test.icc && test.original) {
        size_t size = 0;
        Dec(JxlDecoderGetICCProfileSize(decoder, JXL_COLOR_PROFILE_TARGET_DATA,
                                        &size));
        Bytes actual(size);
        Dec(JxlDecoderGetColorAsICCProfile(
            decoder, JXL_COLOR_PROFILE_TARGET_DATA, actual.data(), size));
        Require(actual == profile, "native original device profile");
      } else if (test.icc && !linear)
        Dec(JxlDecoderSetOutputColorProfile(decoder, nullptr, profile.data(),
                                            profile.size()));
      else {
        auto color = Encoding(test, linear);
        if (test.icc)
          JxlColorEncodingSetToLinearSRGB(&color, test.gray);
        Dec(JxlDecoderSetOutputColorProfile(decoder, &color, nullptr, 0));
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      Dec(JxlDecoderSetImageOutBuffer(decoder, &format, frame.data(),
                                      frame.size() * 4));
      for (unsigned index = 0; index < extras.size(); ++index)
        Dec(JxlDecoderSetExtraChannelBuffer(
            decoder, &scalar, extras[index].data(), kPixels * 4, index));
    } else if (status == JXL_DEC_FULL_IMAGE) {
      output.insert(output.end(), frame.begin(), frame.end());
      for (const auto &extra : extras)
        output.insert(output.end(), extra.begin(), extra.end());
    } else if (status == JXL_DEC_SUCCESS)
      break;
    else {
      std::fprintf(stderr, "decode status %d, spots=%d, compose=%d\n", status,
                   spots, compose);
      Require(false, "decode source");
    }
  }
  JxlDecoderDestroy(decoder);
  Require(output.size() == kPixels * 13 * (test.sequence ? 3 : 1),
          "source frame count");
  if (compose) {
    constexpr size_t stride = kPixels * 13;
    if (uncoalesced_linear)
      *uncoalesced_linear = output;
    auto source_profile = cmsOpenProfileFromMem(profile.data(), profile.size());
    Require(source_profile != nullptr, "XYB original ICC");
    for (unsigned frame = 0; frame < 3; ++frame) {
      float *top = output.data() + frame * stride;
      std::vector<float> linear;
      for (unsigned pixel = 0; pixel < kPixels; ++pixel)
        linear.insert(linear.end(), top + pixel * 4, top + pixel * 4 + 3);
      const auto converted = connection::Native(
          source_profile, connection::kSpaces[0], linear, false);
      for (unsigned pixel = 0; pixel < kPixels; ++pixel)
        for (unsigned c = 0; c < 3; ++c)
          top[pixel * 4 + c] =
              converted[pixel * (test.gray ? 1 : 3) + (test.gray ? 0 : c)];
    }
    cmsCloseProfile(source_profile);
    for (unsigned frame = 1; frame < 3; ++frame) {
      float *top = output.data() + frame * stride;
      const float *base = top - stride;
      for (unsigned pixel = 0; pixel < kPixels; ++pixel) {
        const double alpha = top[kPixels * 6 + pixel];
        for (unsigned c = 0; c < 3; ++c)
          top[pixel * 4 + c] = float(double(top[pixel * 4 + c]) +
                                     base[pixel * 4 + c] * (1 - alpha));
        for (unsigned c = 0; c < 9; ++c)
          if (c != 2) {
            const size_t offset = kPixels * (4 + c) + pixel;
            top[offset] =
                float(double(top[offset]) + base[offset] * (1 - alpha));
          }
      }
    }
    if (spots)
      for (unsigned frame = 0; frame < 3; ++frame) {
        float *top = output.data() + frame * stride;
        for (unsigned pixel = 0; pixel < kPixels; ++pixel)
          for (unsigned c = 0; c < 3; ++c) {
            double value = top[pixel * 4 + c];
            for (unsigned index = 0; index < 9; ++index)
              if (kTypes[index] == JXL_CHANNEL_SPOT_COLOR) {
                const double mix = double(kInks[index][3]) *
                                   top[kPixels * (4 + index) + pixel];
                value =
                    mix * kInks[index][test.gray ? 0 : c] + (1 - mix) * value;
              }
            top[pixel * 4 + c] = float(value);
          }
      }
  }
  return output;
}
void References(const std::filesystem::path &output, const Case &test,
                const std::array<cmsHPROFILE, 2> &profiles,
                const std::vector<float> &source,
                const std::vector<float> &linear,
                const std::vector<float> &rendered, size_t &validated) {
  using reference::Color;
  using reference::Range;
  const std::array<scalar::Profile, 2> descriptions{
      scalar::Profile(profiles[0]), scalar::Profile(profiles[1])};
  const bool device = test.icc && (test.original || test.sequence);
  const unsigned colors = device && test.gray ? 1 : 3;
  const unsigned frames = test.sequence ? 3 : 1;
  constexpr size_t stride = kPixels * 13;
  const auto &space = connection::kSpaces[test.icc || test.gray ? 0 : 2];
  const reference::Source encoding{device ? &descriptions[test.gray] : nullptr,
                                   space, !test.original && !test.sequence};
  std::vector<Color> inputs(kPixels * frames);
  for (unsigned frame = 0; frame < frames; ++frame)
    for (unsigned pixel = 0; pixel < kPixels; ++pixel) {
      auto &input = inputs[frame * kPixels + pixel];
      for (unsigned c = 0; c < 3; ++c) {
        const double value =
            (linear.empty() ? source : linear)[frame * stride + pixel * 4 + c];
        const double tolerance =
            test.original && test.modular ? 1e-5 : 1.0 / 1024;
        input[c] = Range::Around(value, tolerance * (1 + std::abs(value)));
      }
      if (!linear.empty()) {
        const reference::Connection original(
            {nullptr, connection::kSpaces[0], true}, &descriptions[test.gray]);
        input = original.Convert(input);
        if (frame != 0) {
          const double alpha = source[frame * stride + kPixels * 6 + pixel];
          const auto remaining = Range::Around(1 - alpha, 2e-6);
          for (unsigned c = 0; c < colors; ++c)
            input[c] =
                input[c] + inputs[(frame - 1) * kPixels + pixel][c] * remaining;
        }
      }
    }
  for (bool spots : {false, true}) {
    auto bounds = inputs;
    std::vector<float> values;
    for (unsigned frame = 0; frame < frames; ++frame)
      for (unsigned pixel = 0; pixel < kPixels; ++pixel) {
        for (unsigned c = 0; c < colors; ++c) {
          double value = source[frame * stride + pixel * 4 + c];
          auto &range = bounds[frame * kPixels + pixel][c];
          if (spots)
            for (unsigned ink = 0; ink < 9; ++ink)
              if (kTypes[ink] == JXL_CHANNEL_SPOT_COLOR) {
                const double sample =
                    source[frame * stride + kPixels * (4 + ink) + pixel];
                const double mix = sample * kInks[ink][3];
                const auto amount =
                    Range::Around(mix, std::abs(kInks[ink][3]) * 2e-6 *
                                           (1 + std::abs(sample)));
                range = amount * Range{kInks[ink][c], kInks[ink][c]} +
                        Range{1 - amount.high, 1 - amount.low} * range;
                const double roundoff =
                    2e-7 *
                    (1 + std::max(std::abs(range.low), std::abs(range.high)));
                range.low -= roundoff;
                range.high += roundoff;
                value = mix * kInks[ink][c] + (1 - mix) * value;
              }
          values.push_back(float(value));
          // Gray CMS can discard the other two rendered channels. The first
          // channel and every RGB channel still provide native stage/order
          // evidence.
          if (spots && (!test.gray || c == 0)) {
            const double native = rendered[frame * stride + pixel * 4 + c];
            Require(std::isfinite(native) && native >= range.low &&
                        native <= range.high,
                    "native spot stage interval");
          }
        }
      }
    for (unsigned target = 0; target < 3; ++target) {
      const auto *target_description =
          target < 2 ? &descriptions[target] : nullptr;
      const bool identity = device && unsigned(test.gray) == target;
      const reference::Connection transform(encoding, target_description,
                                            identity);
      const unsigned channels = target == 1 ? 1 : 3;
      Bytes bytes;
      for (const auto &input : bounds) {
        const auto result = transform.Convert(input);
        for (unsigned c = 0; c < channels; ++c)
          for (double value : {result[c].low, result[c].high}) {
            Require(std::isfinite(value), "finite primary interval");
            uint64_t word;
            std::memcpy(&word, &value, 8);
            Word(bytes, word, 8);
          }
      }
      const auto prefix =
          Name(test) + (spots ? ".render." : ".preserve.") +
          std::array<const char *, 3>{"rgb", "gray", "linear"}[target];
      Write(output / (prefix + ".bounds"), bytes);
      std::fprintf(stderr, "  %s\n", prefix.c_str());
      const auto native = reference::Native(
          encoding, profiles[test.gray],
          target < 2 ? profiles[target] : nullptr, values, identity, validated);
      WriteFloats(output / (prefix + ".native.f32"), native);
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  Require(argc == 3, "EMBEDDED_ICC_DIRECTORY OUTPUT_DIRECTORY");
  Require(JxlEncoderVersion() == 12000 && JxlDecoderVersion() == 12000,
          "libjxl 0.12.0");
  Require(cmsGetEncodedCMMversion() == 2190, "Little CMS 2.19");
  const std::filesystem::path profiles(argv[1]), output(argv[2]);
  std::filesystem::create_directories(output);
  std::ofstream manifest(output / "manifest.tsv");
  manifest << "name\ticc\tgray\tmodular\toriginal\tsequence\n";
  const std::array<Bytes, 2> profile_bytes{Read(profiles / "rgb.icc"),
                                           Read(profiles / "gray.icc")};
  const std::array<cmsHPROFILE, 2> native_profiles{
      cmsOpenProfileFromMem(profile_bytes[0].data(), profile_bytes[0].size()),
      cmsOpenProfileFromMem(profile_bytes[1].data(), profile_bytes[1].size())};
  Require(native_profiles[0] && native_profiles[1],
          "native reference profiles");
  size_t validated = 0;
  for (bool icc : {false, true})
    for (bool gray : {false, true})
      for (bool modular : {false, true})
        for (bool original : {false, true})
          for (bool sequence : {false, true}) {
            const Case test{icc, gray, modular, original, sequence};
            const auto name = Name(test);
            std::fprintf(stderr, "%s\n", name.c_str());
            const auto &profile = profile_bytes[gray];
            const auto encoded = Encode(test, profile);
            Write(output / (name + ".jxl"), encoded);
            std::vector<float> linear;
            const auto source = Decode(test, encoded, profile, false, &linear);
            const auto rendered = Decode(test, encoded, profile, true);
            WriteFloats(output / (name + ".source.f32"), source);
            WriteFloats(output / (name + ".rendered.f32"), rendered);
            if (!linear.empty())
              WriteFloats(output / (name + ".uncoalesced-linear.f32"), linear);
            References(output, test, native_profiles, source, linear, rendered,
                       validated);
            manifest << name << '\t' << icc << '\t' << gray << '\t' << modular
                     << '\t' << original << '\t' << sequence << '\n';
          }
  Require(manifest.good(), "manifest");
  for (auto profile : native_profiles)
    cmsCloseProfile(profile);
  std::fprintf(stderr, "Validated %zu native CMS components\n", validated);
}
