#include <jxl/cms.h>
#include <jxl/decode.h>
#include <jxl/encode.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Bytes = std::vector<uint8_t>;
void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Enc(JxlEncoderStatus status) {
  Check(status == JXL_ENC_SUCCESS, "encode API");
}
Bytes Read(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "input file");
  return Bytes(std::istreambuf_iterator<char>(input), {});
}
void Write(const std::string &path, const Bytes &bytes) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  Check(output.good(), "output file");
}

float Alpha(size_t pixel, unsigned frame) {
  constexpr std::array<float, 6> alpha = {0.0f, 0.125f, 0.25f,
                                          0.5f, 0.75f,  1.0f};
  return alpha[(pixel * 5 + frame * 2) % alpha.size()];
}

Bytes Encode(const Bytes &profile, bool gray, bool modular, bool associated,
             unsigned mode, bool alpha_reference) {
  std::unique_ptr<JxlEncoder, decltype(&JxlEncoderDestroy)> owned(
      JxlEncoderCreate(nullptr), JxlEncoderDestroy);
  auto *encoder = owned.get();
  Check(encoder != nullptr, "encoder");
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = 17;
  info.ysize = 9;
  info.bits_per_sample = 32;
  info.exponent_bits_per_sample = 8;
  info.num_color_channels = gray ? 1 : 3;
  info.num_extra_channels = 1;
  info.alpha_bits = 32;
  info.alpha_exponent_bits = 8;
  info.alpha_premultiplied = associated;
  info.uses_original_profile = JXL_FALSE;
  info.have_animation = JXL_TRUE;
  info.animation.tps_numerator = 100;
  info.animation.tps_denominator = 1;
  Enc(JxlEncoderSetBasicInfo(encoder, &info));
  Enc(JxlEncoderSetICCProfile(encoder, profile.data(), profile.size()));
  for (unsigned frame = 0; frame < 2; ++frame) {
    auto *settings = JxlEncoderFrameSettingsCreate(encoder, nullptr);
    Enc(JxlEncoderSetFrameDistance(settings, 1));
    for (const auto [option, value] :
         std::array<std::pair<JxlEncoderFrameSettingId, int>, 9>{
             {{JXL_ENC_FRAME_SETTING_MODULAR, modular},
              {JXL_ENC_FRAME_SETTING_EFFORT, 3},
              {JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM, 0},
              {JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1},
              {JXL_ENC_FRAME_SETTING_PATCHES, 0},
              {JXL_ENC_FRAME_SETTING_DOTS, 0},
              {JXL_ENC_FRAME_SETTING_NOISE, 0},
              {JXL_ENC_FRAME_SETTING_GABORISH, 0},
              {JXL_ENC_FRAME_SETTING_EPF, 0}}})
      Enc(JxlEncoderFrameSettingsSetOption(settings, option, value));
    JxlFrameHeader header;
    JxlEncoderInitFrameHeader(&header);
    header.duration = 1;
    header.layer_info.save_as_reference = frame == 0 ? 1 : 0;
    header.layer_info.blend_info.source = frame == 0 ? 0 : 1;
    header.layer_info.blend_info.blendmode =
        frame == 0 ? JXL_BLEND_REPLACE : static_cast<JxlBlendMode>(mode);
    header.layer_info.blend_info.alpha = 0;
    header.layer_info.blend_info.clamp = JXL_TRUE;
    Enc(JxlEncoderSetFrameHeader(settings, &header));
    JxlBlendInfo alpha = header.layer_info.blend_info;
    alpha.blendmode =
        frame == 1 && alpha_reference ? JXL_BLEND_BLEND : JXL_BLEND_REPLACE;
    Enc(JxlEncoderSetExtraChannelBlendInfo(settings, 0, &alpha));
    const unsigned channels = info.num_color_channels + 1;
    std::vector<float> input(17 * 9 * channels);
    for (size_t p = 0; p < 17 * 9; ++p) {
      const float alpha = Alpha(p, frame);
      for (unsigned c = 0; c < info.num_color_channels; ++c) {
        const float value =
            (16 + ((p * info.num_color_channels + c) * 37 + frame * 13) % 63) /
            128.0f;
        input[p * channels + c] = associated ? value * alpha : value;
      }
      input[p * channels + info.num_color_channels] = alpha;
    }
    const JxlPixelFormat format = {channels, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN,
                                   0};
    Enc(JxlEncoderAddImageFrame(settings, &format, input.data(),
                                input.size() * 4));
  }
  JxlEncoderCloseInput(encoder);
  Bytes bytes(1 << 20);
  auto *next = bytes.data();
  size_t available = bytes.size();
  Enc(JxlEncoderProcessOutput(encoder, &next, &available));
  bytes.resize(bytes.size() - available);
  return bytes;
}

Bytes DataProfile(JxlDecoder *decoder) {
  size_t size = 0;
  Check(JxlDecoderGetICCProfileSize(decoder, JXL_COLOR_PROFILE_TARGET_DATA,
                                    &size) == JXL_DEC_SUCCESS,
        "DATA ICC size");
  Bytes profile(size);
  Check(JxlDecoderGetColorAsICCProfile(decoder, JXL_COLOR_PROFILE_TARGET_DATA,
                                       profile.data(),
                                       profile.size()) == JXL_DEC_SUCCESS,
        "DATA ICC bytes");
  return profile;
}

void Decode(const Bytes &bytes, const Bytes &profile, bool gray,
            bool associated, unsigned mode, bool alpha_reference,
            bool coalescing, bool cms, const std::string &prefix) {
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> owned(
      JxlDecoderCreate(nullptr), JxlDecoderDestroy);
  auto *decoder = owned.get();
  Check(decoder != nullptr, "decoder");
  Check(JxlDecoderSetCoalescing(decoder, coalescing) == JXL_DEC_SUCCESS,
        "coalescing");
  Check(JxlDecoderSetUnpremultiplyAlpha(decoder, JXL_FALSE) == JXL_DEC_SUCCESS,
        "preserve alpha association");
  if (cms)
    Check(JxlDecoderSetCms(decoder, *JxlGetDefaultCms()) == JXL_DEC_SUCCESS,
          "CMS");
  Check(JxlDecoderSubscribeEvents(decoder,
                                  JXL_DEC_COLOR_ENCODING | JXL_DEC_FRAME |
                                      JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS,
        "events");
  Check(JxlDecoderSetInput(decoder, bytes.data(), bytes.size()) ==
            JXL_DEC_SUCCESS,
        "input");
  JxlDecoderCloseInput(decoder);
  const JxlPixelFormat format = {gray ? 2u : 4u, JXL_TYPE_FLOAT,
                                 JXL_NATIVE_ENDIAN, 0};
  std::vector<float> pixels(17 * 9 * format.num_channels);
  unsigned frames = 0, headers = 0;
  Bytes initial_data_profile;
  int result = -1;
  for (;;) {
    const auto status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_COLOR_ENCODING) {
      JxlBasicInfo info;
      Check(JxlDecoderGetBasicInfo(decoder, &info) == JXL_DEC_SUCCESS &&
                info.num_extra_channels == 1 && info.alpha_bits == 32 &&
                info.alpha_exponent_bits == 8 &&
                !!info.alpha_premultiplied == associated,
            "alpha declaration");
      size_t size = 0;
      Check(JxlDecoderGetICCProfileSize(decoder,
                                        JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                                        &size) == JXL_DEC_SUCCESS,
            "ICC size");
      Bytes actual(size);
      Check(JxlDecoderGetColorAsICCProfile(
                decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL, actual.data(),
                actual.size()) == JXL_DEC_SUCCESS &&
                actual == profile,
            "original ICC");
      initial_data_profile = DataProfile(decoder);
      JxlColorEncoding fields;
      Check(JxlDecoderGetColorAsEncodedProfile(decoder,
                                               JXL_COLOR_PROFILE_TARGET_DATA,
                                               &fields) == JXL_DEC_SUCCESS &&
                fields.transfer_function == JXL_TRANSFER_FUNCTION_LINEAR &&
                fields.white_point == JXL_WHITE_POINT_D65 &&
                fields.color_space ==
                    (gray ? JXL_COLOR_SPACE_GRAY : JXL_COLOR_SPACE_RGB),
            "actual linear DATA");
      if (!gray)
        Check(fields.primaries == JXL_PRIMARIES_SRGB, "actual BT.709 DATA");
    } else if (status == JXL_DEC_FRAME) {
      JxlFrameHeader header;
      Check(JxlDecoderGetFrameHeader(decoder, &header) == JXL_DEC_SUCCESS,
            "frame header");
      if (!coalescing)
        Check(header.layer_info.blend_info.blendmode ==
                  (headers == 0 ? JXL_BLEND_REPLACE
                                : static_cast<JxlBlendMode>(mode)),
              "blend mode");
      if (!coalescing) {
        JxlBlendInfo alpha;
        Check(JxlDecoderGetExtraChannelBlendInfo(decoder, 0, &alpha) ==
                  JXL_DEC_SUCCESS,
              "extra blend header");
        Check(alpha.blendmode == (headers == 1 && alpha_reference
                                      ? JXL_BLEND_BLEND
                                      : JXL_BLEND_REPLACE),
              "physical alpha blend mode");
        Check(alpha.source == (headers == 1 && alpha_reference ? 1u : 0u),
              "physical alpha background source");
      }
      ++headers;

    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t required = 0;
      Check(JxlDecoderImageOutBufferSize(decoder, &format, &required) ==
                    JXL_DEC_SUCCESS &&
                required == pixels.size() * 4,
            "output size");
      Check(JxlDecoderSetImageOutBuffer(decoder, &format, pixels.data(),
                                        required) == JXL_DEC_SUCCESS,
            "output");
    } else if (status == JXL_DEC_FULL_IMAGE) {
      const auto data_profile = DataProfile(decoder);
      Check(data_profile == initial_data_profile,
            "stable DATA ICC through frames");
      Write(prefix + ".frame" + std::to_string(frames) + ".data.icc",
            data_profile);
      for (size_t p = 0; p < 17 * 9; ++p)
        Check(pixels[p * format.num_channels + format.num_channels - 1] ==
                  Alpha(p, frames),
              "exact physical alpha");
      Bytes words;
      for (const float pixel : pixels) {
        uint32_t bits;
        std::memcpy(&bits, &pixel, sizeof(bits));
        for (unsigned shift = 0; shift < 32; shift += 8)
          words.push_back(bits >> shift);
      }
      Write(prefix + ".frame" + std::to_string(frames) + ".linear.f32le",
            words);
      ++frames;
    } else if (status == JXL_DEC_SUCCESS || status == JXL_DEC_ERROR) {
      result = status;
      break;
    } else
      Check(false, "unexpected decoder status");
  }
  if (!coalescing)
    Check(result == JXL_DEC_SUCCESS && frames == 2 && headers == 2,
          "complete physical layers");
  std::printf("{\"case\":\"%s\",\"coalescing\":%s,\"cms\":%s,\"headers\":%u,"
              "\"frames\":%u,\"result\":%d}\n",
              std::filesystem::path(prefix).filename().string().c_str(),
              coalescing ? "true" : "false", cms ? "true" : "false", headers,
              frames, result);
}

int main(int argc, char **argv) {
  Check(argc == 3, "usage: alpha profiles_directory output_directory");
  Check(JxlEncoderVersion() == 12000 && JxlDecoderVersion() == 12000,
        "libjxl 0.12.0 required");
  std::filesystem::create_directories(argv[2]);
  for (bool gray : {false, true})
    for (bool modular : {true, false})
      for (bool associated : {false, true})
        for (unsigned mode = 0; mode <= 4; ++mode)
          for (bool alpha_reference : {false, true}) {
            if (alpha_reference && mode != 2)
              continue;
            const auto profile =
                Read(std::string(argv[1]) + (gray ? "/gray.icc" : "/rgb.icc"));
            const auto bytes = Encode(profile, gray, modular, associated, mode,
                                      alpha_reference);
            const auto name = std::string(gray ? "gray" : "rgb") +
                              (modular ? "_modular" : "_vardct") +
                              (associated ? "_associated" : "_straight") +
                              "_m" + std::to_string(mode) +
                              (alpha_reference ? "_alpha_ref1" : "");
            Write(std::string(argv[2]) + "/" + name + ".jxl", bytes);
            for (bool coalescing : {false, true})
              for (bool cms : {false, true}) {
                const auto prefix = std::string(argv[2]) + "/" + name +
                                    (coalescing ? "_composed" : "_layers") +
                                    (cms ? "_cms" : "_builtin");
                Decode(bytes, profile, gray, associated, mode, alpha_reference,
                       coalescing, cms, prefix);
              }
          }
}
