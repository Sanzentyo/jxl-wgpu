#include <jxl/encode.h>
#include <jxl/decode.h>
#include <jxl/cms.h>

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
void Check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void Enc(JxlEncoderStatus status) { Check(status == JXL_ENC_SUCCESS, "encode API"); }
Bytes Read(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "input file");
  return Bytes(std::istreambuf_iterator<char>(input), {});
}
void Write(const std::string& path, const Bytes& bytes) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  Check(output.good(), "output file");
}

Bytes Encode(const Bytes& profile, bool gray, bool modular) {
  std::unique_ptr<JxlEncoder, decltype(&JxlEncoderDestroy)> owned(JxlEncoderCreate(nullptr), JxlEncoderDestroy);
  auto* encoder = owned.get();
  Check(encoder != nullptr, "encoder");
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = 17; info.ysize = 9;
  info.bits_per_sample = 32; info.exponent_bits_per_sample = 8;
  info.num_color_channels = gray ? 1 : 3;
  info.uses_original_profile = JXL_FALSE;
  info.have_animation = JXL_TRUE;
  info.animation.tps_numerator = 100; info.animation.tps_denominator = 1;
  Enc(JxlEncoderSetBasicInfo(encoder, &info));
  Enc(JxlEncoderSetICCProfile(encoder, profile.data(), profile.size()));
  for (unsigned frame = 0; frame < 2; ++frame) {
    auto* settings = JxlEncoderFrameSettingsCreate(encoder, nullptr);
    Enc(JxlEncoderSetFrameDistance(settings, 1));
    for (const auto [option, value] : std::array<std::pair<JxlEncoderFrameSettingId, int>, 9>{{
        {JXL_ENC_FRAME_SETTING_MODULAR, modular}, {JXL_ENC_FRAME_SETTING_EFFORT, 3},
        {JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM, 0}, {JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1},
        {JXL_ENC_FRAME_SETTING_PATCHES, 0}, {JXL_ENC_FRAME_SETTING_DOTS, 0},
        {JXL_ENC_FRAME_SETTING_NOISE, 0}, {JXL_ENC_FRAME_SETTING_GABORISH, 0},
        {JXL_ENC_FRAME_SETTING_EPF, 0}}}) Enc(JxlEncoderFrameSettingsSetOption(settings, option, value));
    JxlFrameHeader header;
    JxlEncoderInitFrameHeader(&header);
    header.duration = 1;
    header.layer_info.save_as_reference = frame == 0 ? 1 : 0;
    header.layer_info.blend_info.source = frame == 0 ? 0 : 1;
    header.layer_info.blend_info.blendmode = frame == 0 ? JXL_BLEND_REPLACE : JXL_BLEND_ADD;
    Enc(JxlEncoderSetFrameHeader(settings, &header));
    std::vector<float> input(17 * 9 * info.num_color_channels);
    for (size_t i = 0; i < input.size(); ++i) input[i] = (16 + (i * 37 + frame * 13) % 63) / 128.0f;
    const JxlPixelFormat format = {info.num_color_channels, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
    Enc(JxlEncoderAddImageFrame(settings, &format, input.data(), input.size() * 4));
  }
  JxlEncoderCloseInput(encoder);
  Bytes bytes(1 << 20);
  auto* next = bytes.data(); size_t available = bytes.size();
  Enc(JxlEncoderProcessOutput(encoder, &next, &available));
  bytes.resize(bytes.size() - available);
  return bytes;
}

void Decode(const Bytes& bytes, const Bytes& profile, bool gray, bool coalescing, bool cms,
            const std::string& prefix) {
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> owned(JxlDecoderCreate(nullptr), JxlDecoderDestroy);
  auto* decoder = owned.get();
  Check(decoder != nullptr, "decoder");
  Check(JxlDecoderSetCoalescing(decoder, coalescing) == JXL_DEC_SUCCESS, "coalescing");
  if (cms) Check(JxlDecoderSetCms(decoder, *JxlGetDefaultCms()) == JXL_DEC_SUCCESS, "CMS");
  Check(JxlDecoderSubscribeEvents(decoder, JXL_DEC_COLOR_ENCODING | JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS, "events");
  Check(JxlDecoderSetInput(decoder, bytes.data(), bytes.size()) == JXL_DEC_SUCCESS, "input");
  JxlDecoderCloseInput(decoder);
  const JxlPixelFormat format = {gray ? 1u : 3u, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  std::vector<float> pixels(17 * 9 * format.num_channels);
  unsigned frames = 0, headers = 0;
  int result = -1;
  for (;;) {
    const auto status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_COLOR_ENCODING) {
      size_t size = 0;
      Check(JxlDecoderGetICCProfileSize(decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &size) == JXL_DEC_SUCCESS, "ICC size");
      Bytes actual(size);
      Check(JxlDecoderGetColorAsICCProfile(decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL, actual.data(), actual.size()) == JXL_DEC_SUCCESS && actual == profile, "original ICC");
      JxlColorEncoding fields;
      Check(JxlDecoderGetColorAsEncodedProfile(decoder, JXL_COLOR_PROFILE_TARGET_DATA, &fields) == JXL_DEC_SUCCESS
          && fields.transfer_function == JXL_TRANSFER_FUNCTION_LINEAR
          && fields.white_point == JXL_WHITE_POINT_D65
          && fields.color_space == (gray ? JXL_COLOR_SPACE_GRAY : JXL_COLOR_SPACE_RGB), "actual linear DATA");
      if (!gray) Check(fields.primaries == JXL_PRIMARIES_SRGB, "actual BT.709 DATA");
    } else if (status == JXL_DEC_FRAME) {
      JxlFrameHeader header;
      Check(JxlDecoderGetFrameHeader(decoder, &header) == JXL_DEC_SUCCESS, "frame header");
      if (!coalescing) Check(header.layer_info.blend_info.blendmode == (headers == 0 ? JXL_BLEND_REPLACE : JXL_BLEND_ADD), "blend mode");
      ++headers;
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t required = 0;
      Check(JxlDecoderImageOutBufferSize(decoder, &format, &required) == JXL_DEC_SUCCESS
          && required == pixels.size() * 4, "output size");
      Check(JxlDecoderSetImageOutBuffer(decoder, &format, pixels.data(), required) == JXL_DEC_SUCCESS, "output");
    } else if (status == JXL_DEC_FULL_IMAGE) {
      Bytes words;
      for (const float pixel : pixels) {
        uint32_t bits; std::memcpy(&bits, &pixel, sizeof(bits));
        for (unsigned shift = 0; shift < 32; shift += 8) words.push_back(bits >> shift);
      }
      Write(prefix + ".frame" + std::to_string(frames) + ".linear.f32le", words);
      ++frames;
    } else if (status == JXL_DEC_SUCCESS || status == JXL_DEC_ERROR) { result = status; break; }
    else Check(false, "unexpected decoder status");
  }
  std::printf("{\"case\":\"%s\",\"coalescing\":%s,\"cms\":%s,\"headers\":%u,\"frames\":%u,\"result\":%d}\n",
      std::filesystem::path(prefix).filename().string().c_str(), coalescing ? "true" : "false", cms ? "true" : "false", headers, frames, result);
}

int main(int argc, char** argv) {
  Check(argc == 3, "usage: probe profiles_directory output_directory");
  Check(JxlEncoderVersion() == 12000 && JxlDecoderVersion() == 12000, "libjxl 0.12.0 required");
  std::filesystem::create_directories(argv[2]);
  for (bool gray : {false, true}) for (bool modular : {true, false}) {
    const auto profile = Read(std::string(argv[1]) + (gray ? "/gray.icc" : "/rgb.icc"));
    const auto bytes = Encode(profile, gray, modular);
    const auto name = std::string(gray ? "gray" : "rgb") + (modular ? "_modular" : "_vardct");
    Write(std::string(argv[2]) + "/" + name + ".jxl", bytes);
    for (bool coalescing : {false, true}) for (bool cms : {false, true}) {
      const auto prefix = std::string(argv[2]) + "/" + name + (coalescing ? "_composed" : "_layers") + (cms ? "_cms" : "_builtin");
      Decode(bytes, profile, gray, coalescing, cms, prefix);
    }
  }
}
