#include <jxl/color_encoding.h>
#include <jxl/cms.h>
#include <jxl/decode.h>
#include <jxl/encode.h>

#include <array>
#include <cctype>
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

void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

Bytes Read(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "input file");
  return Bytes(std::istreambuf_iterator<char>(input), {});
}

Bytes Unhex(const Bytes& source) {
  Bytes bytes;
  int high = -1;
  for (const unsigned char c : source) {
    if (std::isspace(c)) continue;
    const int digit = c >= '0' && c <= '9' ? c - '0'
        : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
    Check(digit >= 0, "invalid hex");
    if (high == -1) high = digit;
    else { bytes.push_back(high * 16 + digit); high = -1; }
  }
  Check(high == -1, "incomplete hex");
  return bytes;
}

void Write(const std::string& path, const Bytes& bytes) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  Check(output.good(), "output file");
}

Bytes Icc(const JxlDecoder* decoder, JxlColorProfileTarget target) {
  size_t size = 0;
  Check(JxlDecoderGetICCProfileSize(decoder, target, &size) == JXL_DEC_SUCCESS, "ICC size");
  Bytes bytes(size);
  Check(JxlDecoderGetColorAsICCProfile(decoder, target, bytes.data(), size) == JXL_DEC_SUCCESS, "ICC profile");
  return bytes;
}

void PrintFields(const JxlColorEncoding& color) {
  std::printf("{\"space\":%d,\"white\":%d,\"white_xy\":[%.17g,%.17g],"
      "\"primaries\":%d,\"red\":[%.17g,%.17g],\"green\":[%.17g,%.17g],"
      "\"blue\":[%.17g,%.17g],\"transfer\":%d,\"gamma\":%.17g,\"intent\":%d}",
      color.color_space, color.white_point, color.white_point_xy[0], color.white_point_xy[1],
      color.primaries, color.primaries_red_xy[0], color.primaries_red_xy[1],
      color.primaries_green_xy[0], color.primaries_green_xy[1],
      color.primaries_blue_xy[0], color.primaries_blue_xy[1],
      color.transfer_function, color.gamma, color.rendering_intent);
}

void Decode(const std::string& root, const std::string& output, const std::string& name,
            bool gray, bool set_cms, unsigned request) {
  const auto bytes = Unhex(Read(root + "/" + name + ".jxl.hex"));
  const auto profile = Read(root + (gray ? "/gray.icc" : "/rgb.icc"));
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> owned(JxlDecoderCreate(nullptr), JxlDecoderDestroy);
  auto* decoder = owned.get();
  Check(decoder != nullptr, "decoder");
  if (set_cms) Check(JxlDecoderSetCms(decoder, *JxlGetDefaultCms()) == JXL_DEC_SUCCESS, "CMS");
  Check(JxlDecoderSubscribeEvents(decoder, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE)
      == JXL_DEC_SUCCESS, "events");
  Check(JxlDecoderSetInput(decoder, bytes.data(), bytes.size()) == JXL_DEC_SUCCESS, "set input");
  JxlDecoderCloseInput(decoder);
  const JxlPixelFormat format = {gray ? 2u : 4u, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  std::vector<float> pixels(17 * 9 * format.num_channels);
  JxlColorEncoding fields{};
  unsigned images = 0;
  Bytes data_profile;
  const auto prefix = output + "/" + name + (set_cms ? "_cms" : "_builtin") + "_" + std::to_string(request);
  std::printf("{\"name\":\"%s\",\"cms\":%s,\"request\":%u", name.c_str(), set_cms ? "true" : "false", request);
  for (;;) {
    const auto status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info;
      Check(JxlDecoderGetBasicInfo(decoder, &info) == JXL_DEC_SUCCESS, "basic info");
      Check(info.xsize == 17 && info.ysize == 9 && info.uses_original_profile == JXL_FALSE
          && info.num_color_channels == (gray ? 1u : 3u), "XYB geometry and source mode");
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      Check(Icc(decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL) == profile, "original profile");
      const auto original_status = JxlDecoderGetColorAsEncodedProfile(decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &fields);
      Check(original_status == JXL_DEC_ERROR, "original must retain unrepresentable ICC");
      JxlColorEncoding target;
      JxlColorEncodingSetToLinearSRGB(&target, gray);
      target.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
      if (request == 1) Check(JxlDecoderSetPreferredColorProfile(decoder, &target) == JXL_DEC_SUCCESS, "preferred linear");
      if (request == 2) Check(JxlDecoderSetOutputColorProfile(decoder, &target, nullptr, 0) == JXL_DEC_SUCCESS, "output linear");
      Check(JxlDecoderGetColorAsEncodedProfile(decoder, JXL_COLOR_PROFILE_TARGET_DATA, &fields) == JXL_DEC_SUCCESS, "data encoding");
      Check(fields.color_space == (gray ? JXL_COLOR_SPACE_GRAY : JXL_COLOR_SPACE_RGB)
          && fields.white_point == JXL_WHITE_POINT_D65
          && fields.transfer_function == JXL_TRANSFER_FUNCTION_LINEAR, "actual linear data profile");
      if (!gray) Check(fields.primaries == JXL_PRIMARIES_SRGB, "actual BT.709 primaries");
      std::printf(",\"data_fields\":");
      PrintFields(fields);
      data_profile = Icc(decoder, JXL_COLOR_PROFILE_TARGET_DATA);
      Write(prefix + ".icc", data_profile);
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t required = 0;
      Check(JxlDecoderImageOutBufferSize(decoder, &format, &required) == JXL_DEC_SUCCESS
          && required == pixels.size() * 4, "buffer size");
      Check(JxlDecoderSetImageOutBuffer(decoder, &format, pixels.data(), required) == JXL_DEC_SUCCESS, "set buffer");
    } else if (status == JXL_DEC_FULL_IMAGE) ++images;
    else if (status == JXL_DEC_SUCCESS) break;
    else Check(false, "decode failed");
  }
  Check(images == 1, "image count");
  Check(Icc(decoder, JXL_COLOR_PROFILE_TARGET_DATA) == data_profile, "data profile changed during decoding");
  Bytes words;
  for (const auto pixel : pixels) {
    uint32_t bits;
    std::memcpy(&bits, &pixel, sizeof(bits));
    for (unsigned shift = 0; shift < 32; shift += 8) words.push_back(bits >> shift);
  }
  Write(prefix + ".f32le", words);
  std::printf(",\"first\":[");
  for (unsigned c = 0; c < format.num_channels; ++c) std::printf("%s%.9g", c ? "," : "", pixels[c]);
  std::printf("],\"samples\":%zu}\n", pixels.size());
}

int main(int argc, char** argv) {
  Check(argc == 3, "usage: probe corpus_directory output_directory");
  Check(JxlDecoderVersion() == 12000, "libjxl 0.12.0 required");
  std::filesystem::create_directories(argv[2]);
  for (bool gray : {false, true}) for (bool modular : {true, false}) {
    const auto name = std::string(gray ? "gray" : "rgb") + (modular ? "_modular_xyb" : "_vardct_xyb");
    for (bool cms : {false, true}) for (unsigned request : {0, 1, 2}) Decode(argv[1], argv[2], name, gray, cms, request);
  }
}
