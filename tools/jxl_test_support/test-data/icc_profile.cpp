// Development-only original ICC and image metadata oracle, using public libjxl 0.12.0.
// No production library compiles or links this executable.
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

static void check(bool ok, const char* message) {
  if (!ok) { std::cerr << message << '\n'; std::exit(2); }
}
static void word(uint32_t value) {
  for (unsigned i = 0; i < 4; ++i) check(std::fputc((value >> (8*i)) & 255, stdout) != EOF, "write");
}
static void real(float value) {
  uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  word(bits);
}
int main(int argc, char** argv) {
  check(JxlDecoderVersion() == 12000, "libjxl 0.12.0 required");
  check(argc >= 2, "read/read-info/create");
  const bool info_only = std::string(argv[1]) == "read-info";
  std::vector<uint8_t> input;
  if (std::string(argv[1]) == "create") {
    int space, white, primaries, transfer, intent;
    JxlColorEncoding color{};
    check(static_cast<bool>(std::cin >> space >> white >> primaries >> transfer >> intent
        >> color.white_point_xy[0] >> color.white_point_xy[1]
        >> color.primaries_red_xy[0] >> color.primaries_red_xy[1]
        >> color.primaries_green_xy[0] >> color.primaries_green_xy[1]
        >> color.primaries_blue_xy[0] >> color.primaries_blue_xy[1] >> color.gamma), "color input");
    color.color_space = static_cast<JxlColorSpace>(space);
    color.white_point = static_cast<JxlWhitePoint>(white);
    color.primaries = static_cast<JxlPrimaries>(primaries);
    color.transfer_function = static_cast<JxlTransferFunction>(transfer);
    color.rendering_intent = static_cast<JxlRenderingIntent>(intent);
    std::unique_ptr<JxlEncoder, decltype(&JxlEncoderDestroy)> enc(JxlEncoderCreate(nullptr), JxlEncoderDestroy);
    check(bool(enc), "encoder");
    JxlBasicInfo info;
    JxlEncoderInitBasicInfo(&info);
    info.xsize = info.ysize = 1;
    info.bits_per_sample = 8;
    info.num_color_channels = space == JXL_COLOR_SPACE_GRAY ? 1 : 3;
    info.uses_original_profile = JXL_TRUE;
    check(JxlEncoderSetBasicInfo(enc.get(), &info) == JXL_ENC_SUCCESS, "basic info");
    check(JxlEncoderSetColorEncoding(enc.get(), &color) == JXL_ENC_SUCCESS, "color encoding");
    auto* settings = JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
    check(JxlEncoderSetFrameLossless(settings, JXL_TRUE) == JXL_ENC_SUCCESS, "lossless");
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 1) == JXL_ENC_SUCCESS, "effort");
    JxlPixelFormat format{info.num_color_channels, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    std::array<uint8_t, 3> pixel{57, 126, 218};
    check(JxlEncoderAddImageFrame(settings, &format, pixel.data(), info.num_color_channels) == JXL_ENC_SUCCESS, "image");
    JxlEncoderCloseInput(enc.get());
    input.resize(1 << 20);
    uint8_t* next = input.data();
    size_t available = input.size();
    check(JxlEncoderProcessOutput(enc.get(), &next, &available) == JXL_ENC_SUCCESS, "encode");
    input.resize(input.size() - available);
  } else {
    check((std::string(argv[1]) == "read" || info_only) && argc == 3, "read path");
    std::ifstream file(argv[2], std::ios::binary | std::ios::ate);
    check(bool(file) && file.tellg() > 0 && file.tellg() <= (64 << 20), "input size");
    input.resize(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    check(static_cast<bool>(file.read(reinterpret_cast<char*>(input.data()), input.size())), "read");
  }
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> dec(JxlDecoderCreate(nullptr), JxlDecoderDestroy);
  check(bool(dec), "decoder");
  const auto event = info_only ? JXL_DEC_BASIC_INFO : JXL_DEC_COLOR_ENCODING;
  check(JxlDecoderSetKeepOrientation(dec.get(), JXL_TRUE) == JXL_DEC_SUCCESS, "keep orientation");
  check(JxlDecoderSubscribeEvents(dec.get(), event) == JXL_DEC_SUCCESS, "subscribe");
  check(JxlDecoderSetInput(dec.get(), input.data(), input.size()) == JXL_DEC_SUCCESS, "input");
  JxlDecoderCloseInput(dec.get());
  check(JxlDecoderProcessInput(dec.get()) == event, "metadata event");
  if (info_only) {
    JxlBasicInfo info{};
    check(JxlDecoderGetBasicInfo(dec.get(), &info) == JXL_DEC_SUCCESS, "basic info");
    word(JxlDecoderVersion());
    word(info.xsize); word(info.ysize); word(info.orientation);
    word(info.intrinsic_xsize); word(info.intrinsic_ysize);
    real(info.intensity_target); real(info.min_nits);
    word(info.relative_to_max_display); real(info.linear_below);
    word(info.have_preview); word(info.preview.xsize); word(info.preview.ysize);
    word(info.have_animation);
    check(std::fflush(stdout) == 0, "flush");
    return 0;
  }
  size_t size = 0;
  check(JxlDecoderGetICCProfileSize(dec.get(), JXL_COLOR_PROFILE_TARGET_ORIGINAL, &size) == JXL_DEC_SUCCESS && size <= (16 << 20), "profile size");
  std::vector<uint8_t> profile(size);
  check(JxlDecoderGetColorAsICCProfile(dec.get(), JXL_COLOR_PROFILE_TARGET_ORIGINAL, profile.data(), size) == JXL_DEC_SUCCESS, "profile");
  word(static_cast<uint32_t>(input.size()));
  word(static_cast<uint32_t>(profile.size()));
  check(std::fwrite(input.data(), 1, input.size(), stdout) == input.size(), "input output");
  check(std::fwrite(profile.data(), 1, profile.size(), stdout) == profile.size(), "profile output");
  check(std::fflush(stdout) == 0, "flush");
}
