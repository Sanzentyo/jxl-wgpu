// Development-only jxli producer using the unmodified public libjxl 0.12.0 encoder.
#include <jxl/encode.h>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

static void check(bool ok, const char* message) {
  if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(2); }
}
int main(int argc, char** argv) {
  check(JxlEncoderVersion() == 12000, "libjxl 0.12.0 required");
  check(argc == 2, "still/dense/sparse");
  const std::string mode(argv[1]);
  check(mode == "still" || mode == "dense" || mode == "sparse", "mode");
  std::unique_ptr<JxlEncoder, decltype(&JxlEncoderDestroy)> enc(JxlEncoderCreate(nullptr), JxlEncoderDestroy);
  check(bool(enc), "encoder");
  check(JxlEncoderUseContainer(enc.get(), JXL_TRUE) == JXL_ENC_SUCCESS, "container");
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = 17;
  info.ysize = 9;
  info.bits_per_sample = 8;
  info.num_color_channels = 3;
  info.uses_original_profile = JXL_TRUE;
  info.have_animation = mode != "still";
  info.animation.tps_numerator = 1000;
  info.animation.tps_denominator = 3;
  info.animation.have_timecodes = JXL_TRUE;
  check(JxlEncoderSetBasicInfo(enc.get(), &info) == JXL_ENC_SUCCESS, "basic info");
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
  check(JxlEncoderSetColorEncoding(enc.get(), &color) == JXL_ENC_SUCCESS, "color");
  const unsigned durations[] = {3, 5, 7, 11};
  const unsigned count = info.have_animation ? 4 : 1;
  for (unsigned frame = 0; frame < count; ++frame) {
    auto* settings = JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
    check(settings != nullptr, "settings");
    check(JxlEncoderSetFrameLossless(settings, JXL_TRUE) == JXL_ENC_SUCCESS, "lossless");
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 1) == JXL_ENC_SUCCESS, "effort");
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_INDEX_BOX,
        mode == "sparse" ? frame % 2 == 0 : 1) == JXL_ENC_SUCCESS, "index");
    JxlFrameHeader header;
    JxlEncoderInitFrameHeader(&header);
    header.duration = info.have_animation ? durations[frame] : 0;
    header.timecode = 0x01020300 + frame;
    check(JxlEncoderSetFrameHeader(settings, &header) == JXL_ENC_SUCCESS, "frame header");
    std::vector<uint8_t> pixels(info.xsize * info.ysize * 3);
    for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = (i * 17 + frame * 29) % 256;
    JxlPixelFormat format{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    check(JxlEncoderAddImageFrame(settings, &format, pixels.data(), pixels.size()) == JXL_ENC_SUCCESS, "image");
  }
  JxlEncoderCloseInput(enc.get());
  std::vector<uint8_t> encoded(1 << 20);
  uint8_t* next = encoded.data();
  size_t available = encoded.size();
  check(JxlEncoderProcessOutput(enc.get(), &next, &available) == JXL_ENC_SUCCESS, "encode");
  encoded.resize(encoded.size() - available);
  check(std::fwrite(encoded.data(), 1, encoded.size(), stdout) == encoded.size(), "write");
  check(std::fflush(stdout) == 0, "flush");
}
