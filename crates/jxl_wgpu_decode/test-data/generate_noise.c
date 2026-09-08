/* Offline libjxl 0.12 fixture generator; production never links a CPU codec.
 * cc generate_noise.c $(pkg-config --cflags --libs libjxl) -o /tmp/generate-noise
 * /tmp/generate-noise OUTPUT_DIRECTORY
 */
#include <jxl/encode.h>
#include <jxl/color_encoding.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void check(JxlEncoderStatus status) {
  if (status != JXL_ENC_SUCCESS) { fprintf(stderr, "encoder error %d\n", status); exit(1); }
}
static void generate(const char* dir, const char* name, uint32_t w, uint32_t h,
                     int modular, int factor, int count, int mixed) {
  JxlEncoder* enc = JxlEncoderCreate(NULL);
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = w; info.ysize = h; info.bits_per_sample = 8;
  info.num_color_channels = 3; info.uses_original_profile = JXL_FALSE;
  info.have_animation = count > 1;
  info.animation.tps_numerator = 10; info.animation.tps_denominator = 1;
  check(JxlEncoderSetBasicInfo(enc, &info));
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
  check(JxlEncoderSetColorEncoding(enc, &color));
  const size_t size = (size_t)w * h * 3;
  uint8_t* data = malloc(size);
  if (!data) exit(2);
  for (int frame = 0; frame < count; ++frame) {
    for (uint32_t y = 0; y < h; ++y) for (uint32_t x = 0; x < w; ++x)
      for (uint32_t c = 0; c < 3; ++c)
        data[((size_t)y*w + x)*3 + c] = 32 + ((x*(c+2) + y*(7-c) + frame*11) % 192);
    JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(enc, NULL);
    check(JxlEncoderSetFrameDistance(settings, 2.0f));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 1));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, mixed ? frame%2 : modular));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PATCHES, 0));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_DOTS, 0));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_GABORISH, factor > 1));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EPF, factor > 1 ? 2 : 0));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, 0));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_RESAMPLING, factor));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_NOISE, 1));
    check(JxlEncoderFrameSettingsSetFloatOption(settings, JXL_ENC_FRAME_SETTING_PHOTON_NOISE, 800));
    JxlFrameHeader header;
    JxlEncoderInitFrameHeader(&header);
    header.duration = count == 1 ? 0 : (frame == 0 ? 2 : (frame == 3 ? 3 : 0));
    check(JxlEncoderSetFrameHeader(settings, &header));
    JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    check(JxlEncoderAddImageFrame(settings, &format, data, size));
  }
  free(data);
  JxlEncoderCloseInput(enc);
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s.jxl.hex", dir, name);
  FILE* file = fopen(path, "w"); if (!file) exit(2);
  size_t written = 0;
  JxlEncoderStatus status;
  do {
    uint8_t bytes[16384]; uint8_t* next = bytes; size_t available = sizeof(bytes);
    status = JxlEncoderProcessOutput(enc, &next, &available);
    if (status != JXL_ENC_SUCCESS && status != JXL_ENC_NEED_MORE_OUTPUT) exit(3);
    for (size_t i = 0; i < sizeof(bytes)-available; ++i) {
      fprintf(file, "%02x", bytes[i]); if (++written % 32 == 0) fputc('\n', file);
    }
  } while (status == JXL_ENC_NEED_MORE_OUTPUT);
  if (written % 32) fputc('\n', file);
  fclose(file); JxlEncoderDestroy(enc);
  fprintf(stderr, "%s: %zu bytes\n", name, written);
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  generate(argv[1], "vardct_257x17", 257, 17, 0, 1, 1, 0);
  generate(argv[1], "modular_257x17", 257, 17, 1, 1, 1, 0);
  for (int mode = 0; mode <= 1; ++mode) for (int factor = 2; factor <= 8; factor *= 2) {
    char name[80]; snprintf(name, sizeof(name), "%s_up%d", mode ? "modular" : "vardct", factor);
    generate(argv[1], name, 259, 33, mode, factor, 1, 0);
  }
  generate(argv[1], "mixed_frames", 37, 19, 0, 1, 5, 1);
  return 0;
}
