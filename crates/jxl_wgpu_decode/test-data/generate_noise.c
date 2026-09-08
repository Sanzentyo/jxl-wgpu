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
typedef struct {
  int modular, factor, count, mixed, original, group_shift, gray, channel_palette;
  int bits, exponent, orientation;
} Options;
static void generate(const char* dir, const char* name, uint32_t w, uint32_t h,
                     Options options) {
  const int channels = options.gray ? 1 : 3;
  const int factor = options.factor;
  const int count = options.count;
  JxlEncoder* enc = JxlEncoderCreate(NULL);
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = w; info.ysize = h; info.bits_per_sample = options.bits ? options.bits : 8;
  info.exponent_bits_per_sample = options.exponent;
  if (options.orientation) info.orientation = options.orientation;
  info.num_color_channels = channels; info.uses_original_profile = options.original;
  info.have_animation = count > 1;
  info.animation.tps_numerator = 10; info.animation.tps_denominator = 1;
  check(JxlEncoderSetBasicInfo(enc, &info));
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, options.gray);
  check(JxlEncoderSetColorEncoding(enc, &color));
  const size_t size = (size_t)w * h * channels;
  uint8_t* data = malloc(size);
  uint16_t* wide = options.bits == 16 ? malloc(size * sizeof(uint16_t)) : NULL;
  float* floating = options.exponent ? malloc(size * sizeof(float)) : NULL;
  if (!data || (options.bits == 16 && !wide) || (options.exponent && !floating)) exit(2);
  for (int frame = 0; frame < count; ++frame) {
    for (uint32_t y = 0; y < h; ++y) for (uint32_t x = 0; x < w; ++x)
      for (int c = 0; c < channels; ++c)
        data[((size_t)y*w + x)*channels + c] = 32 + ((x*(c+2) + y*(7-c) + frame*11) % 192);
    JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(enc, NULL);
    check(JxlEncoderSetFrameDistance(settings, 2.0f));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 1));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, options.mixed ? frame%2 : options.modular));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, options.group_shift));
    if (options.original && !options.channel_palette) {
      check(JxlEncoderFrameSettingsSetFloatOption(settings, JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GLOBAL_PERCENT, 0));
      check(JxlEncoderFrameSettingsSetFloatOption(settings, JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GROUP_PERCENT, 0));
    }
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
    JxlPixelFormat format = {(uint32_t)channels, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    const void* pixels = data; size_t bytes = size;
    if (wide) {
      for (size_t i = 0; i < size; ++i) wide[i] = data[i] * 256 + (i * 37 + frame * 13) % 256;
      format.data_type = JXL_TYPE_UINT16; pixels = wide; bytes *= sizeof(uint16_t);
    } else if (floating) {
      for (size_t i = 0; i < size; ++i)
        floating[i] = ((float)data[i] + (float)((i * 37 + frame * 13) % 256) / 256.0f) / 255.0f;
      format.data_type = JXL_TYPE_FLOAT; pixels = floating; bytes *= sizeof(float);
    }
    check(JxlEncoderAddImageFrame(settings, &format, pixels, bytes));
  }
  free(floating); free(wide); free(data);
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
  generate(argv[1], "vardct_257x17", 257, 17, (Options){.factor=1, .count=1, .group_shift=-1});
  generate(argv[1], "modular_257x17", 257, 17, (Options){.modular=1, .factor=1, .count=1, .group_shift=-1});
  for (int mode = 0; mode <= 1; ++mode) for (int factor = 2; factor <= 8; factor *= 2) {
    char name[80]; snprintf(name, sizeof(name), "%s_up%d", mode ? "modular" : "vardct", factor);
    generate(argv[1], name, 259, 33, (Options){.modular=mode, .factor=factor, .count=1, .group_shift=-1});
  }
  generate(argv[1], "mixed_frames", 37, 19, (Options){.factor=1, .count=5, .mixed=1, .group_shift=-1});
  for (int original = 0; original <= 1; ++original) for (int shift = 0; shift <= 3; ++shift) {
    char name[80]; snprintf(name, sizeof(name), "modular_%s_group%d", original ? "rgb" : "xyb", 128 << shift);
    generate(argv[1], name, (128 << shift) + 1, 17,
        (Options){.modular=1, .factor=1, .count=1, .original=original, .group_shift=shift});
  }
  generate(argv[1], "modular_gray", 257, 17,
      (Options){.modular=1, .factor=1, .count=1, .original=1, .group_shift=0, .gray=1});
  /* Preserve the lossy single-channel palette case: libjxl clamps its out-of-range
   * indices, while ISO/IEC 18181-1 H.6.4 specifies implicit palette values. */
  generate(argv[1], "modular_rgb_palette", 129, 17,
      (Options){.modular=1, .factor=1, .count=1, .original=1, .group_shift=0, .channel_palette=1});
  for (int factor = 2; factor <= 8; factor *= 2) {
    char name[80]; snprintf(name, sizeof(name), "modular_rgb_up%d", factor);
    generate(argv[1], name, 259, 33,
        (Options){.modular=1, .factor=factor, .count=1, .original=1, .group_shift=0});
  }
  generate(argv[1], "vardct_rgb_257x17", 257, 17,
      (Options){.factor=1, .count=1, .original=1, .group_shift=-1});
  generate(argv[1], "vardct_rgb_gray", 257, 17,
      (Options){.factor=1, .count=1, .original=1, .group_shift=-1, .gray=1});
  for (int factor = 2; factor <= 8; factor *= 2) {
    char name[80]; snprintf(name, sizeof(name), "vardct_rgb_up%d", factor);
    generate(argv[1], name, 259, 33,
        (Options){.factor=factor, .count=1, .original=1, .group_shift=-1});
  }
  generate(argv[1], "vardct_rgb_gray16", 37, 19,
      (Options){.factor=1, .count=1, .original=1, .group_shift=-1, .gray=1, .bits=16, .orientation=6});
  generate(argv[1], "vardct_rgb_float32_up4", 259, 33,
      (Options){.factor=4, .count=1, .original=1, .group_shift=-1, .bits=32, .exponent=8});
  generate(argv[1], "vardct_rgb_frames", 37, 19,
      (Options){.factor=1, .count=5, .original=1, .group_shift=-1, .orientation=8});
  return 0;
}
