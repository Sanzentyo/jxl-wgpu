/* Offline libjxl 0.12 fixture generator; production never links this CPU codec.
 * cc generate_extra_channels.c $(pkg-config --cflags --libs libjxl) -o /tmp/jxl-extras
 * /tmp/jxl-extras OUTPUT_DIRECTORY [--vardct]
 */
#include <jxl/encode.h>
#include <jxl/color_encoding.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int vardct;

static void check(JxlEncoderStatus status) { if (status != JXL_ENC_SUCCESS) exit(1); }
static const JxlExtraChannelType types[] = {
  JXL_CHANNEL_DEPTH, JXL_CHANNEL_SELECTION_MASK, JXL_CHANNEL_ALPHA,
  JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_CFA, JXL_CHANNEL_THERMAL,
  JXL_CHANNEL_BLACK, JXL_CHANNEL_OPTIONAL, JXL_CHANNEL_ALPHA,
};
static const uint32_t depths[] = {16, 1, 7, 12, 4, 8, 6, 10, 15};

static uint32_t code(uint32_t x, uint32_t y, uint32_t c, uint32_t bits) {
  uint32_t mask = (1u << bits) - 1;
  if (x % 11 == 0) return 0;
  if (x % 11 == 1) return mask;
  return (193*x + 317*y + 97*c + ((x^y)*(23+c))) & mask;
}

static void generate(const char* dir, const char* name, uint32_t width, uint32_t height,
    uint32_t colors, uint32_t bits, uint32_t extras, int orientation, int effort, int alpha_only) {
  JxlEncoder* enc = JxlEncoderCreate(NULL);
  JxlBasicInfo info; JxlEncoderInitBasicInfo(&info);
  info.xsize = width; info.ysize = height; info.bits_per_sample = bits;
  info.num_color_channels = colors; info.num_extra_channels = extras;
  info.uses_original_profile = !vardct; info.orientation = (JxlOrientation)orientation;
  check(JxlEncoderSetBasicInfo(enc, &info));
  for (uint32_t c = 0; c < extras; ++c) {
    JxlExtraChannelInfo ec; JxlEncoderInitExtraChannelInfo(alpha_only ? JXL_CHANNEL_ALPHA : types[c], &ec);
    ec.bits_per_sample = alpha_only ? 5 : depths[c];
    ec.spot_color[0] = 0.25f; ec.spot_color[1] = 0.5f; ec.spot_color[2] = 0.75f; ec.spot_color[3] = 0.5f;
    ec.cfa_channel = 3;
    check(JxlEncoderSetExtraChannelInfo(enc, c, &ec));
    char name[64]; int length = snprintf(name, sizeof(name), "plane-%u-depth-%u", c, ec.bits_per_sample);
    check(JxlEncoderSetExtraChannelName(enc, c, name, (size_t)length));
  }
  JxlColorEncoding color; JxlColorEncodingSetToSRGB(&color, colors == 1);
  check(JxlEncoderSetColorEncoding(enc, &color));
  JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(enc, NULL);
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, effort));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, !vardct));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PATCHES, 0));
  if (vardct) {
    check(JxlEncoderSetFrameDistance(settings, 1.0f));
    for (uint32_t c=0; c<extras; ++c) check(JxlEncoderSetExtraChannelDistance(settings, c, 0.0f));
  } else check(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
  size_t pixels = (size_t)width * height;
  float* data = malloc(pixels * colors * sizeof(float));
  if (!data) exit(2);
  for (uint32_t y=0; y<height; ++y) for (uint32_t x=0; x<width; ++x) for (uint32_t c=0; c<colors; ++c)
    data[((size_t)y*width+x)*colors+c] = (float)code(x,y,c,bits) / (float)((1u<<bits)-1);
  JxlPixelFormat format = {colors, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  check(JxlEncoderAddImageFrame(settings, &format, data, pixels * colors * sizeof(float)));
  for (uint32_t c=0; c<extras; ++c) {
    uint32_t depth = alpha_only ? 5 : depths[c];
    for (uint32_t y=0; y<height; ++y) for (uint32_t x=0; x<width; ++x)
      data[(size_t)y*width+x] = (float)code(x,y,colors+c,depth) / (float)((1u<<depth)-1);
    check(JxlEncoderSetExtraChannelBuffer(settings, &format, data, pixels * sizeof(float), c));
  }
  free(data); JxlEncoderCloseInput(enc);
  char path[1024]; snprintf(path, sizeof(path), "%s/%sextras_%s.jxl.hex", dir, vardct ? "vardct_" : "", name);
  FILE* out = fopen(path, "w"); if (!out) exit(2);
  JxlEncoderStatus status; size_t written = 0;
  do {
    uint8_t buffer[16384]; uint8_t* next = buffer; size_t available = sizeof(buffer);
    status = JxlEncoderProcessOutput(enc, &next, &available);
    if (status != JXL_ENC_SUCCESS && status != JXL_ENC_NEED_MORE_OUTPUT) {
      fprintf(stderr, "%s: encoder error %d\n", name, JxlEncoderGetError(enc)); exit(3);
    }
    for (size_t i=0; i<sizeof(buffer)-available; ++i) {
      fprintf(out, "%02x", buffer[i]); if (++written % 32 == 0) fputc('\n', out);
    }
  } while (status == JXL_ENC_NEED_MORE_OUTPUT);
  if (written % 32) fputc('\n', out);
  fclose(out); JxlEncoderDestroy(enc);
  fprintf(stderr, "%s: %zu bytes\n", name, written);
}

int main(int argc, char** argv) {
  if (argc != 2 && (argc != 3 || strcmp(argv[2], "--vardct"))) return 2;
  vardct = argc == 3;
  if (vardct) {
    generate(argv[1], "data_only", 17, 1, 3, 8, 2, 6, 1, 0);
    generate(argv[1], "rgb12", 33, 7, 3, 12, 9, 6, 1, 0);
    generate(argv[1], "gray8", 33, 7, 1, 8, 9, 8, 1, 0);
    generate(argv[1], "gray_alpha", 37, 9, 1, 16, 1, 5, 1, 1);
    generate(argv[1], "rgba", 63, 9, 3, 8, 1, 3, 1, 1);
    generate(argv[1], "transformed", 127, 129, 3, 12, 9, 7, 7, 0);
    return 0;
  }
  generate(argv[1], "data_only", 17, 1, 3, 8, 2, 6, 1, 0);
  generate(argv[1], "rgb12", 259, 17, 3, 12, 9, 6, 1, 0);
  generate(argv[1], "gray8", 33, 7, 1, 8, 9, 8, 1, 0);
  generate(argv[1], "gray_alpha", 257, 9, 1, 16, 1, 5, 1, 1);
  generate(argv[1], "rgba", 259, 9, 3, 8, 1, 3, 1, 1);
  generate(argv[1], "transformed", 515, 259, 3, 12, 9, 7, 7, 0);
  return 0;
}
