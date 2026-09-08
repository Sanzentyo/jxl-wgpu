/* Offline libjxl 0.12 fixture generator; production never links this CPU codec.
 * cc generate_extra_channels.c $(pkg-config --cflags --libs libjxl) -o /tmp/jxl-extras
 * /tmp/jxl-extras OUTPUT_DIRECTORY [--vardct|--vardct-distributed|--resampled|--associated|--spots|--floating]
 */
#include <jxl/encode.h>
#include <jxl/color_encoding.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "floating_samples.h"
#include "integer_samples.h"

static int vardct;
static int responsive;
static int resampling = 1;
static int ec_resampling = 1;
static int dimension_shift;
static int associated;
static int spots;
static int floating;
static int wide_integer;
static int floating_narrow;
static int integer_primary;
static int progressive_dc;
static uint32_t alpha_depth = 5;

static void check(JxlEncoderStatus status) { if (status != JXL_ENC_SUCCESS) exit(1); }
static const JxlExtraChannelType types[] = {
  JXL_CHANNEL_DEPTH, JXL_CHANNEL_SELECTION_MASK, JXL_CHANNEL_ALPHA,
  JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_CFA, JXL_CHANNEL_THERMAL,
  JXL_CHANNEL_BLACK, JXL_CHANNEL_OPTIONAL, JXL_CHANNEL_ALPHA,
};
static const uint32_t depths[] = {16, 1, 7, 12, 4, 8, 6, 10, 15};
static const JxlExtraChannelType spot_types[] = {
  JXL_CHANNEL_DEPTH, JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_ALPHA,
  JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_THERMAL,
  JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_ALPHA,
};
/* Includes a transparent ink, extended colors/solidity, and an opaque ink. */
static const float spot_rgba[9][4] = {
  {0}, {0.75f, 0.125f, 0.25f, 0}, {0}, {0.25f, 0.5f, 0.75f, 0.5f},
  {1.5f, -0.25f, 0.125f, 1.25f}, {0}, {-0.125f, 1, 0.375f, -0.5f},
  {0.5f, 0.25f, 1, 1}, {0},
};

static uint32_t code(uint32_t x, uint32_t y, uint32_t c, uint32_t bits) {
  uint32_t mask = (1u << bits) - 1;
  if (x % 11 == 0) return 0;
  if (x % 11 == 1) return mask;
  return (193*x + 317*y + 97*c + ((x^y)*(23+c))) & mask;
}

static void generate(const char* dir, const char* name, uint32_t width, uint32_t height,
    uint32_t colors, uint32_t bits, uint32_t extras, int orientation, int effort, int alpha_only, int progressive) {
  JxlEncoder* enc = JxlEncoderCreate(NULL);
  JxlBasicInfo info; JxlEncoderInitBasicInfo(&info);
  info.xsize = width; info.ysize = height; info.bits_per_sample = bits;
  if (floating && !integer_primary) info.exponent_bits_per_sample = floating_exponent(bits);
  info.num_color_channels = colors; info.num_extra_channels = extras;
  info.uses_original_profile = !vardct; info.orientation = (JxlOrientation)orientation;
  if (floating || wide_integer) check(JxlEncoderSetCodestreamLevel(enc, 10));
  check(JxlEncoderSetBasicInfo(enc, &info));
  for (uint32_t c = 0; c < extras; ++c) {
    JxlExtraChannelInfo ec; JxlEncoderInitExtraChannelInfo(alpha_only ? JXL_CHANNEL_ALPHA : (spots ? spot_types[c] : types[c]), &ec);
    ec.bits_per_sample = alpha_only ? alpha_depth : depths[c];
    if (wide_integer) ec.bits_per_sample = alpha_only ? alpha_depth : integer_extra_bits[c];
    if (floating) {
      ec.bits_per_sample = alpha_only ? 32 : floating_extra_bits[c];
      ec.exponent_bits_per_sample = alpha_only ? 8 : floating_extra_exponents[c];
      // Encoder downsampling produces arbitrary binary32 values; keep them losslessly.
      if (ec_resampling > 1 && ec.exponent_bits_per_sample) {
        ec.bits_per_sample = 32; ec.exponent_bits_per_sample = 8;
      }
      if (floating_narrow && ec.exponent_bits_per_sample) {
        ec.bits_per_sample = 16; ec.exponent_bits_per_sample = 5;
      }
    }
    ec.dim_shift = dimension_shift;
    ec.alpha_premultiplied = associated && (alpha_only || c == 2);
    ec.spot_color[0] = 0.25f; ec.spot_color[1] = 0.5f; ec.spot_color[2] = 0.75f; ec.spot_color[3] = 0.5f;
    if (spots) memcpy(ec.spot_color, spot_rgba[c], sizeof(ec.spot_color));
    ec.cfa_channel = 3;
    check(JxlEncoderSetExtraChannelInfo(enc, c, &ec));
    char name[64]; int length = snprintf(name, sizeof(name), "plane-%u-depth-%u", c, ec.bits_per_sample);
    check(JxlEncoderSetExtraChannelName(enc, c, name, (size_t)length));
  }
  JxlColorEncoding color; JxlColorEncodingSetToSRGB(&color, colors == 1);
  check(JxlEncoderSetColorEncoding(enc, &color));
  JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(enc, NULL);
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, floating ? 7 : effort));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, !vardct));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PATCHES, 0));
  if (floating) check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 0));
  if ((floating || wide_integer) && (strstr(name, "distributed") || strstr(name, "squeeze")))
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, 0));
  if (associated) check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1));
  if (progressive) check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1));
  if (progressive_dc) check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, progressive_dc));
  if (responsive) check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_RESPONSIVE, 1));
  if (vardct) {
    check(JxlEncoderSetFrameDistance(settings, 1.0f));
    for (uint32_t c=0; c<extras; ++c) check(JxlEncoderSetExtraChannelDistance(settings, c, 0.0f));
  } else check(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_RESAMPLING, resampling));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING, ec_resampling));
  size_t pixels = (size_t)width * height;
  float* data = malloc(pixels * colors * sizeof(float));
  if (!data) exit(2);
  for (uint32_t y=0; y<height; ++y) for (uint32_t x=0; x<width; ++x) for (uint32_t c=0; c<colors; ++c)
    data[((size_t)y*width+x)*colors+c] = wide_integer ? integer_sample(x,y,c,0,bits) :
      floating && !integer_primary ? floating_sample(x,y,c,0,1) :
      (float)(associated && x % 11 == 0 && y % 2 ? 23 : code(x,y,c,bits)) / (float)((1u<<bits)-1);
  JxlPixelFormat format = {colors, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  check(JxlEncoderAddImageFrame(settings, &format, data, pixels * colors * sizeof(float)));
  for (uint32_t c=0; c<extras; ++c) {
    uint32_t depth = alpha_only ? alpha_depth : depths[c];
    if (wide_integer && !alpha_only) depth = integer_extra_bits[c];
    if (floating) depth = alpha_only ? 32 : floating_extra_bits[c];
    for (uint32_t y=0; y<height; ++y) for (uint32_t x=0; x<width; ++x)
      data[(size_t)y*width+x] = wide_integer ? integer_sample(x,y,colors+c,0,depth) :
        floating && (alpha_only || floating_extra_exponents[c]) ?
        floating_sample(x,y,colors+c,0,0) : (float)code(x,y,colors+c,depth) / (float)((1u<<depth)-1);
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
  if (argc == 3 && !strcmp(argv[2], "--integer")) {
    wide_integer = 1; alpha_depth = 23;
    for (vardct = 0; vardct <= 1; ++vardct) {
      resampling = ec_resampling = 1; dimension_shift = responsive = associated = 0;
      generate(argv[1], "integer_rgb", 33, 7, 3, 17, 9, 6, 7, 0, 0);
      generate(argv[1], "integer_gray", 17, 9, 1, 24, 9, 8, 7, 0, 0);
      associated = 1;
      generate(argv[1], "integer_associated", 37, 9, 3, 23, 9, 5, 7, 0, 0);
      resampling = 2; ec_resampling = 8; dimension_shift = 1;
      generate(argv[1], "integer_resampled", 37, 17, 3, 24, 9, 7, 7, 0, 0);
      resampling = 4; ec_resampling = 4; dimension_shift = 2;
      generate(argv[1], "integer_resampled4", 17, 9, 1, 19, 1, 4, 7, 1, 0);
      resampling = 8; ec_resampling = 8; dimension_shift = 3;
      generate(argv[1], "integer_resampled8", 9, 1, 3, 17, 1, 2, 7, 1, 0);
      resampling = ec_resampling = 1; dimension_shift = associated = 0; responsive = 1;
      generate(argv[1], "integer_distributed", 257, 9, 3, 24, 9, 3, 7, 0, 1);
      responsive = 0;
      if (vardct) {
        progressive_dc = 1;
        generate(argv[1], "integer_progressive_dc", 65, 33, 3, 24, 0, 2, 7, 0, 1);
        progressive_dc = 0;
      }
    }
    return 0;
  }
  if (argc == 3 && !strcmp(argv[2], "--floating")) {
    floating = 1;
    for (vardct = 0; vardct <= 1; ++vardct) {
      resampling = ec_resampling = 1; dimension_shift = responsive = associated = 0;
      generate(argv[1], "float_rgb", 33, 7, 3, 16, 9, 6, 7, 0, 0);
      generate(argv[1], "float_gray", 17, 9, 1, 32, 9, 8, 1, 0, 0);
      associated = 1;
      generate(argv[1], "float_associated", 37, 9, 3, 24, 9, 5, 1, 0, 0);
      resampling = 2; ec_resampling = 8; dimension_shift = 1;
      generate(argv[1], "float_resampled", 37, 17, 3, 32, 9, 7, 1, 0, 0);
      resampling = 4; ec_resampling = 4; dimension_shift = 2;
      generate(argv[1], "float_resampled4", 17, 9, 1, 32, 1, 4, 1, 1, 0);
      resampling = 8; ec_resampling = 8; dimension_shift = 3;
      generate(argv[1], "float_resampled8", 9, 1, 3, 32, 1, 2, 1, 1, 0);
      resampling = ec_resampling = 1; dimension_shift = associated = 0; responsive = 1;
      if (!vardct) generate(argv[1], "float_global", 257, 9, 3, 32, 9, 3, 7, 0, 1);
      generate(argv[1], "float_distributed", 257, 9, 3, 32, 9, 3, 7, 0, 1);
      floating_narrow = 1;
      generate(argv[1], "float_squeeze", 257, 9, 3, 16, 9, 3, 7, 0, 1);
      floating_narrow = 0;
      responsive = 0;
      if (vardct) {
        progressive_dc = 1;
        // libjxl disables progressive DC whenever extra channels are present.
        generate(argv[1], "float_progressive_dc", 65, 33, 3, 16, 0, 2, 7, 0, 1);
        progressive_dc = 0;
      }
      integer_primary = 1;
      generate(argv[1], "float_integer_rgb", 33, 7, 3, 8, 9, 6, 7, 0, 0);
      associated = 1;
      generate(argv[1], "float_integer_gray", 17, 9, 1, 12, 9, 8, 7, 0, 0);
      integer_primary = associated = 0;
    }
    return 0;
  }
  if (argc == 3 && !strcmp(argv[2], "--spots")) {
    spots = 1;
    for (vardct = 0; vardct <= 1; ++vardct) {
      resampling = ec_resampling = 1; dimension_shift = responsive = associated = 0;
      generate(argv[1], "spots_rgb", 33, 7, 3, 12, 9, 6, 1, 0, 0);
      generate(argv[1], "spots_gray", 17, 9, 1, 16, 9, 8, 1, 0, 0);
      associated = 1;
      generate(argv[1], "spots_thin", 1, 9, 3, 8, 9, 5, 1, 0, 0);
      resampling = 2; ec_resampling = 8; dimension_shift = 1;
      generate(argv[1], "spots_resampled", 37, 17, 3, 12, 9, 7, 1, 0, 0);
      resampling = ec_resampling = 1; dimension_shift = associated = 0; responsive = 1;
      generate(argv[1], "spots_distributed", 257, 9, 3, 12, 9, 3, 7, 0, 1);
    }
    return 0;
  }
  if (argc == 3 && !strcmp(argv[2], "--associated")) {
    associated = 1;
    for (vardct = 0; vardct <= 1; ++vardct) {
      resampling = ec_resampling = 1; dimension_shift = responsive = 0;
      alpha_depth = 8;
      generate(argv[1], "associated_same", 33, 7, 3, 8, 1, 4, 1, 1, 0);
      alpha_depth = 5;
      generate(argv[1], "associated_rgb", 259, 9, 3, 8, 1, 6, 1, 1, 0);
      generate(argv[1], "associated_gray", 17, 257, 1, 16, 1, 8, 1, 1, 0);
      generate(argv[1], "associated_data", 33, 7, 3, 12, 9, 5, 1, 0, 0);
      generate(argv[1], "associated_thin", 1, 9, 3, 16, 1, 2, 1, 1, 0);
      resampling = 2; ec_resampling = 8; dimension_shift = 1;
      generate(argv[1], "associated_resampled", 259, 17, 3, 12, 1, 7, 1, 1, 0);
      resampling = ec_resampling = 1; dimension_shift = 0; responsive = 1;
      generate(argv[1], "associated_squeeze", 2051, 17, 3, 12, 1, 3, 7, 1, 1);
    }
    return 0;
  }
  if (argc == 3 && !strcmp(argv[2], "--resampled")) {
    for (vardct = 0; vardct <= 1; ++vardct) {
      ec_resampling = 2;
      generate(argv[1], "resampled_2", 517, 9, 3, 12, 1, 6, 1, 1, 0);
      ec_resampling = 4;
      generate(argv[1], "resampled_4", 37, 17, 1, 8, 9, 8, 1, 0, 0);
      ec_resampling = 8;
      generate(argv[1], "resampled_8", 2051, 9, 3, 16, 1, 5, 1, 1, 0);
      resampling = 2;
      generate(argv[1], "resampled_color", 259, 17, 3, 8, 1, 7, 1, 1, 0);
      resampling = 4; ec_resampling = 4;
      generate(argv[1], "resampled_color4", 17, 257, 1, 8, 1, 4, 1, 1, 0);
      resampling = 8; ec_resampling = 8;
      generate(argv[1], "resampled_color8", 9, 1, 3, 16, 1, 2, 1, 1, 0);
      resampling = 2; ec_resampling = 4; dimension_shift = 2;
      generate(argv[1], "shifted4", 37, 9, 3, 12, 1, 1, 1, 1, 0);
      resampling = 1; ec_resampling = 8; dimension_shift = 3;
      generate(argv[1], "shifted8", 2051, 9, 3, 8, 1, 7, 1, 1, 0);
      resampling = 1; ec_resampling = 2; dimension_shift = 1;
      generate(argv[1], "shifted", 37, 9, 3, 8, 1, 3, 1, 1, 0);
      dimension_shift = 0;
      responsive = 1;
      generate(argv[1], "resampled_squeeze", 2051, 17, 3, 12, 1, 6, 7, 1, 1);
      responsive = 0;
    }
    return 0;
  }
  if (argc != 2 && (argc != 3 || (strcmp(argv[2], "--vardct") && strcmp(argv[2], "--vardct-distributed")))) return 2;
  vardct = argc == 3;
  if (vardct && !strcmp(argv[2], "--vardct-distributed")) {
    generate(argv[1], "distributed_alpha", 257, 17, 3, 8, 1, 6, 1, 1, 0);
    generate(argv[1], "distributed_data", 517, 9, 1, 12, 9, 8, 1, 0, 0);
    generate(argv[1], "distributed_progressive", 259, 257, 3, 8, 1, 2, 7, 1, 1);
    generate(argv[1], "distributed_wide", 2049, 9, 1, 16, 1, 5, 1, 1, 0);
    responsive = 1;
    generate(argv[1], "distributed_squeeze", 2051, 259, 3, 12, 1, 7, 7, 1, 1);
    return 0;
  }
  if (vardct) {
    generate(argv[1], "data_only", 17, 1, 3, 8, 2, 6, 1, 0, 0);
    generate(argv[1], "rgb12", 33, 7, 3, 12, 9, 6, 1, 0, 0);
    generate(argv[1], "gray8", 33, 7, 1, 8, 9, 8, 1, 0, 0);
    generate(argv[1], "gray_alpha", 37, 9, 1, 16, 1, 5, 1, 1, 0);
    generate(argv[1], "rgba", 63, 9, 3, 8, 1, 3, 1, 1, 0);
    generate(argv[1], "transformed", 127, 129, 3, 12, 9, 7, 7, 0, 0);
    generate(argv[1], "rgba_progressive", 63, 9, 3, 8, 1, 2, 7, 1, 1);
    return 0;
  }
  generate(argv[1], "data_only", 17, 1, 3, 8, 2, 6, 1, 0, 0);
  generate(argv[1], "rgb12", 259, 17, 3, 12, 9, 6, 1, 0, 0);
  generate(argv[1], "gray8", 33, 7, 1, 8, 9, 8, 1, 0, 0);
  generate(argv[1], "gray_alpha", 257, 9, 1, 16, 1, 5, 1, 1, 0);
  generate(argv[1], "rgba", 259, 9, 3, 8, 1, 3, 1, 1, 0);
  generate(argv[1], "transformed", 515, 259, 3, 12, 9, 7, 7, 0, 0);
  return 0;
}
