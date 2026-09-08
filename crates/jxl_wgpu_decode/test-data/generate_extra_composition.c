/* Offline libjxl 0.12.0 oracle inputs, never linked by the production decoder.
 * cc generate_extra_composition.c $(pkg-config --cflags --libs libjxl) -o /tmp/jxl-extra-composition
 * /tmp/jxl-extra-composition OUTPUT_DIRECTORY [--floating]
 */
#include <jxl/encode.h>
#include <jxl/color_encoding.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "floating_samples.h"
#include "integer_samples.h"

static int floating;
static int lossy;
static int wide_integer;

static void check(JxlEncoderStatus status) { if (status != JXL_ENC_SUCCESS) exit(1); }
static const JxlExtraChannelType types[] = {
  JXL_CHANNEL_DEPTH, JXL_CHANNEL_SELECTION_MASK, JXL_CHANNEL_ALPHA,
  JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_CFA, JXL_CHANNEL_THERMAL,
  JXL_CHANNEL_BLACK, JXL_CHANNEL_OPTIONAL, JXL_CHANNEL_ALPHA,
};
static const uint32_t depths[] = {16, 1, 7, 12, 4, 8, 6, 10, 15};
typedef struct {
  int x, y;
  uint32_t width, height, duration, save, source;
  JxlBlendMode mode;
} Layer;

static float sample(uint32_t x, uint32_t y, uint32_t channel, uint32_t frame, uint32_t bits) {
  uint32_t mask = (1u << bits) - 1;
  uint32_t value = (193*x + 317*y + 97*channel + (x^y)*(23+channel) + frame*(71+channel)) & mask;
  if (x % 11 == 0) value = 0;
  if (x % 11 == 1) value = mask;
  return (float)value / (float)mask;
}

static void generate(const char* dir, const char* name, uint32_t width, uint32_t height,
    uint32_t colors, uint32_t bits, uint32_t extras, int orientation, int vardct,
    int responsive, int resampling, int associated) {
  JxlEncoder* enc = JxlEncoderCreate(NULL);
  JxlBasicInfo info; JxlEncoderInitBasicInfo(&info);
  info.xsize = width; info.ysize = height; info.bits_per_sample = bits;
  if (floating) info.exponent_bits_per_sample = floating_exponent(bits);
  info.num_color_channels = colors; info.num_extra_channels = extras;
  info.uses_original_profile = !vardct && !lossy; info.orientation = (JxlOrientation)orientation;
  info.have_animation = JXL_TRUE;
  info.animation.tps_numerator = 30000; info.animation.tps_denominator = 1001;
  info.animation.num_loops = 2; info.animation.have_timecodes = JXL_TRUE;
  if (floating || wide_integer) check(JxlEncoderSetCodestreamLevel(enc, 10));
  check(JxlEncoderSetBasicInfo(enc, &info));
  for (uint32_t c = 0; c < extras; ++c) {
    JxlExtraChannelInfo extra; JxlEncoderInitExtraChannelInfo(types[c], &extra);
    extra.bits_per_sample = depths[c]; extra.dim_shift = resampling ? 1 : 0;
    if (wide_integer) extra.bits_per_sample = integer_extra_bits[c];
    if (floating) {
      extra.bits_per_sample = floating_extra_bits[c];
      extra.exponent_bits_per_sample = floating_extra_exponents[c];
      if (resampling && extra.exponent_bits_per_sample) {
        extra.bits_per_sample = 32; extra.exponent_bits_per_sample = 8;
      }
    }
    extra.alpha_premultiplied = c == 2 ? associated : c == 8 ? !associated : 0;
    extra.spot_color[0] = 0.25f; extra.spot_color[1] = 0.5f;
    extra.spot_color[2] = 0.75f; extra.spot_color[3] = 0.5f;
    extra.cfa_channel = 3;
    check(JxlEncoderSetExtraChannelInfo(enc, c, &extra));
    char name[64]; int n = snprintf(name, sizeof(name), "composed-plane-%u-depth-%u", c, wide_integer ? extra.bits_per_sample : depths[c]);
    check(JxlEncoderSetExtraChannelName(enc, c, name, (size_t)n));
  }
  JxlColorEncoding color; JxlColorEncodingSetToSRGB(&color, colors == 1);
  check(JxlEncoderSetColorEncoding(enc, &color));
  const Layer layers[] = {
    {0, 0, width, height, 1, 1, 0, JXL_BLEND_REPLACE},
    {-2, 3, 20, 12, 0, 2, 1, JXL_BLEND_BLEND},
    {(int)width-9, -2, 23, 13, 2, 1, 2, JXL_BLEND_ADD},
    {0, 0, width, height, 1, 2, 1, JXL_BLEND_MUL},
    {-4, -3, width+8, height+6, 0, 1, 2, JXL_BLEND_MULADD},
    {(int)width+1, -100, 7, 2, 1, 2, 1, JXL_BLEND_BLEND},
    {0, 0, width, height, 1, 1, 2, JXL_BLEND_BLEND},
    {2, 1, 11, 7, 0, 2, 0, JXL_BLEND_ADD},
    {15, 8, 23, 17, 2, 0, 2, JXL_BLEND_MUL},
  };
  for (uint32_t frame = 0; frame < sizeof(layers)/sizeof(*layers); ++frame) {
    const Layer* layer = &layers[frame];
    JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(enc, NULL);
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, (floating || responsive) ? 7 : 1));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, !vardct));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PATCHES, 0));
    if (floating) check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 0));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1));
    if (responsive) check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_RESPONSIVE, 1));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_RESAMPLING, resampling ? 2 : 1));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING, resampling ? 8 : 1));
    if (vardct || lossy) check(JxlEncoderSetFrameDistance(settings, 2));
    else check(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
    for (uint32_t c = 0; c < extras; ++c) check(JxlEncoderSetExtraChannelDistance(settings, c, 0));
    JxlFrameHeader header; JxlEncoderInitFrameHeader(&header);
    header.duration = layer->duration; header.timecode = 0x01030000u + frame;
    header.layer_info.have_crop = layer->x || layer->y || layer->width != width || layer->height != height;
    header.layer_info.crop_x0 = layer->x; header.layer_info.crop_y0 = layer->y;
    header.layer_info.xsize = layer->width; header.layer_info.ysize = layer->height;
    header.layer_info.save_as_reference = layer->save;
    header.layer_info.blend_info.blendmode = layer->mode;
    header.layer_info.blend_info.source = layer->source;
    header.layer_info.blend_info.alpha = extras > 2 ? (frame % 2 ? 8 : 2) : 1;
    header.layer_info.blend_info.clamp = frame % 2;
    check(JxlEncoderSetFrameHeader(settings, &header));
    for (uint32_t c = 0; c < extras; ++c) {
      JxlBlendInfo blend = header.layer_info.blend_info;
      blend.blendmode = frame ? (JxlBlendMode)((frame + c) % 5) : JXL_BLEND_REPLACE;
      blend.source = frame ? (layer->source + c % 3) % 4 : 0;
      blend.alpha = extras > 2 ? ((frame+c) % 2 ? 8 : 2) : (frame+c) % extras;
      blend.clamp = (frame+c) % 2;
      check(JxlEncoderSetExtraChannelBlendInfo(settings, c, &blend));
    }
    char name[64]; snprintf(name, sizeof(name), "extra-composed-%u", frame);
    check(JxlEncoderSetFrameName(settings, name));
    size_t pixels = (size_t)layer->width * layer->height;
    float* data = malloc(pixels * colors * sizeof(float)); if (!data) exit(2);
    for (uint32_t y=0; y<layer->height; ++y) for (uint32_t x=0; x<layer->width; ++x)
      for (uint32_t c=0; c<colors; ++c) data[((size_t)y*layer->width+x)*colors+c] =
        wide_integer ? integer_sample(x,y,c,frame,bits) :
          floating ? floating_sample(x,y,c,frame,1) : sample(x,y,c,frame,bits);
    JxlPixelFormat format = {colors, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
    check(JxlEncoderAddImageFrame(settings, &format, data, pixels * colors * sizeof(float)));
    for (uint32_t c=0; c<extras; ++c) {
      for (uint32_t y=0; y<layer->height; ++y) for (uint32_t x=0; x<layer->width; ++x)
        data[(size_t)y*layer->width+x] = wide_integer ? integer_sample(x,y,colors+c,frame,integer_extra_bits[c]) :
          floating ? (floating_extra_exponents[c] ?
          floating_sample(x,y,colors+c,frame,0) : sample(x,y,colors+c,frame,floating_extra_bits[c])) :
          sample(x,y,colors+c,frame,depths[c]);
      check(JxlEncoderSetExtraChannelBuffer(settings, &format, data, pixels * sizeof(float), c));
    }
    free(data);
  }
  JxlEncoderCloseInput(enc);
  char path[1024]; snprintf(path, sizeof(path), "%s/composition_extras_%s.jxl.hex", dir, name);
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
  if (argc == 3 && !strcmp(argv[2], "--lossy")) {
    lossy = 1;
    generate(argv[1], "lossy_rgb", 37, 17, 3, 12, 9, 6, 0, 0, 0, 0);
    generate(argv[1], "lossy_gray", 37, 17, 1, 16, 9, 8, 0, 0, 0, 0);
    generate(argv[1], "lossy_resampled", 37, 17, 3, 16, 9, 2, 0, 0, 1, 0);
    floating = 1;
    generate(argv[1], "lossy_float", 37, 17, 3, 32, 9, 3, 0, 0, 0, 0);
    return 0;
  }
  if (argc == 3 && !strcmp(argv[2], "--integer")) {
    wide_integer = 1;
    generate(argv[1], "integer_rgb", 37, 17, 3, 24, 9, 6, 0, 0, 0, 0);
    generate(argv[1], "integer_gray", 37, 9, 1, 23, 9, 8, 0, 0, 0, 1);
    generate(argv[1], "integer_vardct", 37, 17, 3, 20, 9, 5, 1, 0, 0, 0);
    generate(argv[1], "integer_resampled", 37, 17, 3, 24, 9, 7, 0, 0, 1, 1);
    generate(argv[1], "integer_vardct_resampled", 37, 17, 3, 24, 9, 2, 1, 0, 1, 0);
    return 0;
  }
  if (argc == 3 && !strcmp(argv[2], "--floating")) {
    floating = 1;
    generate(argv[1], "float_rgb", 37, 17, 3, 16, 9, 6, 0, 0, 0, 0);
    generate(argv[1], "float_gray", 37, 9, 1, 32, 9, 8, 0, 0, 0, 1);
    generate(argv[1], "float_vardct", 37, 17, 3, 24, 9, 5, 1, 0, 0, 0);
    generate(argv[1], "float_resampled", 37, 17, 3, 32, 9, 7, 0, 0, 1, 1);
    generate(argv[1], "float_vardct_resampled", 37, 17, 3, 32, 9, 2, 1, 0, 1, 0);
    return 0;
  }
  if (argc != 2) return 2;
  generate(argv[1], "rgb", 259, 17, 3, 12, 9, 6, 0, 0, 0, 0);
  generate(argv[1], "gray", 37, 9, 1, 16, 9, 8, 0, 0, 0, 1);
  generate(argv[1], "vardct", 259, 17, 3, 12, 9, 5, 1, 0, 0, 0);
  generate(argv[1], "distributed", 2051, 17, 3, 12, 9, 3, 1, 1, 0, 1);
  generate(argv[1], "resampled", 259, 17, 3, 12, 9, 7, 0, 0, 1, 1);
  generate(argv[1], "vardct_resampled", 259, 17, 3, 12, 9, 2, 1, 0, 1, 0);
  generate(argv[1], "data", 33, 7, 3, 8, 2, 4, 0, 0, 0, 0);
  return 0;
}
