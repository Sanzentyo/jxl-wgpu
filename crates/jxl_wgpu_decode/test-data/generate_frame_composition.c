/* Offline libjxl 0.12.0 interoperability fixtures; never linked by production.
 * cc generate_frame_composition.c $(pkg-config --cflags --libs libjxl) -o /tmp/jxl-composition
 * /tmp/jxl-composition OUTPUT_DIRECTORY [259x17_RGB_JPEG|--associated]
 */
#include <jxl/encode.h>
#include <jxl/color_encoding.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(JxlEncoderStatus status) { if (status != JXL_ENC_SUCCESS) exit(1); }
static uint8_t* jpeg;
static size_t jpeg_size;
static int associated;
/* Offline standalone headers for testing progressive composition without FlushImage blending. */
static int isolated_layer = -1;

typedef struct {
  int x, y;
  uint32_t width, height, duration, save, color_source, alpha_source;
  JxlBlendMode color, alpha;
  int clamp;
} Layer;

static void generate(const char* dir, const char* name, uint32_t width, uint32_t height,
    uint32_t channels, uint32_t bits, int orientation, int vardct, int dc, int still) {
  const Layer layers[] = {
    {0, 0, width, height, 1, 1, 0, 0, JXL_BLEND_REPLACE, JXL_BLEND_REPLACE, 0},
    {-2, 3, 20, 12, 0, 2, 1, 1, JXL_BLEND_BLEND, JXL_BLEND_REPLACE, 1},
    {(int)width-9, -2, 23, 13, 2, 1, 2, 1, JXL_BLEND_ADD, JXL_BLEND_MULADD, 0},
    {0, 0, width, height, 1, 2, 1, 2, JXL_BLEND_MUL, JXL_BLEND_ADD, 0},
    {-4, -3, width+8, height+6, 0, 1, 2, 1, JXL_BLEND_MULADD, JXL_BLEND_REPLACE, 0},
    {(int)width+1, -100, 7, 2, 1, 2, 1, 2, JXL_BLEND_BLEND, JXL_BLEND_MUL, 1},
    {0, 0, width, height, 1, 1, 2, 1, JXL_BLEND_BLEND, JXL_BLEND_REPLACE, 0},
    {2, 1, 11, 7, 0, 2, 0, 3, JXL_BLEND_ADD, JXL_BLEND_ADD, 0},
    {15, 8, 23, 17, 2, 0, 2, 1, JXL_BLEND_MUL, JXL_BLEND_BLEND, 1},
  };
  JxlEncoder* enc = JxlEncoderCreate(NULL);
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = isolated_layer < 0 ? width : layers[isolated_layer].width; info.ysize = isolated_layer < 0 ? height : layers[isolated_layer].height; info.bits_per_sample = bits;
  int different_alpha = channels == 2 || strcmp(name, "rgba_mixed_depth") == 0 || associated;
  uint32_t color_channels = channels <= 2 ? 1 : 3;
  int has_alpha = channels != color_channels;
  uint32_t alpha_bits = has_alpha ? (different_alpha ? 5 : bits) : 0;
  info.num_color_channels = color_channels;
  info.num_extra_channels = has_alpha; info.alpha_bits = alpha_bits;
  info.alpha_premultiplied = associated;
  info.uses_original_profile = vardct <= 0; info.orientation = (JxlOrientation)orientation;
  info.have_animation = !still && isolated_layer < 0;
  info.animation.tps_numerator = 30000; info.animation.tps_denominator = 1001;
  info.animation.num_loops = 2; info.animation.have_timecodes = !still && isolated_layer < 0;
  check(JxlEncoderSetBasicInfo(enc, &info));
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, color_channels == 1);
  check(JxlEncoderSetColorEncoding(enc, &color));
  for (uint32_t frame = 0; frame < sizeof(layers)/sizeof(*layers); ++frame) {
    if (isolated_layer >= 0 && frame != (uint32_t)isolated_layer) continue;
    const Layer* layer = &layers[frame];
    size_t bytes = (size_t)layer->width * layer->height * channels * (different_alpha ? 4 : bits > 8 ? 2 : 1);
    void* pixels = malloc(bytes);
    uint32_t mask = (1u << bits) - 1;
    uint32_t alpha_mask = has_alpha ? (1u << alpha_bits) - 1 : mask;
    for (uint32_t y = 0; y < layer->height; ++y) for (uint32_t x = 0; x < layer->width; ++x) {
      const uint32_t values[4] = {
        (613*x + 107*y + 43*(x^y) + 193*frame) & mask,
        ((153*x) ^ (271*y) ^ (79*frame)) & mask,
        (259*x + 307*y + 31*(x^y) + 131*frame) & mask,
        x % 5 == 0 ? 0 : x % 5 == 1 ? alpha_mask : (181*x + 97*y + 193*frame) & alpha_mask,
      };
      for (uint32_t c = 0; c < channels; ++c) {
        size_t pos = ((size_t)y*layer->width + x)*channels + c;
        uint32_t canonical = channels == 2 && c == 1 ? 3 : c;
        if (different_alpha) ((float*)pixels)[pos] = (float)values[canonical] / (float)(canonical == 3 ? alpha_mask : mask);
        else if (bits > 8) ((uint16_t*)pixels)[pos] = values[c];
        else ((uint8_t*)pixels)[pos] = values[c];
      }
    }
    JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(enc, NULL);
    int jpeg_frame = vardct < 0 && frame == 3;
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, dc ? 4 : 1));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, vardct <= 0 && !jpeg_frame));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PATCHES, 0));
    if (associated) check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1));
    if (vardct > 0) {
      check(JxlEncoderSetFrameDistance(settings, 2));
      check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1));
      /* libjxl 0.12 cannot emit recursive DC for these tiny cropped layers. */
      int layer_dc = layer->width == width && layer->height == height ? dc : 0;
      check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, layer_dc));
    } else if (!jpeg_frame) {
      check(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
      if (!different_alpha) {
        JxlBitDepth depth = {JXL_BIT_DEPTH_FROM_CODESTREAM, bits, 0};
        check(JxlEncoderSetFrameBitDepth(settings, &depth));
      }
    }
    JxlFrameHeader header;
    JxlEncoderInitFrameHeader(&header);
    header.duration = still ? 0 : layer->duration;
    header.timecode = still ? 0 : 0x01020000u + frame;
    header.layer_info.have_crop = layer->x || layer->y || layer->width != width || layer->height != height;
    header.layer_info.crop_x0 = layer->x; header.layer_info.crop_y0 = layer->y;
    header.layer_info.xsize = layer->width; header.layer_info.ysize = layer->height;
    header.layer_info.save_as_reference = layer->save;
    header.layer_info.blend_info.blendmode = layer->color;
    header.layer_info.blend_info.source = layer->color_source;
    header.layer_info.blend_info.alpha = 0; header.layer_info.blend_info.clamp = layer->clamp;
    if (frame == 3 && strcmp(name, "gray_clamp") == 0) header.layer_info.blend_info.clamp = JXL_TRUE;
    if (isolated_layer >= 0) JxlEncoderInitFrameHeader(&header);
    check(JxlEncoderSetFrameHeader(settings, &header));
    if (has_alpha) {
      JxlBlendInfo alpha = header.layer_info.blend_info;
      if (isolated_layer < 0) {
        alpha.blendmode = layer->alpha; alpha.source = layer->alpha_source;
      }
      check(JxlEncoderSetExtraChannelBlendInfo(settings, 0, &alpha));
    }
    char frame_name[64];
    snprintf(frame_name, sizeof(frame_name), "composed-%u", frame);
    check(JxlEncoderSetFrameName(settings, frame_name));
    JxlPixelFormat format = {channels, different_alpha ? JXL_TYPE_FLOAT : bits > 8 ? JXL_TYPE_UINT16 : JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    if (jpeg_frame) check(JxlEncoderAddJPEGFrame(settings, jpeg, jpeg_size));
    else check(JxlEncoderAddImageFrame(settings, &format, pixels, bytes));
    free(pixels);
  }
  JxlEncoderCloseInput(enc);
  char path[1024]; snprintf(path, sizeof(path), "%s/composition_%s.jxl.hex", dir, name);
  if (isolated_layer >= 0) snprintf(path, sizeof(path), "%s/composition_%s_layer%d.jxl.hex", dir, name, isolated_layer);
  FILE* out = fopen(path, "w"); if (!out) exit(2);
  JxlEncoderStatus status; size_t written = 0;
  do {
    uint8_t buffer[16384]; uint8_t* next = buffer; size_t available = sizeof(buffer);
    status = JxlEncoderProcessOutput(enc, &next, &available);
    if (status != JXL_ENC_SUCCESS && status != JXL_ENC_NEED_MORE_OUTPUT) {
      fprintf(stderr, "%s: encoder error %d\n", name, JxlEncoderGetError(enc)); exit(3);
    }
    for (size_t i = 0; i < sizeof(buffer)-available; ++i) {
      fprintf(out, "%02x", buffer[i]); if (++written % 32 == 0) fputc('\n', out);
    }
  } while (status == JXL_ENC_NEED_MORE_OUTPUT);
  if (written % 32) fputc('\n', out);
  fclose(out); JxlEncoderDestroy(enc);
  fprintf(stderr, "%s: %zu bytes\n", name, written);
}

int main(int argc, char** argv) {
  if (argc == 3 && !strcmp(argv[2], "--progressive-layers")) {
    for (isolated_layer = 0; isolated_layer < 9; ++isolated_layer) {
      generate(argv[1], "vardct", 259, 17, 3, 8, 1, 1, 0, 1);
      generate(argv[1], "vardct_gray", 37, 13, 1, 8, 1, 1, 0, 1);
      generate(argv[1], "vardct_dc", 1024, 128, 3, 8, 1, 1, 2, 1);
      associated = 1;
      generate(argv[1], "associated_vardct", 259, 17, 4, 12, 1, 1, 0, 1);
      associated = 0;
    }
    return 0;
  }
  if (argc == 3 && !strcmp(argv[2], "--associated")) {
    associated = 1;
    generate(argv[1], "associated_rgb", 259, 17, 4, 12, 6, 0, 0, 0);
    generate(argv[1], "associated_gray", 37, 9, 2, 16, 8, 0, 0, 0);
    generate(argv[1], "associated_vardct", 259, 17, 4, 12, 5, 1, 0, 0);
    return 0;
  }
  if (argc != 2 && argc != 3) return 2;
  generate(argv[1], "gray_alpha", 259, 17, 2, 16, 8, 0, 0, 0);
  generate(argv[1], "rgba_mixed_depth", 33, 7, 4, 12, 6, 0, 0, 0);
  generate(argv[1], "gray", 259, 17, 1, 8, 6, 0, 0, 0);
  generate(argv[1], "gray_clamp", 259, 17, 1, 8, 6, 0, 0, 0);
  generate(argv[1], "rgb12", 257, 9, 3, 12, 8, 0, 0, 0);
  generate(argv[1], "rgba8", 33, 7, 4, 8, 5, 0, 0, 0);
  generate(argv[1], "rgba16", 33, 7, 4, 16, 2, 0, 0, 0);
  generate(argv[1], "still", 37, 13, 1, 8, 7, 0, 0, 1);
  generate(argv[1], "vardct", 259, 17, 3, 8, 6, 1, 0, 0);
  generate(argv[1], "vardct_gray", 37, 13, 1, 8, 8, 1, 0, 0);
  generate(argv[1], "vardct_dc", 1024, 128, 3, 8, 6, 1, 2, 0);
  if (argc == 3) {
    FILE* input = fopen(argv[2], "rb"); if (!input) return 2;
    fseek(input, 0, SEEK_END); long length = ftell(input); rewind(input);
    if (length <= 0) return 2;
    jpeg_size = (size_t)length; jpeg = malloc(jpeg_size);
    if (fread(jpeg, 1, jpeg_size, input) != jpeg_size) return 2;
    fclose(input);
    generate(argv[1], "mixed", 259, 17, 3, 8, 4, -1, 0, 0);
    free(jpeg);
  }
  return 0;
}
