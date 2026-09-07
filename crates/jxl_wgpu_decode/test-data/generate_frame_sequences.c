/* Offline conformance fixture generator. Production never links to libjxl.
 * cc generate_frame_sequences.c $(pkg-config --cflags --libs libjxl) -o /tmp/jxl-sequences
 * /tmp/jxl-sequences OUTPUT_DIRECTORY [259x17_RGB_JPEG]
 * Generated with libjxl 0.12.0. The test suite consumes checked-in .jxl.hex files.
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

static void generate(const char* dir, const char* name, uint32_t width, uint32_t height,
                     uint32_t channels, uint32_t bits, int orientation, int vardct,
                     int dc, int count) {
  JxlEncoder* enc = JxlEncoderCreate(NULL);
  int still = strcmp(name, "layered_still") == 0;
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = width; info.ysize = height;
  info.bits_per_sample = bits; info.num_color_channels = channels == 1 ? 1 : 3;
  info.num_extra_channels = channels == 4;
  info.alpha_bits = channels == 4 ? bits : 0;
  info.uses_original_profile = vardct <= 0;
  info.orientation = (JxlOrientation)orientation;
  info.have_animation = !still;
  info.animation.tps_numerator = 30000; info.animation.tps_denominator = 1001;
  info.animation.num_loops = strcmp(name, "modular_many") == 0 ? 0 : 3;
  info.animation.have_timecodes = !still;
  check(JxlEncoderSetBasicInfo(enc, &info));
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, channels == 1);
  check(JxlEncoderSetColorEncoding(enc, &color));
  size_t samples = (size_t)width * height * channels;
  size_t bytes = samples * (bits > 8 ? 2 : 1);
  void* pixels = malloc(bytes);
  uint32_t mask = (1u << bits) - 1;
  for (int frame = 0; frame < count; ++frame) {
    for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x) {
      uint32_t values[4] = {
        (613*x + 107*y + 43*(x^y) + 193*frame) & mask,
        ((153*x) ^ (271*y) ^ (79*frame)) & mask,
        (259*x + 307*y + 31*(x^y) + 131*frame) & mask,
        mask - ((181*x + 97*y + 17*frame) & (mask - 1))
      };
      for (uint32_t c = 0; c < channels; ++c) {
        size_t pos = ((size_t)y*width + x)*channels + c;
        if (bits > 8) ((uint16_t*)pixels)[pos] = values[c];
        else ((uint8_t*)pixels)[pos] = values[c];
      }
    }
    JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(enc, NULL);
    int jpeg_frame = vardct < 0 && frame % 3 == 1;
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, dc ? 4 : 1));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, vardct <= 0 && !jpeg_frame));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PATCHES, 0));
    if (vardct > 0) {
      check(JxlEncoderSetFrameDistance(settings, 2));
      check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1));
      check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, dc));
    } else if (!jpeg_frame) {
      check(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
      JxlBitDepth depth = { JXL_BIT_DEPTH_FROM_CODESTREAM, bits, 0 };
      check(JxlEncoderSetFrameBitDepth(settings, &depth));
    }
    JxlFrameHeader header;
    JxlEncoderInitFrameHeader(&header);
    header.duration = still || frame % 2 == 0 ? 0 : 3*frame + 1;
    header.timecode = still ? 0 : 0x01020000u + frame;
    if (frame == 1 && strcmp(name, "rejected_crop") == 0) {
      header.layer_info.have_crop = JXL_TRUE;
      header.layer_info.crop_x0 = -1;
      header.layer_info.xsize = width; header.layer_info.ysize = height;
    }
    if (frame == 1 && strcmp(name, "rejected_add") == 0) {
      header.layer_info.blend_info.blendmode = JXL_BLEND_ADD;
    }
    check(JxlEncoderSetFrameHeader(settings, &header));
    char frame_name[80];
    snprintf(frame_name, sizeof(frame_name), "frame-%d-\xe6\x99\x82\xe9\x96\x93", frame);
    check(JxlEncoderSetFrameName(settings, frame_name));
    JxlPixelFormat format = { channels, bits > 8 ? JXL_TYPE_UINT16 : JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0 };
    if (jpeg_frame) check(JxlEncoderAddJPEGFrame(settings, jpeg, jpeg_size));
    else check(JxlEncoderAddImageFrame(settings, &format, pixels, bytes));
  }
  free(pixels);
  JxlEncoderCloseInput(enc);
  char path[1024];
  snprintf(path, sizeof(path), "%s/sequence_%s.jxl.hex", dir, name);
  FILE* out = fopen(path, "w"); if (!out) exit(2);
  JxlEncoderStatus status;
  size_t written = 0;
  do {
    uint8_t buffer[16384]; uint8_t* next = buffer; size_t available = sizeof(buffer);
    status = JxlEncoderProcessOutput(enc, &next, &available);
    if (status != JXL_ENC_SUCCESS && status != JXL_ENC_NEED_MORE_OUTPUT) exit(3);
    for (size_t i = 0; i < sizeof(buffer)-available; ++i) {
      fprintf(out, "%02x", buffer[i]); if (++written % 32 == 0) fputc('\n', out);
    }
  } while (status == JXL_ENC_NEED_MORE_OUTPUT);
  if (written % 32) fputc('\n', out);
  fclose(out); JxlEncoderDestroy(enc);
  fprintf(stderr, "%s: %zu bytes\n", name, written);
}

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) return 2;
  generate(argv[1], "modular_gray", 259, 17, 1, 8, 6, 0, 0, 5);
  generate(argv[1], "modular_rgb12", 257, 9, 3, 12, 8, 0, 0, 5);
  generate(argv[1], "modular_rgba16", 33, 7, 4, 16, 2, 0, 0, 5);
  generate(argv[1], "modular_many", 1, 9, 1, 8, 5, 0, 0, 17);
  generate(argv[1], "layered_still", 37, 13, 1, 8, 7, 0, 0, 5);
  generate(argv[1], "vardct_rgb", 257, 33, 3, 8, 6, 1, 0, 5);
  generate(argv[1], "vardct_gray", 259, 17, 1, 8, 8, 1, 0, 5);
  generate(argv[1], "vardct_dc", 1024, 128, 3, 8, 6, 1, 2, 3);
  generate(argv[1], "rejected_crop", 19, 9, 1, 8, 1, 0, 0, 3);
  generate(argv[1], "rejected_add", 19, 9, 1, 8, 1, 0, 0, 3);
  if (argc == 3) {
    FILE* input = fopen(argv[2], "rb"); if (!input) return 2;
    fseek(input, 0, SEEK_END); long length = ftell(input); rewind(input);
    if (length <= 0) return 2;
    jpeg_size = (size_t)length; jpeg = malloc(jpeg_size);
    if (fread(jpeg, 1, jpeg_size, input) != jpeg_size) return 2;
    fclose(input);
    generate(argv[1], "mixed_jpeg_modular", 259, 17, 3, 8, 1, -1, 0, 5);
    free(jpeg);
  }
  return 0;
}
