/* Offline full-width integer fixtures. The public libjxl encoder accepts at most 24 integer
 * bits and routes source pixels through float. Store exact source words using binary32 Modular
 * coding, then let regenerate_integer serialize their integer image header. The resulting
 * codestream is independently decoded by libjxl, never by a production CPU fallback.
 * BITS COLORS ALPHA_BITS WIDTH HEIGHT PREDICTOR RCT OUTPUT.jxl OUTPUT.u32.hex
 */
#include <jxl/encode.h>
#include <jxl/color_encoding.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(JxlEncoderStatus status) {
  if (status != JXL_ENC_SUCCESS) { fprintf(stderr, "encode status %d\n", status); exit(2); }
}

static uint32_t sample(uint32_t x, uint32_t y, uint32_t c, unsigned bits) {
  uint32_t maximum = (1u << bits) - 1;
  uint32_t edge[] = {0, 1, maximum, maximum - 1, maximum / 2, maximum / 2 + 1,
      0x00ffffff, 0x01000001, 0x3fffffff, 0x40000001, 0x7fffff80, 0x7fffffc0};
  if (x < sizeof(edge) / sizeof(*edge)) return edge[(x + y + c) % 12] & maximum;
  uint32_t hash = x * 0x9e3779b9u + y * 0x85ebca6bu + c * 0xc2b2ae35u;
  hash ^= hash >> 16; hash *= 0x7feb352du; hash ^= hash >> 15;
  return hash & maximum;
}

static uint32_t widen(uint32_t word, unsigned bits) {
  unsigned mantissa_bits = bits - 8;
  uint32_t sign = (word >> (bits - 1)) << 31, magnitude = word & ((1u << (bits - 1)) - 1);
  if (!magnitude) return sign;
  int exponent = (int)(magnitude >> mantissa_bits);
  uint32_t mantissa = (magnitude & ((1u << mantissa_bits) - 1)) << (23 - mantissa_bits);
  if (exponent == 127) return sign | 0x7f800000u | mantissa;
  if (!exponent) {
    while (!(mantissa & 0x800000u)) { mantissa <<= 1; --exponent; }
    ++exponent; mantissa &= 0x7fffffu;
  }
  return sign | ((uint32_t)(exponent + 64) << 23) | mantissa;
}

int main(int argc, char** argv) {
  if (argc != 10) return 1;
  unsigned bits = (unsigned)atoi(argv[1]), colors = (unsigned)atoi(argv[2]);
  unsigned alpha = (unsigned)atoi(argv[3]), width = (unsigned)atoi(argv[4]), height = (unsigned)atoi(argv[5]);
  int predictor = atoi(argv[6]), rct = atoi(argv[7]);
  if (bits < 1 || bits > 31 || (colors != 1 && colors != 3) || alpha > 31 || !width || !height) return 1;
  unsigned channels = colors + (alpha != 0);
  size_t count = (size_t)width * height * channels;
  float* data = malloc(count * sizeof(float)); if (!data) return 2;
  FILE* words = fopen(argv[9], "w"); if (!words) return 3;
  for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) for (unsigned c = 0; c < channels; ++c) {
    uint32_t word = sample(x, y, c, c == colors ? alpha : bits);
    uint32_t encoded = rct ? widen(word, bits) : word;
    memcpy(data + ((size_t)y * width + x) * channels + c, &encoded, sizeof(encoded));
    fprintf(words, "%08x\n", word);
  }
  if (fclose(words)) return 3;
  JxlEncoder* enc = JxlEncoderCreate(NULL); if (!enc) return 2;
  JxlBasicInfo info; JxlEncoderInitBasicInfo(&info);
  info.xsize = width; info.ysize = height;
  info.bits_per_sample = rct ? bits : 32; info.exponent_bits_per_sample = rct ? 7 : 8;
  info.num_color_channels = colors; info.num_extra_channels = alpha != 0;
  info.alpha_bits = alpha ? 32 : 0; info.alpha_exponent_bits = alpha ? 8 : 0;
  info.uses_original_profile = JXL_TRUE;
  check(JxlEncoderSetCodestreamLevel(enc, 10)); check(JxlEncoderSetBasicInfo(enc, &info));
  JxlColorEncoding color; JxlColorEncodingSetToSRGB(&color, colors == 1);
  check(JxlEncoderSetColorEncoding(enc, &color));
  JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(enc, NULL);
  check(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 7));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PALETTE_COLORS, 0));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_RESPONSIVE, 0));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, 0));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, predictor));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE, rct));
  JxlPixelFormat format = {channels, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  check(JxlEncoderAddImageFrame(settings, &format, data, count * sizeof(float)));
  free(data); JxlEncoderCloseInput(enc);
  FILE* output = fopen(argv[8], "wb"); if (!output) return 3;
  JxlEncoderStatus status;
  do {
    uint8_t buffer[16384], *next = buffer; size_t available = sizeof(buffer);
    status = JxlEncoderProcessOutput(enc, &next, &available);
    if (status != JXL_ENC_SUCCESS && status != JXL_ENC_NEED_MORE_OUTPUT) return 2;
    size_t size = (size_t)(next - buffer);
    if (fwrite(buffer, 1, size, output) != size) return 3;
  } while (status != JXL_ENC_SUCCESS);
  if (fclose(output)) return 3;
  JxlEncoderDestroy(enc);
  return 0;
}
