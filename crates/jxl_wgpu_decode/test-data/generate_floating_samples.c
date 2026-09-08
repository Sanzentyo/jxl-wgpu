/* Offline libjxl 0.12 precision fixtures. Invoked by the regenerate_floating example.
 * encode BITS EXPONENT MODE OUTPUT.jxl; MODE is samples, header, or words.
 * decode INPUT.jxl prints the 120 independently decoded binary32 words as hex.
 * In words mode, a binary32 frame stores raw custom E=8 words. The Rust driver
 * transplants its frame into a matching custom-precision header: libjxl 0.12's
 * custom E=8 encoder incorrectly rounds binary32 subnormals, while its decoder
 * correctly supports them. No CPU codec is linked into the production decoder.
 */
#include <jxl/encode.h>
#include <jxl/decode.h>
#include <jxl/color_encoding.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(JxlEncoderStatus status) {
  if (status != JXL_ENC_SUCCESS) { fprintf(stderr, "encode status %d\n", status); exit(2); }
}

static uint32_t widen(uint32_t word, unsigned bits, unsigned exponent) {
  unsigned mantissa = bits - exponent - 1;
  uint32_t sign = (word >> (bits - 1)) ? 0x80000000u : 0;
  uint32_t magnitude = word & ((1u << (bits - 1)) - 1);
  if (!magnitude) return sign;
  int exp = (int)(magnitude >> mantissa);
  uint32_t fraction = (magnitude & ((1u << mantissa) - 1)) << (23 - mantissa);
  if (exp == (int)((1u << exponent) - 1)) return sign | 0x7f800000u | fraction;
  if (!exp && exponent < 8) {
    while (!(fraction & 0x800000u)) { fraction <<= 1; --exp; }
    ++exp; fraction &= 0x7fffffu;
  }
  exp += 127 - (int)((1u << (exponent - 1)) - 1);
  return sign | ((uint32_t)exp << 23) | fraction;
}

static void encode(unsigned bits, unsigned exponent, const char* mode, const char* path) {
  int words = !strcmp(mode, "words"), header = !strcmp(mode, "header");
  if ((!words && !header && strcmp(mode, "samples")) || exponent < 2 || exponent > 8
      || bits < exponent + 3 || bits > exponent + 24 || (words && exponent != 8)) exit(1);
  uint32_t raw[] = {0, 0x80000000, 1, 0x80000001, 0x7fffff, 0x807fffff,
      0x800000, 0x80800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
      0x7fc00001, 0xffc01234, 0x7f812345, 0xff812345, 0x3f800000, 0xbf800000,
      0x3e800000, 0xbe800000, 0x40200000, 0xc0200000, 0x3a800000, 0xb9800000};
  if (bits != 32) {
    unsigned mantissa = bits - exponent - 1;
    uint32_t sign = 1u << (bits - 1), em = (1u << exponent) - 1, mm = (1u << mantissa) - 1;
    uint32_t codes[] = {0, 1, mm, 1u << mantissa, ((em - 1) << mantissa) | mm, em << mantissa,
      (em << mantissa) | (1u << (mantissa - 1)) | 1u, (em << mantissa) | 1u,
      ((1u << (exponent - 1)) - 1) << mantissa, 2u << mantissa, mm >> 1, (em - 1) << mantissa};
    for (unsigned i = 0; i < 24; ++i) {
      uint32_t word = codes[i / 2] | (i % 2 ? sign : 0);
      raw[i] = words ? word : widen(word, bits, exponent);
    }
  }
  if (header) for (unsigned i = 0; i < 24; ++i) raw[i] = 0x3f800000;
  float pixels[120];
  for (unsigned i = 0; i < 120; ++i) memcpy(&pixels[i], &raw[(i * 7 + i / 24) % 24], 4);
  JxlEncoder* enc = JxlEncoderCreate(NULL); if (!enc) exit(2);
  JxlBasicInfo info; JxlEncoderInitBasicInfo(&info);
  info.xsize = 24; info.ysize = 5; info.bits_per_sample = words ? 32 : bits;
  info.exponent_bits_per_sample = exponent; info.num_color_channels = 1;
  info.uses_original_profile = JXL_TRUE;
  check(JxlEncoderSetCodestreamLevel(enc, 10)); check(JxlEncoderSetBasicInfo(enc, &info));
  JxlColorEncoding color; JxlColorEncodingSetToSRGB(&color, JXL_TRUE);
  check(JxlEncoderSetColorEncoding(enc, &color));
  JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(enc, NULL);
  check(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 7));
  if (words) {
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PALETTE_COLORS, 0));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_RESPONSIVE, 0));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 0));
  }
  JxlPixelFormat format = {1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  check(JxlEncoderAddImageFrame(settings, &format, pixels, sizeof(pixels)));
  JxlEncoderCloseInput(enc);
  uint8_t encoded[65536], *next = encoded; size_t available = sizeof(encoded);
  check(JxlEncoderProcessOutput(enc, &next, &available));
  FILE* output = fopen(path, "wb"); if (!output) exit(3);
  size_t size = (size_t)(next - encoded);
  if (fwrite(encoded, 1, size, output) != size || fclose(output)) exit(3);
  JxlEncoderDestroy(enc);
}

static void decode(const char* path) {
  FILE* input = fopen(path, "rb"); if (!input) exit(3);
  uint8_t encoded[65536]; size_t size = fread(encoded, 1, sizeof(encoded), input);
  if (ferror(input) || !feof(input) || fclose(input)) exit(3);
  JxlDecoder* dec = JxlDecoderCreate(NULL); if (!dec) exit(4);
  if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) != JXL_DEC_SUCCESS
      || JxlDecoderSetInput(dec, encoded, size) != JXL_DEC_SUCCESS) exit(4);
  JxlDecoderCloseInput(dec);
  JxlPixelFormat format = {1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  float pixels[120]; int complete = 0;
  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_SUCCESS) { if (complete != 1) exit(4); break; }
    if (status == JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info;
      if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS
          || info.xsize != 24 || info.ysize != 5 || info.num_color_channels != 1) exit(4);
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      if (JxlDecoderSetImageOutBuffer(dec, &format, pixels, sizeof(pixels)) != JXL_DEC_SUCCESS) exit(4);
    } else if (status == JXL_DEC_FULL_IMAGE) ++complete;
    else { fprintf(stderr, "decode status %d\n", status); exit(4); }
  }
  for (unsigned i = 0; i < 120; ++i) {
    uint32_t word; memcpy(&word, &pixels[i], 4); printf("%08x\n", word);
  }
  JxlDecoderDestroy(dec);
}

int main(int argc, char** argv) {
  if (argc == 6 && !strcmp(argv[1], "encode"))
    encode((unsigned)atoi(argv[2]), (unsigned)atoi(argv[3]), argv[4], argv[5]);
  else if (argc == 3 && !strcmp(argv[1], "decode")) decode(argv[2]);
  else return 1;
  return 0;
}
