// Offline libjxl seeds. The Rust generator reframes the unchanged entropy into LF chains.
#include <jxl/encode.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void check(JxlEncoderStatus status) {
  if (status != JXL_ENC_SUCCESS) abort();
}

static void generate(const char* directory, const char* name, uint32_t width,
                     uint32_t height, int modular, uint32_t extras) {
  JxlEncoder* encoder = JxlEncoderCreate(NULL);
  JxlBasicInfo info; JxlEncoderInitBasicInfo(&info);
  info.xsize = width; info.ysize = height; info.bits_per_sample = 8;
  info.num_color_channels = 3; info.num_extra_channels = extras;
  info.uses_original_profile = JXL_FALSE;
  check(JxlEncoderSetBasicInfo(encoder, &info));
  for (uint32_t c = 0; c < extras; ++c) {
    JxlExtraChannelInfo ec;
    JxlEncoderInitExtraChannelInfo(c ? JXL_CHANNEL_DEPTH : JXL_CHANNEL_ALPHA, &ec);
    ec.bits_per_sample = 8;
    check(JxlEncoderSetExtraChannelInfo(encoder, c, &ec));
  }
  JxlColorEncoding color; JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
  check(JxlEncoderSetColorEncoding(encoder, &color));
  JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(encoder, NULL);
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, modular));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 7));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EPF, 0));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_GABORISH, 0));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PATCHES, 0));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_RESAMPLING, 1));
  check(JxlEncoderSetFrameDistance(settings, 1));
  for (uint32_t c = 0; c < extras; ++c) {
    check(JxlEncoderSetExtraChannelDistance(settings, c, 0));
  }
  const size_t size = (size_t)width * height;
  uint8_t* pixels = malloc(size * 3); if (!pixels) abort();
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      pixels[(y * width + x) * 3] = (uint8_t)(13 * x + 7 * y);
      pixels[(y * width + x) * 3 + 1] = (uint8_t)((3 * x) ^ (11 * y));
      pixels[(y * width + x) * 3 + 2] = (uint8_t)(5 * x + 17 * y);
    }
  }
  JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  check(JxlEncoderAddImageFrame(settings, &format, pixels, size * 3));
  for (uint32_t c = 0; c < extras; ++c) {
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        pixels[y * width + x] = (uint8_t)((c ? 31 : 11) * x + (c ? 3 : 17) * y);
      }
    }
    JxlPixelFormat plane = {1, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    check(JxlEncoderSetExtraChannelBuffer(settings, &plane, pixels, size, c));
  }
  free(pixels);
  JxlEncoderCloseInput(encoder);
  uint8_t output[65536]; uint8_t* next = output; size_t available = sizeof(output);
  check(JxlEncoderProcessOutput(encoder, &next, &available));
  char path[4096];
  if (snprintf(path, sizeof(path), "%s/%s.jxl", directory, name) >= (int)sizeof(path)) abort();
  FILE* file = fopen(path, "wb"); if (!file) abort();
  if (fwrite(output, 1, sizeof(output) - available, file) != sizeof(output) - available) abort();
  if (fclose(file)) abort();
  JxlEncoderDestroy(encoder);
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  generate(argv[1], "modular_root", 9, 5, 1, 2);
  generate(argv[1], "vardct_root", 9, 5, 0, 2);
  generate(argv[1], "modular_root2", 2, 1, 1, 2);
  generate(argv[1], "vardct_root2", 2, 1, 0, 2);
  generate(argv[1], "extras", 65, 33, 0, 2);
  return 0;
}
