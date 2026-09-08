// Offline libjxl seeds; the Rust conformance test reframes their unchanged entropy as LF level 2.
#include <jxl/encode.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void check(JxlEncoderStatus status) {
  if (status != JXL_ENC_SUCCESS) abort();
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  for (uint32_t factor = 1; factor <= 8; factor *= 2) {
    const uint32_t width = 16 / factor, height = (2 + factor - 1) / factor;
    uint8_t pixels[16 * 2 * 3];
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        pixels[(y * width + x) * 3] = (uint8_t)(13 * x * factor + 7 * y);
        pixels[(y * width + x) * 3 + 1] = (uint8_t)((3 * x * factor) ^ (11 * y));
        pixels[(y * width + x) * 3 + 2] = (uint8_t)(5 * x * factor + 17 * y);
      }
    }
    JxlEncoder* encoder = JxlEncoderCreate(NULL);
    JxlBasicInfo info; JxlEncoderInitBasicInfo(&info);
    info.xsize = width; info.ysize = height; info.bits_per_sample = 8;
    info.num_color_channels = 3; info.uses_original_profile = JXL_FALSE;
    check(JxlEncoderSetBasicInfo(encoder, &info));
    JxlColorEncoding color; JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
    check(JxlEncoderSetColorEncoding(encoder, &color));
    JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(encoder, NULL);
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, 1));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 7));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EPF, 0));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_GABORISH, 0));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_RESAMPLING, 1));
    check(JxlEncoderSetFrameDistance(settings, 2));
    JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    check(JxlEncoderAddImageFrame(settings, &format, pixels, width * height * 3));
    JxlEncoderCloseInput(encoder);
    uint8_t output[65536]; uint8_t* next = output; size_t available = sizeof(output);
    check(JxlEncoderProcessOutput(encoder, &next, &available));
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/lf_modular_root_%ux%u.jxl.hex", argv[1], width, height)
        >= (int)sizeof(path)) abort();
    FILE* file = fopen(path, "wb"); if (!file) abort();
    for (size_t i = 0; i < sizeof(output) - available; ++i) {
      fprintf(file, "%02x", output[i]);
      if (i % 32 == 31) fputc('\n', file);
    }
    if ((sizeof(output) - available) % 32) fputc('\n', file);
    if (fclose(file)) abort();
    JxlEncoderDestroy(encoder);
  }
  return 0;
}
