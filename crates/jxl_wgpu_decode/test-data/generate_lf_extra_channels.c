// Offline libjxl seeds. The Rust generator reframes the unchanged entropy into LF chains.
#include <jxl/encode.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check_at(JxlEncoderStatus status, int line) {
  if (status != JXL_ENC_SUCCESS) {
    fprintf(stderr, "libjxl encoder status %d at line %d\n", status, line);
    abort();
  }
}
#define check(status) check_at(status, __LINE__)

typedef struct {
  const char* name;
  uint32_t width, height, levels, colors, bits, exponent;
  uint32_t alpha_bits, alpha_exponent, depth_bits, depth_exponent;
  int associated, orientation, resampling, extra_resampling;
  uint32_t alpha_shift, depth_shift;
  int signed_depth;
} Case;

static void generate_case(const char* directory, const char* name, uint32_t width,
                          uint32_t height, int modular, uint32_t extras, int responsive,
                          const Case* config, uint32_t level, int root) {
  JxlEncoder* encoder = JxlEncoderCreate(NULL);
  JxlBasicInfo info; JxlEncoderInitBasicInfo(&info);
  info.xsize = width; info.ysize = height; info.bits_per_sample = 8;
  info.num_color_channels = 3; info.num_extra_channels = extras;
  info.uses_original_profile = JXL_FALSE;
  if (config) {
    check(JxlEncoderSetCodestreamLevel(encoder, 10));
    info.bits_per_sample = config->bits;
    info.exponent_bits_per_sample = config->exponent;
    info.num_color_channels = config->colors;
    info.orientation = (JxlOrientation)config->orientation;
  }
  check(JxlEncoderSetBasicInfo(encoder, &info));
  for (uint32_t c = 0; c < extras; ++c) {
    JxlExtraChannelInfo ec;
    JxlEncoderInitExtraChannelInfo(c ? JXL_CHANNEL_DEPTH : JXL_CHANNEL_ALPHA, &ec);
    ec.bits_per_sample = 8;
    if (config) {
      ec.bits_per_sample = c ? config->depth_bits : config->alpha_bits;
      ec.exponent_bits_per_sample = c ? config->depth_exponent : config->alpha_exponent;
      ec.alpha_premultiplied = !c && config->associated;
      ec.dim_shift = c ? config->depth_shift : config->alpha_shift;
    }
    check(JxlEncoderSetExtraChannelInfo(encoder, c, &ec));
  }
  JxlColorEncoding color; JxlColorEncodingSetToSRGB(&color, info.num_color_channels == 1);
  check(JxlEncoderSetColorEncoding(encoder, &color));
  JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(encoder, NULL);
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, modular));
  if (responsive >= 0)
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_RESPONSIVE, responsive));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 7));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EPF, 0));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_GABORISH, 0));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_PATCHES, 0));
  check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_RESAMPLING,
      config && root ? config->resampling : 1));
  if (config) {
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING,
        root ? config->extra_resampling : 1 << config->alpha_shift));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1));
    check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 0));
  }
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
  float* samples = config ? malloc(size * info.num_color_channels * sizeof(float)) : NULL;
  if (config && !samples) abort();
  if (config) {
    for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x)
      for (uint32_t c = 0; c < info.num_color_channels; ++c)
        samples[((size_t)y * width + x) * info.num_color_channels + c] =
            (float)((19 + 11*x + 23*y + 31*c + 17*level) % 129) / 128.0f;
  }
  JxlPixelFormat format = {info.num_color_channels,
      config ? JXL_TYPE_FLOAT : JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  check(JxlEncoderAddImageFrame(settings, &format, config ? (void*)samples : (void*)pixels,
      config ? size * info.num_color_channels * sizeof(float) : size * 3));
  for (uint32_t c = 0; c < extras; ++c) {
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        pixels[y * width + x] = (uint8_t)((c ? 31 : 11) * x + (c ? 3 : 17) * y);
        // Custom floating extras must be exactly representable. Dyadic values survive
        // the native encoder's lossless float-to-word conversion at both declared depths.
        if (config) samples[y * width + x] =
            (float)((13 + (c ? 31 : 11)*x + (c ? 3 : 17)*y + 7*level) % 129) / 128.0f;
        if (config && config->signed_depth && c) samples[y * width + x] =
            (float)((int)((13 + 31*x + 3*y + 7*level) % 257) - 64) / 128.0f;
      }
    }
    JxlPixelFormat plane = {1, config ? JXL_TYPE_FLOAT : JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    check(JxlEncoderSetExtraChannelBuffer(settings, &plane,
        config ? (void*)samples : (void*)pixels, config ? size * sizeof(float) : size, c));
  }
  free(samples);
  free(pixels);
  JxlEncoderCloseInput(encoder);
  char path[4096];
  if (snprintf(path, sizeof(path), "%s/%s.jxl", directory, name) >= (int)sizeof(path)) abort();
  FILE* file = fopen(path, "wb"); if (!file) abort();
  JxlEncoderStatus status;
  do {
    uint8_t output[16384]; uint8_t* next = output; size_t available = sizeof(output);
    status = JxlEncoderProcessOutput(encoder, &next, &available);
    if (status != JXL_ENC_SUCCESS && status != JXL_ENC_NEED_MORE_OUTPUT) {
      fprintf(stderr, "%s: libjxl output status %d, error %d\n",
          name, status, JxlEncoderGetError(encoder));
      abort();
    }
    const size_t size = sizeof(output) - available;
    if (fwrite(output, 1, size, file) != size) abort();
  } while (status == JXL_ENC_NEED_MORE_OUTPUT);
  if (fclose(file)) abort();
  JxlEncoderDestroy(encoder);
}

static void generate(const char* directory, const char* name, uint32_t width,
                     uint32_t height, int modular, uint32_t extras, int responsive) {
  generate_case(directory, name, width, height, modular, extras, responsive, NULL, 0, 0);
}

static void conformance(const char* directory, int geometry) {
  const Case cases[] = {
    {"integer_associated", 65, 33, 4, 3, 12, 0, 16, 0, 20, 0, 1, 6, 1, 1, 0, 0, 0},
    {"floating", 65, 33, 3, 3, 24, 7, 16, 5, 24, 7, 0, 8, 1, 1, 0, 0, 0},
    {"resampled_associated", 193, 129, 1, 3, 16, 0, 16, 0, 20, 0, 1, 2, 2, 8, 0, 0, 0},
    {"floating_resampled", 193, 129, 1, 3, 32, 8, 32, 8, 32, 8, 1, 7, 2, 8, 0, 0, 0},
    {"gray_float", 65, 33, 2, 1, 32, 8, 16, 5, 24, 7, 1, 5, 1, 1, 0, 0, 0},
  };
  const Case geometry_cases[] = {
    {"shifted_integer", 65, 33, 2, 3, 12, 0, 16, 0, 20, 0, 1, 6, 1, 4, 1, 1, 0},
    {"shifted_float", 65, 33, 2, 3, 24, 7, 32, 8, 32, 8, 1, 8, 1, 8, 2, 2, 1},
    {"shifted_thin_float", 193, 65, 1, 3, 24, 7, 32, 8, 32, 8, 0, 3, 1, 8, 3, 3, 1},
    {"signed_float", 65, 33, 3, 3, 24, 7, 16, 5, 24, 7, 0, 5, 1, 1, 0, 0, 1},
  };
  char path[4096];
  if (snprintf(path, sizeof(path), "%s/conformance.txt", directory) >= (int)sizeof(path)) abort();
  FILE* manifest = fopen(path, "w"); if (!manifest) abort();
  const size_t count = geometry ? sizeof(geometry_cases) / sizeof(*geometry_cases) : sizeof(cases) / sizeof(*cases);
  for (size_t i = 0; i < count; ++i) {
    const Case* config = geometry ? &geometry_cases[i] : &cases[i];
    fprintf(manifest, "%s %u\n", config->name, config->levels);
    for (uint32_t level = 0; level <= config->levels; ++level) {
      uint32_t factor = 1u << (3 * level);
      for (int modular = 0; modular <= (level == config->levels); ++modular) {
        char name[128];
        snprintf(name, sizeof(name), "%s_%s_lf%u", config->name,
            modular ? "modular" : "vardct", level);
        generate_case(directory, name, (config->width + factor - 1) / factor,
            (config->height + factor - 1) / factor, modular, 2, 0, config, level,
            level == config->levels);
      }
    }
    if (geometry) {
      char name[4096];
      if (snprintf(name, sizeof(name), "%s_crop", config->name) >= (int)sizeof(name)) abort();
      generate_case(directory, name, config->width - 12, config->height - 8,
                    0, 2, 0, config, 0, 0);
    }
  }
  if (fclose(manifest)) abort();
}

int main(int argc, char** argv) {
  if (argc == 3 && !strcmp(argv[2], "--conformance")) {
    conformance(argv[1], 0);
    return 0;
  }
  if (argc == 3 && !strcmp(argv[2], "--geometry")) {
    conformance(argv[1], 1);
    return 0;
  }
  if (argc == 3 && !strcmp(argv[2], "--distributed")) {
    generate(argv[1], "modular_root", 257, 5, 1, 2, 0);
    generate(argv[1], "vardct_root", 257, 5, 0, 2, 0);
    generate(argv[1], "extras", 2051, 33, 0, 2, 1);
    return 0;
  }
  if (argc != 2) return 2;
  generate(argv[1], "modular_root", 9, 5, 1, 2, -1);
  generate(argv[1], "vardct_root", 9, 5, 0, 2, -1);
  generate(argv[1], "modular_root2", 2, 1, 1, 2, -1);
  generate(argv[1], "vardct_root2", 2, 1, 0, 2, -1);
  generate(argv[1], "extras", 65, 33, 0, 2, -1);
  return 0;
}
