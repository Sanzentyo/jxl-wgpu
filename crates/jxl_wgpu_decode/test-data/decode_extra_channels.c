/* Offline differential oracle, linked only by optional native interoperability tests.
 * Prints each coalesced frame as little-endian F32 RGBA then full-resolution extra planes.
 * cc decode_extra_channels.c $(pkg-config --cflags --libs libjxl libjxl_cms) -o /tmp/jxl-extra-oracle
 * /tmp/jxl-extra-oracle INPUT.jxl [--preserve-alpha] [--linear] [--keep-orientation] [--render-spots] > CHANNELS.f32
 */
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/color_encoding.h>
#include <jxl/cms.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  int unpremultiply = 1, linear = 0, keep_orientation = 0, render_spots = 0;
  for (int i=2; i<argc; ++i) {
    if (!strcmp(argv[i], "--preserve-alpha")) unpremultiply = 0;
    else if (!strcmp(argv[i], "--linear")) linear = 1;
    else if (!strcmp(argv[i], "--keep-orientation")) keep_orientation = 1;
    else if (!strcmp(argv[i], "--render-spots")) render_spots = 1;
    else return 2;
  }
  FILE* in = fopen(argv[1], "rb"); if (!in) return 2;
  if (fseek(in, 0, SEEK_END)) return 2;
  long length = ftell(in); if (length <= 0) return 2;
  rewind(in);
  uint8_t* data = malloc((size_t)length); if (!data) return 2;
  if (fread(data, 1, (size_t)length, in) != (size_t)length) return 2;
  fclose(in);
  JxlDecoder* dec = JxlDecoderCreate(NULL);
  if (linear && dec) JxlDecoderSetCms(dec, *JxlGetDefaultCms());
  if (!dec || JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE) != JXL_DEC_SUCCESS
      || JxlDecoderSetRenderSpotcolors(dec, render_spots) != JXL_DEC_SUCCESS
      || JxlDecoderSetUnpremultiplyAlpha(dec, unpremultiply) != JXL_DEC_SUCCESS
      || JxlDecoderSetKeepOrientation(dec, keep_orientation) != JXL_DEC_SUCCESS
      || JxlDecoderSetInput(dec, data, (size_t)length) != JXL_DEC_SUCCESS) return 3;
  JxlDecoderCloseInput(dec);
  JxlBasicInfo info; uint8_t* rgba = NULL; uint8_t** extras = NULL;
  size_t color_size = 0, plane_size = 0;
  const JxlPixelFormat color_format = {4, JXL_TYPE_FLOAT, JXL_LITTLE_ENDIAN, 0};
  const JxlPixelFormat plane_format = {1, JXL_TYPE_FLOAT, JXL_LITTLE_ENDIAN, 0};
  int complete = 0;
  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_BASIC_INFO) {
      if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS) return 3;
      extras = calloc(info.num_extra_channels ? info.num_extra_channels : 1, sizeof(*extras));
      if (!extras) return 2;
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      if (linear) {
        JxlColorEncoding color; JxlColorEncodingSetToLinearSRGB(&color, info.num_color_channels == 1);
        if (JxlDecoderSetOutputColorProfile(dec, &color, NULL, 0) != JXL_DEC_SUCCESS) return 3;
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      if (JxlDecoderImageOutBufferSize(dec, &color_format, &color_size) != JXL_DEC_SUCCESS) return 3;
      free(rgba); rgba = malloc(color_size); if (!rgba) return 2;
      if (JxlDecoderSetImageOutBuffer(dec, &color_format, rgba, color_size) != JXL_DEC_SUCCESS) return 3;
      for (uint32_t c=0; c<info.num_extra_channels; ++c) {
        size_t size;
        if (JxlDecoderExtraChannelBufferSize(dec, &plane_format, &size, c) != JXL_DEC_SUCCESS) return 3;
        if (c && size != plane_size) return 3;
        plane_size = size; free(extras[c]); extras[c] = malloc(size); if (!extras[c]) return 2;
        if (JxlDecoderSetExtraChannelBuffer(dec, &plane_format, extras[c], size, c) != JXL_DEC_SUCCESS) return 3;
      }
    } else if (status == JXL_DEC_FULL_IMAGE) {
      ++complete;
      if (fwrite(rgba, 1, color_size, stdout) != color_size) return 2;
      for (uint32_t c=0; c<info.num_extra_channels; ++c) {
        if (fwrite(extras[c], 1, plane_size, stdout) != plane_size) return 2;
      }
    } else if (status == JXL_DEC_SUCCESS) {
      if (!complete) return 3;
      break;
    } else return 3;
  }
  for (uint32_t c=0; c<info.num_extra_channels; ++c) free(extras[c]);
  free(extras); free(rgba); free(data); JxlDecoderDestroy(dec);
  return 0;
}
