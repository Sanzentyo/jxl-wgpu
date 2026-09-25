/* Offline differential oracle, linked only by optional native interoperability tests.
 * Prints each coalesced frame as little-endian F32 RGBA then full-resolution extra planes.
 * --preview prints only the independent preview RGBA image, without extra planes.
 * cc decode_extra_channels.c $(pkg-config --cflags --libs libjxl libjxl_cms) -o /tmp/jxl-extra-oracle
 * --prefix flushes one incomplete frame at the supplied physical section boundary.
 * --xyb requests libjxl's scaled XYB output for offline component-domain references.
 * --original requests and verifies the enumerated original encoding using libjxl 0.12.0.
 * --original-icc verifies exact original/data ICC identity before original-component output.
 * --no-cms requires original ICC passthrough or XYB-to-linear output, neither of which
 * needs an ICC connection; verify the selected output instead of parsing an unused method.
 * /tmp/jxl-extra-oracle INPUT.jxl [--preview] [--prefix] [--preserve-alpha] [--linear|--xyb|--original|--original-icc] [--keep-orientation] [--render-spots] > CHANNELS.f32
 */
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/color_encoding.h>
#include <jxl/cms.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require_icc(int ok, const char* message) {
  if (!ok) { fprintf(stderr, "original ICC oracle: %s\n", message); exit(3); }
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  int no_cms = 0, unpremultiply = 1, linear = 0, xyb = 0, original = 0, original_icc = 0, keep_orientation = 0, render_spots = 0, preview = 0, prefix = 0;
  for (int i=2; i<argc; ++i) {
    if (!strcmp(argv[i], "--no-cms")) no_cms = 1;
    else if (!strcmp(argv[i], "--preserve-alpha")) unpremultiply = 0;
    else if (!strcmp(argv[i], "--linear")) linear = 1;
    else if (!strcmp(argv[i], "--xyb")) xyb = 1;
    else if (!strcmp(argv[i], "--original")) original = 1;
    else if (!strcmp(argv[i], "--original-icc")) original_icc = 1;
    else if (!strcmp(argv[i], "--keep-orientation")) keep_orientation = 1;
    else if (!strcmp(argv[i], "--render-spots")) render_spots = 1;
    else if (!strcmp(argv[i], "--preview")) preview = 1;
    else if (!strcmp(argv[i], "--prefix")) prefix = 1;
    else return 2;
  }
  if (xyb + linear + original + original_icc > 1 || (no_cms && !linear && !original_icc)) return 2;
  if ((original || original_icc) && JxlDecoderVersion() != 12000) return 2;
  FILE* in = fopen(argv[1], "rb"); if (!in) return 2;
  if (fseek(in, 0, SEEK_END)) return 2;
  long length = ftell(in); if (length <= 0) return 2;
  rewind(in);
  uint8_t* data = malloc((size_t)length); if (!data) return 2;
  if (fread(data, 1, (size_t)length, in) != (size_t)length) return 2;
  fclose(in);
  JxlDecoder* dec = JxlDecoderCreate(NULL);
  if (!no_cms && (linear || original || original_icc) && dec) JxlDecoderSetCms(dec, *JxlGetDefaultCms());
  if (!dec || JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | (preview ? JXL_DEC_PREVIEW_IMAGE : JXL_DEC_FULL_IMAGE)) != JXL_DEC_SUCCESS
      || JxlDecoderSetRenderSpotcolors(dec, render_spots) != JXL_DEC_SUCCESS
      || JxlDecoderSetUnpremultiplyAlpha(dec, unpremultiply) != JXL_DEC_SUCCESS
      || JxlDecoderSetKeepOrientation(dec, keep_orientation) != JXL_DEC_SUCCESS
      || JxlDecoderSetInput(dec, data, (size_t)length) != JXL_DEC_SUCCESS) return 3;
  if (!prefix) JxlDecoderCloseInput(dec);
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
      if (no_cms) require_icc((linear && !info.uses_original_profile) || (original_icc && info.uses_original_profile), "requested codec domain requires a CMS connection");
      if (original_icc) {
        size_t original_size = 0, actual_size = 0;
        if (JxlDecoderGetICCProfileSize(dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &original_size) != JXL_DEC_SUCCESS
            || !original_size || original_size > (16 << 20)) return 3;
        uint8_t* declared = malloc(original_size);
        uint8_t* actual = malloc(original_size);
        if (!declared || !actual) return 2;
        require_icc(JxlDecoderGetColorAsICCProfile(dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL, declared, original_size) == JXL_DEC_SUCCESS, "read declared profile");
        /* Original-profile streams already expose the declared device samples. libjxl 0.12
         * rejects an explicit ICC request here; require exact DATA profile identity below.
         * XYB needs an explicit output-profile request. */
        if (!info.uses_original_profile)
          require_icc(JxlDecoderSetOutputColorProfile(dec, NULL, declared, original_size) == JXL_DEC_SUCCESS, "request declared profile");
        require_icc(JxlDecoderGetICCProfileSize(dec, JXL_COLOR_PROFILE_TARGET_DATA, &actual_size) == JXL_DEC_SUCCESS, "read output profile size");
        require_icc(actual_size == original_size, "output profile size differs");
        require_icc(JxlDecoderGetColorAsICCProfile(dec, JXL_COLOR_PROFILE_TARGET_DATA, actual, actual_size) == JXL_DEC_SUCCESS, "read output profile");
        require_icc(!memcmp(declared, actual, original_size), "output profile bytes differ");
        free(declared);
        free(actual);
      }
      if (original) {
        JxlColorEncoding declared, actual;
        if (JxlDecoderGetColorAsEncodedProfile(dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &declared) != JXL_DEC_SUCCESS) return 3;
        /* v0.12.0 rejects explicit non-D65 Gray on original streams; their default is
         * already original. XYB Gray needs an explicit request to avoid linear fallback. */
        if (!(info.uses_original_profile && declared.color_space == JXL_COLOR_SPACE_GRAY && declared.white_point != JXL_WHITE_POINT_D65)
            && JxlDecoderSetOutputColorProfile(dec, &declared, NULL, 0) != JXL_DEC_SUCCESS) return 3;
        if (JxlDecoderGetColorAsEncodedProfile(dec, JXL_COLOR_PROFILE_TARGET_DATA, &actual) != JXL_DEC_SUCCESS) return 3;
        if (actual.color_space != declared.color_space || actual.white_point != declared.white_point
            || actual.transfer_function != declared.transfer_function || actual.rendering_intent != declared.rendering_intent
            || (declared.transfer_function == JXL_TRANSFER_FUNCTION_GAMMA && actual.gamma != declared.gamma)) return 3;
        for (int c=0; c<2; ++c) {
          if (actual.white_point_xy[c] != declared.white_point_xy[c]) return 3;
          if (declared.color_space == JXL_COLOR_SPACE_RGB &&
              (actual.primaries != declared.primaries || actual.primaries_red_xy[c] != declared.primaries_red_xy[c]
               || actual.primaries_green_xy[c] != declared.primaries_green_xy[c] || actual.primaries_blue_xy[c] != declared.primaries_blue_xy[c])) return 3;
        }
      }
      if (linear || xyb) {
        JxlColorEncoding color; JxlColorEncodingSetToLinearSRGB(&color, info.num_color_channels == 1);
        if (xyb) {
          color.color_space = JXL_COLOR_SPACE_XYB;
          color.transfer_function = JXL_TRANSFER_FUNCTION_GAMMA;
          color.gamma = 1.0 / 3.0;
          color.rendering_intent = JXL_RENDERING_INTENT_PERCEPTUAL;
        }
        if (JxlDecoderSetOutputColorProfile(dec, &color, NULL, 0) != JXL_DEC_SUCCESS) return 3;
        if (no_cms) {
          JxlColorEncoding actual;
          require_icc(JxlDecoderGetColorAsEncodedProfile(dec, JXL_COLOR_PROFILE_TARGET_DATA, &actual) == JXL_DEC_SUCCESS, "read linear output encoding");
          require_icc(actual.color_space == color.color_space && actual.transfer_function == color.transfer_function
                      && actual.white_point == color.white_point
                      && (info.num_color_channels == 1 || actual.primaries == color.primaries), "linear output encoding differs");
        }
      }
    } else if (status == JXL_DEC_NEED_PREVIEW_OUT_BUFFER) {
      if (!preview || JxlDecoderPreviewOutBufferSize(dec, &color_format, &color_size) != JXL_DEC_SUCCESS) return 3;
      free(rgba); rgba = malloc(color_size); if (!rgba) return 2;
      if (JxlDecoderSetPreviewOutBuffer(dec, &color_format, rgba, color_size) != JXL_DEC_SUCCESS) return 3;
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
    } else if (status == JXL_DEC_PREVIEW_IMAGE) {
      if (!preview || fwrite(rgba, 1, color_size, stdout) != color_size) return 3;
      ++complete;
    } else if (status == JXL_DEC_FULL_IMAGE || (prefix && status == JXL_DEC_NEED_MORE_INPUT)) {
      if (status == JXL_DEC_NEED_MORE_INPUT && JxlDecoderFlushImage(dec) != JXL_DEC_SUCCESS) return 3;
      ++complete;
      if (fwrite(rgba, 1, color_size, stdout) != color_size) return 2;
      for (uint32_t c=0; c<info.num_extra_channels; ++c) {
        if (fwrite(extras[c], 1, plane_size, stdout) != plane_size) return 2;
      }
      if (prefix) break;
    } else if (status == JXL_DEC_SUCCESS) {
      if (!complete) return 3;
      break;
    } else return 3;
  }
  for (uint32_t c=0; c<info.num_extra_channels; ++c) free(extras[c]);
  free(extras); free(rgba); free(data); JxlDecoderDestroy(dec);
  return 0;
}
