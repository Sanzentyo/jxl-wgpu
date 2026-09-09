/* Offline libjxl oracle: INPUT [CHUNK_BYTES [SNAPSHOT_PREFIX [linear] [keep]]].
 * Production decoding does not link to this helper. Use whole input for noisy
 * images: libjxl 0.12.0 retries incomplete frame headers without rolling back
 * its persistent noise-frame counters. GPU fragmented input is tested separately.
 */
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/color_encoding.h>
#include <jxl/cms.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc < 2 || argc > 6) return 2;
  int linear = 0, keep = 0;
  for (int arg = 4; arg < argc; ++arg) {
    if (strcmp(argv[arg], "linear") == 0) linear = 1;
    else if (strcmp(argv[arg], "keep") == 0) keep = 1;
    else return 2;
  }
  FILE *input = fopen(argv[1], "rb");
  if (!input || fseek(input, 0, SEEK_END)) return 2;
  long length = ftell(input);
  if (length <= 0) return 2;
  rewind(input);
  unsigned char *bytes = malloc((size_t)length);
  if (!bytes || fread(bytes, 1, (size_t)length, input) != (size_t)length) return 2;
  fclose(input);
  size_t quantum = argc > 2 ? (size_t)strtoull(argv[2], NULL, 10) : (size_t)length;
  if (!quantum) return 2;
  size_t delivered = quantum < (size_t)length ? quantum : (size_t)length;
  JxlDecoder *decoder = JxlDecoderCreate(NULL);
  if (linear && decoder) JxlDecoderSetCms(decoder, *JxlGetDefaultCms());
  if (!decoder || JxlDecoderSubscribeEvents(decoder, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE | JXL_DEC_FRAME_PROGRESSION) != JXL_DEC_SUCCESS || JxlDecoderSetProgressiveDetail(decoder, kPasses) != JXL_DEC_SUCCESS || JxlDecoderSetInput(decoder, bytes, delivered) != JXL_DEC_SUCCESS) return 3;
  if (JxlDecoderSetKeepOrientation(decoder, keep) != JXL_DEC_SUCCESS) return 3;
  JxlBasicInfo info;
  const JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_LITTLE_ENDIAN, 0};
  float *pixels = NULL;
  size_t output_bytes = 0, frame = 0, step = 0;
  int closed = 0;
  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_BASIC_INFO) {
      if (JxlDecoderGetBasicInfo(decoder, &info) != JXL_DEC_SUCCESS) return 3;
      printf("image,%u,%u,%u,%u\n", info.xsize, info.ysize, info.num_extra_channels, info.have_animation);
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      if (linear) {
        JxlColorEncoding color;
        JxlColorEncodingSetToLinearSRGB(&color, info.num_color_channels == 1);
        if (JxlDecoderSetPreferredColorProfile(decoder, &color) != JXL_DEC_SUCCESS) return 3;
      }
    } else if (status == JXL_DEC_FRAME) {
      step = 0;
      JxlFrameHeader header;
      if (JxlDecoderGetFrameHeader(decoder, &header) != JXL_DEC_SUCCESS) return 3;
      printf("frame,%zu,%u,%u,%u\n", frame, header.duration, header.timecode, header.is_last);
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      if (JxlDecoderImageOutBufferSize(decoder, &format, &output_bytes) != JXL_DEC_SUCCESS) return 3;
      free(pixels);
      pixels = calloc(1, output_bytes);
      if (!pixels || JxlDecoderSetImageOutBuffer(decoder, &format, pixels, output_bytes) != JXL_DEC_SUCCESS) return 3;
    } else if (status == JXL_DEC_FRAME_PROGRESSION || status == JXL_DEC_FULL_IMAGE) {
      int final = status == JXL_DEC_FULL_IMAGE;
      if (!final && JxlDecoderFlushImage(decoder) != JXL_DEC_SUCCESS) { printf("unavailable,%zu,%zu,%zu\n", frame, step, JxlDecoderGetIntendedDownsamplingRatio(decoder)); continue; }
      uint64_t hash = UINT64_C(14695981039346656037);
      size_t nonfinite = 0;
      for (size_t i = 0; i < output_bytes / sizeof(float); ++i) if (!isfinite(pixels[i])) ++nonfinite;
      for (size_t i = 0; i < output_bytes; ++i) { hash ^= ((unsigned char*)pixels)[i]; hash *= UINT64_C(1099511628211); }
      size_t remaining = JxlDecoderReleaseInput(decoder);
      size_t consumed = delivered - remaining;
      printf("%s,%zu,%zu,%zu,%zu,%zu,%016llx\n", final ? "final" : "progress", frame, step++, final ? 1 : JxlDecoderGetIntendedDownsamplingRatio(decoder), consumed, nonfinite, (unsigned long long)hash);
      if (JxlDecoderSetInput(decoder, bytes + consumed, remaining) != JXL_DEC_SUCCESS) return 3;
      if (argc > 3) {
        char output_path[4096];
        int count = snprintf(output_path, sizeof(output_path), "%s-frame%zu-step%zu-%s.f32", argv[3], frame, step - 1, final ? "final" : "progress");
        if (count < 0 || (size_t)count >= sizeof(output_path)) return 2;
        FILE *output = fopen(output_path, "wb");
        if (!output || fwrite(pixels, 1, output_bytes, output) != output_bytes || fclose(output)) return 2;
      }
      if (final) ++frame;
    } else if (status == JXL_DEC_NEED_MORE_INPUT && !closed) {
      size_t remaining = JxlDecoderReleaseInput(decoder);
      size_t consumed = delivered - remaining;
      if (delivered < (size_t)length) {
        size_t added = (size_t)length - delivered;
        if (added > quantum) added = quantum;
        delivered += added;
      } else { JxlDecoderCloseInput(decoder); closed = 1; }
      if (JxlDecoderSetInput(decoder, bytes + consumed, delivered - consumed) != JXL_DEC_SUCCESS) return 3;
    } else if (status == JXL_DEC_SUCCESS) {
      break;
    } else { fprintf(stderr, "decoder status %d\n", status); return 3; }
  }
  free(pixels); free(bytes); JxlDecoderDestroy(decoder);
  return 0;
}
