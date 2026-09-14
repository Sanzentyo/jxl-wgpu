#include "references.hpp"
#include <jxl/cms.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void Require(bool condition, const char* operation) {
  if (!condition) { std::fprintf(stderr, "%s failed\n", operation); std::exit(1); }
}
namespace {
void Dec(JxlDecoderStatus status) { Require(status == JXL_DEC_SUCCESS, "decode"); }
}
std::vector<float> Decode(const std::vector<uint8_t>& encoded,
                         JxlColorEncoding original, uint32_t nits, bool linear) {
  auto* decoder = JxlDecoderCreate(nullptr);
  Require(decoder != nullptr, "decoder allocation");
  Dec(JxlDecoderSetCms(decoder, *JxlGetDefaultCms()));
  Dec(JxlDecoderSubscribeEvents(decoder, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE));
  Dec(JxlDecoderSetUnpremultiplyAlpha(decoder, JXL_FALSE));
  Dec(JxlDecoderSetRenderSpotcolors(decoder, JXL_FALSE));
  Dec(JxlDecoderSetKeepOrientation(decoder, JXL_TRUE));
  Dec(JxlDecoderSetInput(decoder, encoded.data(), encoded.size()));
  JxlDecoderCloseInput(decoder);
  std::vector<float> frame, result;
  const JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  for (;;) {
    const auto status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info;
      Dec(JxlDecoderGetBasicInfo(decoder, &info));
      Require(info.intensity_target == float(nits), "image intensity target");
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      JxlColorEncoding declared;
      Dec(JxlDecoderGetColorAsEncodedProfile(decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &declared));
      Require(declared.transfer_function == original.transfer_function &&
          declared.color_space == original.color_space &&
          (original.color_space == JXL_COLOR_SPACE_GRAY || declared.primaries == original.primaries), "original color metadata");
      if (linear) original.transfer_function = JXL_TRANSFER_FUNCTION_LINEAR;
      Dec(JxlDecoderSetOutputColorProfile(decoder, &original, nullptr, 0));
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t bytes;
      Dec(JxlDecoderImageOutBufferSize(decoder, &format, &bytes));
      Require(bytes % sizeof(float) == 0, "F32 buffer");
      frame.resize(bytes / sizeof(float));
      Dec(JxlDecoderSetImageOutBuffer(decoder, &format, frame.data(), bytes));
    } else if (status == JXL_DEC_FULL_IMAGE) {
      for (auto value : frame) Require(std::isfinite(value), "finite native reference");
      result.insert(result.end(), frame.begin(), frame.end());
    } else if (status == JXL_DEC_SUCCESS) break;
    else Require(false, "native HDR decode");
  }
  JxlDecoderDestroy(decoder);
  return result;
}
void WriteBytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  FILE* stream = std::fopen(path.c_str(), "w");
  Require(stream != nullptr, "codestream file");
  for (size_t i = 0; i < bytes.size(); ++i)
    std::fprintf(stream, "%02x%s", bytes[i], (i + 1) % 32 == 0 ? "\n" : "");
  if (bytes.size() % 32) std::fputc('\n', stream);
  Require(std::fclose(stream) == 0, "codestream close");
}
void WriteFloats(const std::filesystem::path& path, const std::vector<float>& values) {
  FILE* stream = std::fopen(path.c_str(), "w");
  Require(stream != nullptr, "reference file");
  for (size_t i = 0; i < values.size(); ++i) {
    uint32_t word;
    std::memcpy(&word, &values[i], sizeof(word));
    std::fprintf(stream, "%08x%s", word, (i + 1) % 8 == 0 || i + 1 == values.size() ? "\n" : " ");
  }
  Require(std::fclose(stream) == 0, "reference close");
}
