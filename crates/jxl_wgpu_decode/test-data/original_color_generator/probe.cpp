#include <jxl/color_encoding.h>
#include <jxl/cms.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define Check(ok) do { if (!(ok)) { std::fprintf(stderr, "failed line %d\n", __LINE__); std::abort(); } } while (false)
std::vector<unsigned char> Encode(JxlColorEncoding color) {
  auto* enc = JxlEncoderCreate(nullptr);
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = 7; info.ysize = 1;
  info.bits_per_sample = 32; info.exponent_bits_per_sample = 8;
  info.uses_original_profile = JXL_TRUE;
  Check(JxlEncoderSetBasicInfo(enc, &info) == JXL_ENC_SUCCESS);
  Check(JxlEncoderSetColorEncoding(enc, &color) == JXL_ENC_SUCCESS);
  auto* settings = JxlEncoderFrameSettingsCreate(enc, nullptr);
  Check(JxlEncoderSetFrameLossless(settings, JXL_TRUE) == JXL_ENC_SUCCESS);
  Check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, 1) == JXL_ENC_SUCCESS);
  const std::array<float, 21> pixels = {-.25f,-.25f,-.25f, -1e-8f,-1e-8f,-1e-8f,
      0,0,0, .125f,.125f,.125f, 1,1,1, 1.25f,1.25f,1.25f, .5f,.2f,.7f};
  const JxlPixelFormat format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  Check(JxlEncoderAddImageFrame(settings, &format, pixels.data(), sizeof(pixels)) == JXL_ENC_SUCCESS);
  JxlEncoderCloseInput(enc);
  std::vector<unsigned char> data(65536);
  unsigned char* next = data.data(); size_t available = data.size();
  Check(JxlEncoderProcessOutput(enc, &next, &available) == JXL_ENC_SUCCESS);
  data.resize(data.size() - available);
  JxlEncoderDestroy(enc);
  return data;
}
void Decode(const std::vector<unsigned char>& bytes, JxlColorEncoding output) {
  auto* dec = JxlDecoderCreate(nullptr);
  Check(JxlDecoderSetCms(dec, *JxlGetDefaultCms()) == JXL_DEC_SUCCESS);
  Check(JxlDecoderSubscribeEvents(dec, JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS);
  Check(JxlDecoderSetInput(dec, bytes.data(), bytes.size()) == JXL_DEC_SUCCESS);
  JxlDecoderCloseInput(dec);
  std::array<float, 21> pixels;
  const JxlPixelFormat format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  for (;;) {
    const auto status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_COLOR_ENCODING) {
      Check(JxlDecoderSetOutputColorProfile(dec, &output, nullptr, 0) == JXL_DEC_SUCCESS);
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      Check(JxlDecoderSetImageOutBuffer(dec, &format, pixels.data(), sizeof(pixels)) == JXL_DEC_SUCCESS);
    } else if (status == JXL_DEC_FULL_IMAGE) {
      for (float x : pixels) std::printf(" %.9g", x);
      std::puts("");
    } else if (status == JXL_DEC_SUCCESS) break;
    else Check(false);
  }
  JxlDecoderDestroy(dec);
}
int main() {
  Check(JxlEncoderVersion() == 12000 && JxlDecoderVersion() == 12000);
  for (const auto tf : {JXL_TRANSFER_FUNCTION_GAMMA, JXL_TRANSFER_FUNCTION_DCI, JXL_TRANSFER_FUNCTION_LINEAR}) {
    JxlColorEncoding source;
    JxlColorEncodingSetToSRGB(&source, JXL_FALSE);
    source.transfer_function = tf;
    source.gamma = .4545455;
    source.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
    auto bytes = Encode(source);
    JxlColorEncoding linear = source;
    linear.transfer_function = JXL_TRANSFER_FUNCTION_LINEAR;
    std::printf("source transfer %u to linear", tf); Decode(bytes, linear);
    bytes = Encode(linear);
    std::printf("linear to transfer %u", tf); Decode(bytes, source);
  }
  JxlColorEncoding source;
  JxlColorEncodingSetToLinearSRGB(&source, JXL_FALSE);
  source.white_point = JXL_WHITE_POINT_E;
  source.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
  const auto bytes = Encode(source);
  for (const auto intent : {JXL_RENDERING_INTENT_PERCEPTUAL, JXL_RENDERING_INTENT_RELATIVE,
                            JXL_RENDERING_INTENT_SATURATION, JXL_RENDERING_INTENT_ABSOLUTE}) {
    JxlColorEncoding output;
    JxlColorEncodingSetToLinearSRGB(&output, JXL_FALSE);
    output.rendering_intent = intent;
    std::printf("white E to D65 intent %u", intent); Decode(bytes, output);
  }
}
