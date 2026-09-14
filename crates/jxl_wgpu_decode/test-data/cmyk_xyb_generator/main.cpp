// Offline legal CMYK-suggested XYB sequences and independent ICC device
// references.
#include "../../../../tools/jxl_test_support/native/icc/linear.hpp"
#include "../../../jxl_wgpu/test-data/icc_generator/lut/black.hpp"
#include "../cmyk_generator/original.hpp"
#include <iterator>
#include <jxl/cms.h>
#include <jxl/decode.h>
#include <jxl/encode.h>

namespace {
using namespace lut;
using namespace cmyk;
void Enc(JxlEncoderStatus status) {
  Check(status == JXL_ENC_SUCCESS, "CMYK encoder operation");
}
float Sample(unsigned frame, unsigned pixel, unsigned channel) {
  return float((pixel * (channel * 2 + 3) + frame * 7 + channel * 11) % 17) /
         16;
}

Bytes Encode(const Bytes &profile, unsigned mode, unsigned black) {
  auto *encoder = JxlEncoderCreate(nullptr);
  Check(encoder, "create CMYK encoder");
  JxlEncoderSetCms(encoder, *JxlGetDefaultCms());
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = 17;
  info.ysize = 9;
  info.bits_per_sample = 32;
  info.exponent_bits_per_sample = 8;
  info.num_color_channels = 3;
  info.num_extra_channels = 3;
  info.alpha_bits = 32;
  info.alpha_exponent_bits = 8;
  info.uses_original_profile = JXL_FALSE;
  info.have_animation = JXL_TRUE;
  info.animation.tps_numerator = 10;
  info.animation.tps_denominator = 1;
  Enc(JxlEncoderSetBasicInfo(encoder, &info));
  Enc(JxlEncoderSetICCProfile(encoder, profile.data(), profile.size()));
  for (unsigned index = 0; index < 3; ++index) {
    JxlExtraChannelInfo extra;
    JxlEncoderInitExtraChannelInfo(index == black ? JXL_CHANNEL_BLACK
                                   : index == 1   ? JXL_CHANNEL_ALPHA
                                                  : JXL_CHANNEL_SPOT_COLOR,
                                   &extra);
    extra.bits_per_sample = 32;
    extra.exponent_bits_per_sample = 8;
    std::copy(kInk.begin(), kInk.end(), extra.spot_color);
    Enc(JxlEncoderSetExtraChannelInfo(encoder, index, &extra));
  }
  const JxlPixelFormat colors{3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  const JxlPixelFormat scalar{1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  for (unsigned frame = 0; frame < 3; ++frame) {
    auto *settings = JxlEncoderFrameSettingsCreate(encoder, nullptr);
    Enc(JxlEncoderSetFrameDistance(settings, 1));
    for (const auto [option, value] :
         std::array<std::pair<JxlEncoderFrameSettingId, int>, 9>{
             {{JXL_ENC_FRAME_SETTING_MODULAR, mode == 0},
              {JXL_ENC_FRAME_SETTING_EFFORT, 3},
              {JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM, 0},
              {JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1},
              {JXL_ENC_FRAME_SETTING_PATCHES, 0},
              {JXL_ENC_FRAME_SETTING_DOTS, 0},
              {JXL_ENC_FRAME_SETTING_NOISE, 0},
              {JXL_ENC_FRAME_SETTING_GABORISH, 0},
              {JXL_ENC_FRAME_SETTING_EPF, 0}}})
      Enc(JxlEncoderFrameSettingsSetOption(settings, option, value));
    JxlFrameHeader header;
    JxlEncoderInitFrameHeader(&header);
    header.duration = frame + 1;
    header.layer_info.save_as_reference = 0;
    auto &color_blend = header.layer_info.blend_info;
    color_blend.blendmode = JXL_BLEND_REPLACE;
    color_blend.source = 0;
    color_blend.alpha = 1;
    color_blend.clamp = JXL_FALSE;
    Enc(JxlEncoderSetFrameHeader(settings, &header));
    for (unsigned index = 0; index < 3; ++index) {
      JxlBlendInfo blend = color_blend;
      blend.blendmode = JXL_BLEND_REPLACE;
      Enc(JxlEncoderSetExtraChannelBlendInfo(settings, index, &blend));
      Enc(JxlEncoderSetExtraChannelDistance(settings, index, 0));
    }
    std::vector<float> input(kPixels * 3);
    for (unsigned pixel = 0; pixel < kPixels; ++pixel)
      for (unsigned c = 0; c < 3; ++c)
        input[pixel * 3 + c] = Sample(frame, pixel, c);
    Enc(JxlEncoderAddImageFrame(settings, &colors, input.data(),
                                input.size() * 4));
    for (unsigned index = 0; index < 3; ++index) {
      std::vector<float> values(kPixels);
      for (unsigned pixel = 0; pixel < kPixels; ++pixel)
        values[pixel] = Sample(frame, pixel, index + 3);
      Enc(JxlEncoderSetExtraChannelBuffer(settings, &scalar, values.data(),
                                          values.size() * 4, index));
    }
  }
  JxlEncoderCloseInput(encoder);
  Bytes bytes;
  for (;;) {
    std::array<uint8_t, 16384> buffer;
    auto *next = buffer.data();
    size_t available = buffer.size();
    const auto status = JxlEncoderProcessOutput(encoder, &next, &available);
    Check(status == JXL_ENC_SUCCESS || status == JXL_ENC_NEED_MORE_OUTPUT,
          "encode CMYK stream");
    bytes.insert(bytes.end(), buffer.data(), next);
    if (status == JXL_ENC_SUCCESS)
      break;
  }
  JxlEncoderDestroy(encoder);
  return bytes;
}

Bytes Read(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  Check(bool(input), "read frozen profile");
  return Bytes(std::istreambuf_iterator<char>{input}, {});
}

std::vector<float> Linear(const Bytes &bytes, const Bytes &profile,
                          unsigned black) {
  auto *decoder = JxlDecoderCreate(nullptr);
  Check(decoder, "create XYB decoder");
  Dec(JxlDecoderSubscribeEvents(decoder,
                                JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING |
                                    JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE));
  Dec(JxlDecoderSetCms(decoder, *JxlGetDefaultCms()));
  Dec(JxlDecoderSetRenderSpotcolors(decoder, JXL_FALSE));
  Dec(JxlDecoderSetUnpremultiplyAlpha(decoder, JXL_FALSE));
  Dec(JxlDecoderSetInput(decoder, bytes.data(), bytes.size()));
  JxlDecoderCloseInput(decoder);
  const JxlPixelFormat colors{3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  const JxlPixelFormat scalar{1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  std::vector<float> color(kPixels * 3), output;
  std::array<std::vector<float>, 3> extras;
  for (auto &extra : extras)
    extra.resize(kPixels);
  unsigned frames = 0;
  for (;;) {
    const auto status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info;
      Dec(JxlDecoderGetBasicInfo(decoder, &info));
      Check(info.xsize == 17 && info.ysize == 9 &&
                info.num_color_channels == 3 && info.num_extra_channels == 3 &&
                !info.uses_original_profile,
            "CMYK-suggested XYB geometry");
      JxlExtraChannelInfo extra;
      Dec(JxlDecoderGetExtraChannelInfo(decoder, black, &extra));
      Check(extra.type == JXL_CHANNEL_BLACK, "Black metadata index");
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      size_t size = 0;
      Dec(JxlDecoderGetICCProfileSize(
          decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &size));
      Bytes actual(size);
      Dec(JxlDecoderGetColorAsICCProfile(
          decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL, actual.data(), size));
      Check(actual == profile, "original CMYK profile retained");
      JxlColorEncoding linear;
      JxlColorEncodingSetToLinearSRGB(&linear, JXL_FALSE);
      Dec(JxlDecoderSetOutputColorProfile(decoder, &linear, nullptr, 0));
      Dec(JxlDecoderGetColorAsEncodedProfile(
          decoder, JXL_COLOR_PROFILE_TARGET_DATA, &linear));
      Check(linear.color_space == JXL_COLOR_SPACE_RGB &&
                linear.white_point == JXL_WHITE_POINT_D65 &&
                linear.primaries == JXL_PRIMARIES_SRGB &&
                linear.transfer_function == JXL_TRANSFER_FUNCTION_LINEAR,
            "actual DATA is linear D65 BT.709");
    } else if (status == JXL_DEC_FRAME) {
      JxlFrameHeader header;
      Dec(JxlDecoderGetFrameHeader(decoder, &header));
      Check(header.duration == frames + 1, "native presentation duration");
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      Dec(JxlDecoderSetImageOutBuffer(decoder, &colors, color.data(),
                                      color.size() * 4));
      for (unsigned index = 0; index < 3; ++index)
        Dec(JxlDecoderSetExtraChannelBuffer(
            decoder, &scalar, extras[index].data(), kPixels * 4, index));
    } else if (status == JXL_DEC_FULL_IMAGE) {
      for (unsigned p = 0; p < kPixels; ++p) {
        for (unsigned c = 0; c < 3; ++c)
          output.push_back(color[p * 3 + c]);
        for (unsigned index = 0; index < 3; ++index) {
          Check(extras[index][p] == Sample(frames, p, index + 3),
                "XYB must retain each independent extra sample");
          output.push_back(extras[index][p]);
        }
      }
      ++frames;
    } else if (status == JXL_DEC_SUCCESS)
      break;
    else
      Check(false, "native linear CMYK-suggested XYB decode");
  }
  JxlDecoderDestroy(decoder);
  Check(frames == 3 && output.size() == 3 * kPixels * 6, "linear frame count");
  return output;
}

Values Pcs(const connection::Matrix &matrix, const connection::Vector &input,
           bool uncertain) {
  Values output;
  for (unsigned r = 0; r < 3; ++r) {
    double value = 0, radius = 0, magnitude = 0, coefficients = 0;
    for (unsigned c = 0; c < 3; ++c) {
      value += matrix[r][c] * input[c];
      magnitude += std::abs(matrix[r][c] * input[c]);
      coefficients += std::abs(matrix[r][c]);
      if (uncertain)
        radius += std::abs(matrix[r][c]) * (1 + std::abs(input[c])) / 1024;
    }
    radius += 4e-7 * (1 + magnitude + coefficients);
    output.push_back({value, radius});
  }
  return output;
}

unsigned References(const Profile &target, const std::vector<float> &linear,
                    unsigned black, const std::filesystem::path &out,
                    const std::string &name) {
  auto xyz = cmsCreateXYZProfile();
  auto dst = cmsOpenProfileFromMem(target.bytes.data(), target.bytes.size());
  Check(xyz && dst, "native CMYK destination");
  const auto matrix = connection::kSpaces[0].ToPcs();
  unsigned components = 0;
  for (unsigned intent = 0; intent < 4; ++intent) {
    auto transform =
        cmsCreateTransform(xyz, TYPE_XYZ_DBL, dst, TYPE_CMYK_FLT, intent,
                           cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
    Check(transform, "native linear-to-CMYK transform");
    const unsigned selected = intent == 3 ? 1 : intent;
    Bytes records;
    unsigned distinct_black = 0;
    for (unsigned p = 0; p < 3 * kPixels; ++p) {
      const connection::Vector input{linear[p * 6], linear[p * 6 + 1],
                                     linear[p * 6 + 2]};
      const auto value = connection::Apply(matrix, input);
      std::array<float, 4> native;
      cmsDoTransform(transform, value.data(), native.data(), 1);
      auto primary = Pcs(matrix, input, true);
      auto native_bound = Pcs(matrix, input, false);
      if (intent == 0 || intent == 2) {
        const auto compensation = BlackConnection(
            Values(3, {0, 0}), Values(3, {0, 0}), {.00336, .0034731, .0028646});
        primary = compensation.apply(primary, false);
        native_bound = compensation.apply(native_bound, true);
      }
      if (intent == 3) {
        // The ideal linear endpoint's media white is decimal PCS D50, while the
        // destination retains its exact fixed-point wtpt. Native CMM applies
        // this connection internally; the independent equations apply it
        // explicitly.
        const auto absolute = Diagonal(
            {.9642 / (0xf6d6 / 65536.0), 1, .8249 / (0xd32d / 65536.0)},
            {0, 0, 0});
        primary = absolute.apply(primary, false);
        native_bound = absolute.apply(native_bound, true);
      }
      for (const auto &stage : target.pipelines[selected * 2 + 1].stages) {
        primary = stage.apply(primary, false);
        native_bound = stage.apply(native_bound, true);
      }
      for (unsigned c = 0; c < 4; ++c) {
        Record(records, native[c] / 100, primary[c], native_bound[c]);
        ++components;
      }
      const double stored_ink = 1 - linear[p * 6 + 3 + black];
      if (std::abs(primary[3].x - stored_ink) > primary[3].radius + .01)
        ++distinct_black;
    }
    Check(distinct_black > kPixels,
          "generated K must distinguish stored Black");
    cmsDeleteTransform(transform);
    Save(out / (name + "_" + std::to_string(intent) + ".reference"), records);
  }
  cmsCloseProfile(xyz);
  cmsCloseProfile(dst);
  return components;
}
} // namespace

int main(int argc, char **argv) try {
  namespace fs = std::filesystem;
  Check(argc == 3 && !fs::exists(argv[2]),
        "usage: cmyk_xyb_generator LUT_CORPUS NEW_OUTPUT_DIRECTORY");
  Check(JxlEncoderVersion() == 12000 && JxlDecoderVersion() == 12000,
        "requires libjxl 0.12.0");
  Check(cmsGetEncodedCMMversion() == 2190, "requires Little CMS 2.19");
  const fs::path profiles(argv[1]), out(argv[2]);
  fs::create_directories(out);
  std::ofstream manifest(out / "manifest.json");
  manifest << "{\"width\":17,\"height\":9,\"frames\":3,\"cases\":[";
  unsigned cases = 0, components = 0;
  for (auto format : {Format::Eight, Format::Sixteen, Format::AB})
    for (bool lab : {false, true}) {
      const std::string kind = format == Format::Eight     ? "lut8"
                               : format == Format::Sixteen ? "lut16"
                                                           : "ab";
      const auto profile =
          Build({kind + (lab ? "_lab_4" : "_xyz_4"), format, lab, 4});
      Check(profile.bytes == Read(profiles / (profile.recipe.name + ".icc")),
            "unchanged independently generated CMYK profile");
      for (unsigned mode = 0; mode < 2; ++mode) {
        const unsigned black = (mode + lab) % 2 == 0 ? 0 : 2;
        const auto name =
            profile.recipe.name + (mode == 0 ? "_modular" : "_vardct");
        const auto bytes = Encode(profile.bytes, mode, black);
        Save(out / (name + ".jxl"), bytes);
        const auto linear = Linear(bytes, profile.bytes, black);
        Bytes samples;
        for (float value : linear) {
          Check(std::isfinite(value), "finite native linear and extra samples");
          LE(samples, value);
        }
        Save(out / (name + ".f32"), samples);
        components += References(profile, linear, black, out, name);
        manifest << (cases ? "," : "") << "{\"name\":\"" << name
                 << "\",\"profile\":\"" << profile.recipe.name
                 << "\",\"modular\":" << (mode == 0 ? "true" : "false")
                 << ",\"black\":" << black << '}';
        ++cases;
        std::cerr << name << " black=" << black << '\n';
      }
    }
  manifest << "]}\n";
  Check(bool(manifest) && cases == 12, "CMYK XYB manifest");
  std::cout << cases << " legal XYB sequences; " << components
            << " independent/native components\n";
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
