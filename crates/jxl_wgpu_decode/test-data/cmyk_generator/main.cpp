// Offline CMYK frame composition and independent ICC LUT color references.
#include "original.hpp"
#include <iterator>
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
  info.uses_original_profile = JXL_TRUE;
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
    if (mode == 0)
      Enc(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
    for (const auto [option, value] :
         std::array<std::pair<JxlEncoderFrameSettingId, int>, 9>{
             {{JXL_ENC_FRAME_SETTING_MODULAR, mode == 0},
              {JXL_ENC_FRAME_SETTING_EFFORT, 3},
              {JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM, mode == 2 ? 2 : 1},
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
    header.layer_info.save_as_reference = frame < 2 ? frame + 1 : 0;
    auto &color_blend = header.layer_info.blend_info;
    color_blend.blendmode = frame == 0 ? JXL_BLEND_REPLACE : JXL_BLEND_BLEND;
    color_blend.source = frame;
    color_blend.alpha = 1;
    color_blend.clamp = JXL_FALSE;
    Enc(JxlEncoderSetFrameHeader(settings, &header));
    for (unsigned index = 0; index < 3; ++index) {
      JxlBlendInfo blend = color_blend;
      blend.blendmode = JXL_BLEND_REPLACE;
      if (frame && index == black) {
        blend.blendmode = frame == 1 ? JXL_BLEND_ADD : JXL_BLEND_MUL;
        // The last color frame uses reference 2, while Black still uses 1.
        blend.source = 1;
      }
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

} // namespace

int main(int argc, char **argv) try {
  namespace fs = std::filesystem;
  const bool seeds = argc == 4 && std::string(argv[1]) == "seeds";
  Check(seeds || (argc == 5 && std::string(argv[1]) == "references"),
        "usage: cmyk_generator seeds LUT_CORPUS NEW_OUTPUT_DIRECTORY | "
        "references LUT_CORPUS ASSEMBLED_SOURCES NEW_OUTPUT_DIRECTORY");
  Check(JxlEncoderVersion() == 12000 && JxlDecoderVersion() == 12000,
        "requires libjxl 0.12.0");
  Check(cmsGetEncodedCMMversion() == 2190, "requires Little CMS 2.19");
  const fs::path base(argv[2]), out(argv[seeds ? 3 : 4]);
  Check(!fs::exists(out), "output directory must be new");
  fs::create_directories(out);
  auto profile = [&](Format format, bool lab, unsigned channels) {
    const std::string kind = format == Format::Eight     ? "lut8"
                             : format == Format::Sixteen ? "lut16"
                                                         : "ab";
    auto result =
        Build({kind + (lab ? "_lab_" : "_xyz_") + std::to_string(channels),
               format, lab, channels});
    std::ifstream file(base / (result.recipe.name + ".icc"), std::ios::binary);
    Check(bool(file), "read existing CMYK LUT corpus");
    const Bytes existing(std::istreambuf_iterator<char>{file}, {});
    Check(existing == result.bytes, "unchanged CMYK LUT profile");
    return result;
  };
  std::ofstream manifest(out / "manifest.json");
  manifest << "{\"width\":17,\"height\":9,\"frames\":3,\"cases\":[";
  unsigned cases = 0, components = 0;
  for (auto format : {Format::Eight, Format::Sixteen, Format::AB})
    for (bool lab : {false, true}) {
      const auto source = profile(format, lab, 4);
      for (unsigned mode = 0; mode < 3; ++mode) {
        const unsigned black = (mode + lab) % 2 == 0 ? 0 : 2;
        const unsigned channels = black == 0 ? 3 : 1;
        const auto target =
            profile(format == Format::Eight ? Format::Sixteen : Format::Eight,
                    !lab, channels);
        const auto name = source.recipe.name + "_" + std::to_string(mode);
        manifest << (cases ? "," : "") << "{\"name\":\"" << name
                 << "\",\"source\":\"" << source.recipe.name
                 << "\",\"target\":\"" << target.recipe.name
                 << "\",\"channels\":" << channels << ",\"mode\":" << mode
                 << ",\"black\":" << black << '}';
        ++cases;
        if (seeds) {
          Save(out / (name + ".jxl"), Encode(source.bytes, mode, black));
          continue;
        }
        std::ifstream file(fs::path(argv[3]) / (name + ".jxl"),
                           std::ios::binary);
        Check(bool(file), "read assembled CMYK source");
        const Bytes bytes(std::istreambuf_iterator<char>{file}, {});
        Save(out / (name + ".jxl"), bytes);
        const auto native = Decode(bytes, source.bytes);
        Bytes original;
        for (float value : native) {
          Check(std::isfinite(value), "finite CMYK original sample");
          LE(original, value);
        }
        Save(out / (name + ".f32"), original);
        auto src =
            cmsOpenProfileFromMem(source.bytes.data(), source.bytes.size());
        auto dst =
            cmsOpenProfileFromMem(target.bytes.data(), target.bytes.size());
        Check(src && dst, "native CMYK profiles");
        for (bool spots : {false, true})
          for (unsigned intent = 0; intent < 4; ++intent) {
            auto transform = cmsCreateTransform(
                src, TYPE_CMYK_FLT, dst,
                channels == 1 ? TYPE_GRAY_FLT : TYPE_RGB_FLT, intent,
                cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
            Check(transform, "native CMYK conversion");
            Bytes records;
            const unsigned selected = intent == 3 ? 1 : intent;
            for (unsigned p = 0; p < 3 * kPixels; ++p) {
              Values primary, native_bound;
              std::array<float, 4> ink;
              for (unsigned c = 0; c < 4; ++c) {
                const double sample = native[p * 6 + (c == 3 ? 3 + black : c)];
                const double coverage = native[p * 6 + 3 + (2 - black)];
                const auto value =
                    Ink(sample, coverage, c, spots, mode && c < 3 ? 2e-5 : 0);
                primary.push_back(value);
                auto exact = value;
                exact.radius = 0;
                native_bound.push_back(exact);
                ink[c] = static_cast<float>(100 * value.x);
              }
              std::array<float, 3> converted;
              cmsDoTransform(transform, ink.data(), converted.data(), 1);
              for (const auto *pipeline : {&source.pipelines[selected * 2],
                                           &target.pipelines[selected * 2 + 1]})
                for (const auto &stage : pipeline->stages) {
                  primary = stage.apply(primary, false);
                  native_bound = stage.apply(native_bound, true);
                }
              for (unsigned c = 0; c < channels; ++c) {
                Record(records, converted[c], primary[c], native_bound[c]);
                ++components;
              }
            }
            cmsDeleteTransform(transform);
            Save(out / (name + "_" + std::to_string(spots) + "_" +
                        std::to_string(intent) + ".reference"),
                 records);
          }
        cmsCloseProfile(src);
        cmsCloseProfile(dst);
        std::cerr << name << " black=" << black << '\n';
      }
    }
  manifest << "]}\n";
  Check(bool(manifest), "CMYK manifest");
  std::cout << cases << " CMYK streams; " << components
            << " independent/native components\n";
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
