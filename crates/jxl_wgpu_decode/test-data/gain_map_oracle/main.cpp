// Offline interoperability oracle. Production code never links either CPU codec.
#include <jxl/cms.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/gain_map.h>
#include "ultrahdr/gainmapmath.h"
#include "ultrahdr/gainmapmetadata.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

using Bytes = std::vector<uint8_t>;
namespace fs = std::filesystem;
void Check(bool condition, const char* message) {
  if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}
void Uhdr(uhdr_error_info_t status) {
  if (status.error_code != UHDR_CODEC_OK) {
    std::cerr << status.detail << '\n'; std::exit(1);
  }
}
Bytes Read(const fs::path& path) {
  std::ifstream file(path, std::ios::binary); Check(bool(file), "read");
  return Bytes(std::istreambuf_iterator<char>(file), {});
}
void Write(const fs::path& path, const Bytes& bytes) {
  std::ofstream file(path, std::ios::binary); Check(bool(file), "write");
  file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()); Check(bool(file), "write bytes");
}
void Floats(const fs::path& path, const std::vector<float>& values) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  Bytes bytes(values.size() * 4);
  for (size_t i = 0; i < values.size(); ++i) {
    uint32_t bits; std::memcpy(&bits, &values[i], 4);
    for (size_t c = 0; c < 4; ++c) bytes[i * 4 + c] = uint8_t(bits >> (8 * c));
  }
  Write(path, bytes);
}
JxlColorEncoding Color(bool gray, bool linear, bool wide) {
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, gray);
  color.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
  if (linear) color.transfer_function = JXL_TRANSFER_FUNCTION_LINEAR;
  if (wide) color.primaries = JXL_PRIMARIES_2100;
  return color;
}
Bytes Encode(int mode, bool map, bool gray, uint32_t width, uint32_t height, int orientation) {
  auto* encoder = JxlEncoderCreate(nullptr); Check(encoder != nullptr, "encoder");
  auto enc = [](JxlEncoderStatus s) { Check(s == JXL_ENC_SUCCESS, "encode status"); };
  JxlBasicInfo info; JxlEncoderInitBasicInfo(&info);
  info.xsize = width; info.ysize = height; info.bits_per_sample = 8;
  info.num_color_channels = gray ? 1 : 3; info.uses_original_profile = mode < 2;
  info.num_extra_channels = map ? 0 : 1; info.alpha_bits = map ? 0 : 8;
  info.intensity_target = 203; info.orientation = static_cast<JxlOrientation>(orientation);
  enc(JxlEncoderSetBasicInfo(encoder, &info));
  const auto color = Color(gray, false, false); enc(JxlEncoderSetColorEncoding(encoder, &color));
  auto* settings = JxlEncoderFrameSettingsCreate(encoder, nullptr);
  const bool modular = mode % 2 == 0;
  enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_MODULAR, modular));
  enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 3));
  enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1));
  enc(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM, mode < 2 ? 1 : 0));
  enc(JxlEncoderSetFrameDistance(settings, 0.5f));
  if (mode == 0) enc(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
  for (auto option : {JXL_ENC_FRAME_SETTING_PATCHES, JXL_ENC_FRAME_SETTING_DOTS,
      JXL_ENC_FRAME_SETTING_NOISE, JXL_ENC_FRAME_SETTING_GABORISH, JXL_ENC_FRAME_SETTING_EPF,
      JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC}) enc(JxlEncoderFrameSettingsSetOption(settings, option, 0));
  const uint32_t channels = info.num_color_channels + info.num_extra_channels;
  Bytes pixels(size_t(width) * height * channels);
  for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x)
    for (uint32_t c = 0; c < channels; ++c)
      pixels[(size_t(y) * width + x) * channels + c] = static_cast<uint8_t>((x * (31 + c * 17) + y * (53 + c * 13) + (x ^ y) * 7 + c * 51) % 256);
  const JxlPixelFormat format = {channels, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  enc(JxlEncoderAddImageFrame(settings, &format, pixels.data(), pixels.size()));
  JxlEncoderCloseInput(encoder);
  Bytes output(1 << 20); auto* next = output.data(); size_t remaining = output.size();
  enc(JxlEncoderProcessOutput(encoder, &next, &remaining)); output.resize(output.size() - remaining);
  JxlEncoderDestroy(encoder); return output;
}
std::vector<float> Decode(const Bytes& bytes, JxlColorEncoding color) {
  auto* decoder = JxlDecoderCreate(nullptr); Check(decoder != nullptr, "decoder");
  auto dec = [](JxlDecoderStatus s) { Check(s == JXL_DEC_SUCCESS, "decode status"); };
  dec(JxlDecoderSetCms(decoder, *JxlGetDefaultCms()));
  dec(JxlDecoderSubscribeEvents(decoder, JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE));
  dec(JxlDecoderSetKeepOrientation(decoder, JXL_TRUE));
  dec(JxlDecoderSetUnpremultiplyAlpha(decoder, JXL_TRUE));
  dec(JxlDecoderSetInput(decoder, bytes.data(), bytes.size())); JxlDecoderCloseInput(decoder);
  const JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  std::vector<float> output;
  for (;;) {
    const auto status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_COLOR_ENCODING) dec(JxlDecoderSetOutputColorProfile(decoder, &color, nullptr, 0));
    else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t size; dec(JxlDecoderImageOutBufferSize(decoder, &format, &size)); output.resize(size / 4);
      dec(JxlDecoderSetImageOutBuffer(decoder, &format, output.data(), size));
    } else if (status == JXL_DEC_FULL_IMAGE) continue;
    else if (status == JXL_DEC_SUCCESS) break;
    else Check(false, "image decoding");
  }
  JxlDecoderDestroy(decoder); return output;
}
double Sample(const std::vector<float>& map, uint32_t w, uint32_t h, uint32_t x, uint32_t y, uint32_t c) {
  const double px = std::clamp((double(x) + 0.5) * w / 17 - 0.5, 0.0, double(w - 1));
  const double py = std::clamp((double(y) + 0.5) * h / 9 - 0.5, 0.0, double(h - 1));
  const auto x0 = uint32_t(px), y0 = uint32_t(py);
  const auto x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const auto value = [&](uint32_t a, uint32_t b) { return map[(size_t(b) * w + a) * 4 + c]; };
  const double fx = px - x0, fy = py - y0;
  const double top = (1 - fx) * value(x0, y0) + fx * value(x1, y0);
  const double bottom = (1 - fx) * value(x0, y1) + fx * value(x1, y1);
  return std::clamp((1 - fy) * top + fy * bottom, 0.0, 1.0);
}
void Box(Bytes& bytes, const char* type, const Bytes& payload) {
  const uint32_t size = uint32_t(payload.size()) + 8;
  for (int shift : {24, 16, 8, 0}) bytes.push_back(uint8_t(size >> shift));
  bytes.insert(bytes.end(), type, type + 4); bytes.insert(bytes.end(), payload.begin(), payload.end());
}
void Generate(const fs::path& directory) {
  fs::create_directories(directory); std::ofstream manifest(directory / "cases.txt");
  size_t index = 0;
  for (int base_mode = 0; base_mode < 4; ++base_mode)
    for (int map_mode = 0; map_mode < 4; ++map_mode)
      for (bool gray : {false, true}) for (bool wide : {false, true}) {
        const uint32_t widths[] = {1, 17, 7, 29}, heights[] = {1, 9, 5, 13};
        const uint32_t w = widths[(index / 4 + index) % 4], h = heights[(index / 4 + index) % 4];
        const int orientation = int(index % 8) + 1;
        const auto name = "case_" + std::to_string(index++);
        const Bytes base = Encode(base_mode, false, false, 17, 9, orientation);
        const Bytes map = Encode(map_mode, true, gray, w, h, 1);
        // Decode in source primaries: CMS connections can choose a different extension below
        // black. Explicit linear-primary conversion preserves this crate's unbounded contract.
        const auto base_pixels = Decode(base, Color(false, true, false));
        const auto map_pixels = Decode(map, Color(gray, false, false));
        ultrahdr::uhdr_gainmap_metadata_frac metadata;
        metadata.useBaseColorSpace = !wide;
        metadata.alternateHdrHeadroomN = 2;
        for (int c = 0; c < 3; ++c) {
          if (index % 3 == 0) continue; // canonical one-channel/common-denominator records
          metadata.gainMapMinN[c] = -c; metadata.gainMapMinD[c] = 4;
          metadata.gainMapMaxN[c] = 4 + c; metadata.gainMapMaxD[c] = 2;
          metadata.gainMapGammaN[c] = 1 + c; metadata.gainMapGammaD[c] = 2;
          metadata.baseOffsetN[c] = 1 + c; metadata.baseOffsetD[c] = 64;
          metadata.alternateOffsetN[c] = 1 + c; metadata.alternateOffsetD[c] = 128;
        }
        Bytes iso; Uhdr(ultrahdr::uhdr_gainmap_metadata_frac::encodeGainmapMetadata(&metadata, iso));
        JxlGainMapBundle bundle{};
        bundle.gain_map_metadata_size = uint16_t(iso.size()); bundle.gain_map_metadata = iso.data();
        bundle.has_color_encoding = JXL_TRUE; bundle.color_encoding = Color(false, false, wide);
        bundle.gain_map_size = uint32_t(map.size()); bundle.gain_map = map.data();
        size_t size; Check(JxlGainMapGetBundleSize(&bundle, &size), "bundle size");
        Bytes payload(size); size_t written;
        Check(JxlGainMapWriteBundle(&bundle, payload.data(), payload.size(), &written) && written == size, "bundle write");
        Bytes container;
        Box(container, "JXL ", {13, 10, 135, 10}); Box(container, "ftyp", {'j','x','l',' ',0,0,0,0,'j','x','l',' '});
        Box(container, "jhgm", payload); Box(container, "jxlc", base);
        ultrahdr::uhdr_gainmap_metadata_ext_t floating;
        Uhdr(ultrahdr::uhdr_gainmap_metadata_frac::gainmapMetadataFractionToFloat(&metadata, &floating));
        auto working_pixels = base_pixels;
        auto expected = base_pixels;
        for (uint32_t y = 0; y < 9; ++y) for (uint32_t x = 0; x < 17; ++x) {
          const size_t p = (y * 17 + x) * 4;
          ultrahdr::Color source{{{base_pixels[p], base_pixels[p + 1], base_pixels[p + 2]}}};
          if (wide) source = ultrahdr::bt709ToBt2100(source);
          working_pixels[p] = source.r; working_pixels[p + 1] = source.g; working_pixels[p + 2] = source.b;
          const ultrahdr::Color gain{{{float(Sample(map_pixels, w, h, x, y, 0)),
              float(Sample(map_pixels, w, h, x, y, 1)), float(Sample(map_pixels, w, h, x, y, 2))}}};
          const auto result = ultrahdr::applyGain(source, gain, &floating);
          expected[p] = result.r; expected[p + 1] = result.g; expected[p + 2] = result.b;
        }
        Write(directory / (name + ".jxl"), container);
        Floats(directory / (name + ".base.f32"), base_pixels);
        Floats(directory / (name + ".working.f32"), working_pixels);
        Floats(directory / (name + ".gain.f32"), map_pixels);
        Floats(directory / (name + ".expected.f32"), expected);
        manifest << name << ' ' << base_mode << ' ' << map_mode << ' ' << gray << ' ' << wide << ' ' << w << ' ' << h << ' ' << orientation << '\n';
      }
  std::cerr << "generated " << index << " independent gain-map cases\n";
}
int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "generate") { Generate(argv[2]); return 0; }
  Check(argc == 4, "usage: oracle generate directory | iso/bundle input output");
  const auto bytes = Read(argv[2]); Bytes output;
  if (std::string(argv[1]) == "iso") {
    ultrahdr::uhdr_gainmap_metadata_frac metadata;
    Uhdr(ultrahdr::uhdr_gainmap_metadata_frac::decodeGainmapMetadata(bytes, &metadata));
    Uhdr(ultrahdr::uhdr_gainmap_metadata_frac::encodeGainmapMetadata(&metadata, output));
  } else if (std::string(argv[1]) == "bundle") {
    JxlGainMapBundle bundle{}; size_t read;
    Check(JxlGainMapReadBundle(&bundle, bytes.data(), bytes.size(), &read) && read == bytes.size(), "bundle read");
    size_t size; Check(JxlGainMapGetBundleSize(&bundle, &size), "bundle size"); output.resize(size);
    size_t written; Check(JxlGainMapWriteBundle(&bundle, output.data(), size, &written) && written == size, "bundle write");
  } else Check(false, "operation");
  Write(argv[3], output);
}
