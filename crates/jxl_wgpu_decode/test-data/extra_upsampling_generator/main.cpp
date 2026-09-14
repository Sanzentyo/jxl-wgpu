// Offline fixtures. libjxl encodes Modular samples; the frame header is written
// explicitly because libjxl 0.12.0 rejects effective extra-channel upsampling
// above eight.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <jxl/color_encoding.h>
#include <jxl/encode.h>

#include "lib/jxl/dec_modular.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/enc_toc.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/modular_image.h"

namespace {
void Check(JxlEncoderStatus status) {
  if (status != JXL_ENC_SUCCESS) {
    std::fprintf(stderr, "Native fixture encoder status %d\n", status);
    std::abort();
  }
}

// Constant 8x8 input tiles survive the native encoder's box downsampling
// exactly: all extra values and all intermediate sums are small dyadic values.
void VarDct(const std::filesystem::path &directory, size_t width,
            size_t height) {
  const auto name =
      "vardct_" + std::to_string(width) + "x" + std::to_string(height);
  auto *encoder = JxlEncoderCreate(nullptr);
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = width;
  info.ysize = height;
  info.bits_per_sample = 12;
  info.num_extra_channels = 4;
  info.uses_original_profile = JXL_FALSE;
  Check(JxlEncoderSetCodestreamLevel(encoder, 10));
  Check(JxlEncoderSetBasicInfo(encoder, &info));
  const std::array<JxlExtraChannelType, 4> types = {
      JXL_CHANNEL_DEPTH, JXL_CHANNEL_ALPHA, JXL_CHANNEL_SPOT_COLOR,
      JXL_CHANNEL_SELECTION_MASK};
  for (size_t c = 0; c < types.size(); ++c) {
    JxlExtraChannelInfo extra;
    JxlEncoderInitExtraChannelInfo(types[c], &extra);
    extra.bits_per_sample = 32;
    extra.exponent_bits_per_sample = 8;
    extra.dim_shift = 3;
    extra.spot_color[0] = 0.25f;
    extra.spot_color[1] = 0.5f;
    extra.spot_color[2] = 0.75f;
    extra.spot_color[3] = 0.5f;
    Check(JxlEncoderSetExtraChannelInfo(encoder, c, &extra));
  }
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
  Check(JxlEncoderSetColorEncoding(encoder, &color));
  auto *settings = JxlEncoderFrameSettingsCreate(encoder, nullptr);
  Check(JxlEncoderFrameSettingsSetOption(settings,
                                         JXL_ENC_FRAME_SETTING_MODULAR, 0));
  Check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT,
                                         7));
  Check(JxlEncoderFrameSettingsSetOption(
      settings, JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 0));
  Check(JxlEncoderFrameSettingsSetOption(settings,
                                         JXL_ENC_FRAME_SETTING_PATCHES, 0));
  Check(JxlEncoderFrameSettingsSetOption(
      settings, JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING, 8));
  Check(JxlEncoderSetFrameDistance(settings, 1.0f));
  for (size_t c = 0; c < 4; ++c)
    Check(JxlEncoderSetExtraChannelDistance(settings, c, 0.0f));
  const size_t pixels = width * height;
  std::vector<float> samples(pixels * 3);
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x)
      for (size_t c = 0; c < 3; ++c)
        samples[(y * width + x) * 3 + c] =
            ((x * 11 + y * 17 + c * 53) % 257) / 256.0f;
  const JxlPixelFormat rgb = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  Check(JxlEncoderAddImageFrame(settings, &rgb, samples.data(),
                                samples.size() * sizeof(float)));
  const JxlPixelFormat scalar = {1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  for (size_t c = 0; c < 4; ++c) {
    for (size_t y = 0; y < height; ++y)
      for (size_t x = 0; x < width; ++x) {
        const int code = ((x / 8) * 11 + (y / 8) * 17 + c * 53) % 97;
        samples[y * width + x] = (c == 1 ? code % 17 : code - 43) / 16.0f;
      }
    Check(JxlEncoderSetExtraChannelBuffer(settings, &scalar, samples.data(),
                                          pixels * sizeof(float), c));
  }
  JxlEncoderCloseInput(encoder);
  std::ofstream file(directory / (name + ".jxl"), std::ios::binary);
  JxlEncoderStatus status;
  do {
    std::array<uint8_t, 16384> buffer;
    auto *next = buffer.data();
    size_t available = buffer.size();
    status = JxlEncoderProcessOutput(encoder, &next, &available);
    if (status != JXL_ENC_SUCCESS && status != JXL_ENC_NEED_MORE_OUTPUT) {
      std::fprintf(stderr, "%s: native output status %d, encoder error %d\n",
                   name.c_str(), status, JxlEncoderGetError(encoder));
      std::abort();
    }
    file.write(reinterpret_cast<const char *>(buffer.data()),
               buffer.size() - available);
  } while (status == JXL_ENC_NEED_MORE_OUTPUT);
  JxlEncoderDestroy(encoder);
  if (!file)
    std::abort();
}

void Word(std::ostream &out, uint32_t word) {
  for (int i = 0; i < 4; ++i)
    out.put(static_cast<char>(word >> (8 * i)));
}

jxl::Status Header(const jxl::FrameHeader &frame, jxl::BitWriter &writer) {
  JXL_RETURN_IF_ERROR(
      writer.WithMaxBits(128, jxl::LayerType::Header, nullptr, [&] {
        writer.Write(1, 0); // Explicit frame header.
        writer.Write(2, 0); // Regular frame.
        writer.Write(1, 1); // Modular.
        writer.Write(2, 0); // No feature flags.
        writer.Write(1, 0); // Original RGB, no YCbCr.
        writer.Write(2, jxl::CeilLog2Nonzero(frame.upsampling));
        for (size_t i = 0; i < frame.extra_channel_upsampling.size(); ++i) {
          writer.Write(
              2, jxl::CeilLog2Nonzero(frame.extra_channel_upsampling[i]) - 3);
        }
        writer.Write(2, 1); // 256-pixel Modular groups.
        writer.Write(2, 0); // One pass.
        writer.Write(1, 0); // Full canvas.
        for (size_t i = 0; i <= frame.extra_channel_upsampling.size(); ++i)
          writer.Write(2, 0);
        writer.Write(1, 1); // Last frame.
        writer.Write(2, 0); // Empty name.
        return true;
      }));
  JXL_RETURN_IF_ERROR(jxl::Bundle::Write(frame.loop_filter, &writer,
                                         jxl::LayerType::Header, nullptr));
  return writer.WithMaxBits(2, jxl::LayerType::Header, nullptr, [&] {
    writer.Write(2, 0); // No extensions.
    return true;
  });
}

jxl::Status Group(const jxl::Image &image, size_t first, size_t tile, size_t gx,
                  size_t gy, int minimum, int maximum,
                  const jxl::ModularOptions &options, size_t stream_id,
                  jxl::BitWriter &writer) {
  JXL_ASSIGN_OR_RETURN(auto group,
                       jxl::Image::Create(image.memory_manager(), tile, tile,
                                          image.bitdepth, 0));
  for (size_t c = first; c < image.channel.size(); ++c) {
    const auto &source = image.channel[c];
    const int shift = std::min(source.hshift, source.vshift);
    if (shift < minimum || shift > maximum)
      continue;
    const size_t width = tile >> source.hshift, height = tile >> source.vshift;
    const size_t x0 = gx * width, y0 = gy * height;
    if (x0 >= source.w || y0 >= source.h)
      continue;
    JXL_ASSIGN_OR_RETURN(auto channel,
                         jxl::Channel::Create(image.memory_manager(),
                                              std::min(width, source.w - x0),
                                              std::min(height, source.h - y0),
                                              source.hshift, source.vshift));
    for (size_t y = 0; y < channel.h; ++y)
      std::copy_n(source.Row(y0 + y) + x0, channel.w, channel.Row(y));
    group.channel.push_back(std::move(channel));
  }
  return jxl::ModularGenericCompress(group, options, writer, nullptr,
                                     jxl::LayerType::ModularGlobal, stream_id);
}

jxl::Status Generate(const std::filesystem::path &directory, size_t width,
                     size_t height, uint32_t color_factor,
                     uint32_t extra_factor, const std::string &name) {
  JxlMemoryManager memory = {
      nullptr, [](void *, size_t size) { return std::malloc(size); },
      [](void *, void *address) { std::free(address); }};
  jxl::CodecMetadata metadata;
  JXL_RETURN_IF_ERROR(metadata.size.Set(width, height));
  metadata.m.color_encoding = jxl::ColorEncoding::SRGB(false);
  metadata.m.xyb_encoded = false;
  metadata.m.SetUintSamples(12);
  metadata.m.modular_16_bit_buffer_sufficient = false;
  std::array<jxl::ExtraChannel, 4> types = {
      jxl::ExtraChannel::kDepth, jxl::ExtraChannel::kAlpha,
      jxl::ExtraChannel::kSpotColor, jxl::ExtraChannel::kSelectionMask};
  for (size_t c = 0; c < types.size(); ++c) {
    jxl::ExtraChannelInfo extra;
    extra.type = types[c];
    extra.dim_shift = 3; // Combined with the frame's 1/2/4/8 factor.
    extra.bit_depth.bits_per_sample = c == 1 ? 12 : c == 3 ? 17 : 32;
    extra.bit_depth.floating_point_sample = c == 0 || c == 2;
    extra.bit_depth.exponent_bits_per_sample =
        extra.bit_depth.floating_point_sample ? 8 : 0;
    extra.alpha_associated = false;
    constexpr std::array<float, 4> spot = {0.25f, 0.5f, 0.75f, 0.5f};
    std::copy(spot.begin(), spot.end(), extra.spot_color);
    metadata.m.extra_channel_info.push_back(extra);
  }
  metadata.m.num_extra_channels = types.size();
  jxl::FrameHeader frame(&metadata);
  frame.encoding = jxl::FrameEncoding::kModular;
  frame.color_transform = jxl::ColorTransform::kNone;
  frame.upsampling = color_factor;
  frame.extra_channel_upsampling.assign(types.size(), extra_factor);
  frame.loop_filter.gab = false;
  frame.loop_filter.epf_iters = 0;
  frame.loop_filter.nonserialized_is_modular = true;
  const auto dimensions = frame.ToFrameDimensions();
  JXL_ASSIGN_OR_RETURN(auto image, jxl::Image::Create(&memory, dimensions.xsize,
                                                      dimensions.ysize, 12, 7));
  std::ofstream source_file(directory / (name + ".coded"), std::ios::binary);
  for (size_t c = 0; c < image.channel.size(); ++c) {
    auto &plane = image.channel[c];
    const uint32_t factor = c < 3 ? color_factor : extra_factor;
    plane.hshift = plane.vshift =
        jxl::CeilLog2Nonzero(factor) - jxl::CeilLog2Nonzero(color_factor);
    JXL_RETURN_IF_ERROR(plane.shrink(jxl::DivCeil(width, factor),
                                     jxl::DivCeil(height, factor)));
    const auto &depth = c < 3 ? metadata.m.bit_depth
                              : metadata.m.extra_channel_info[c - 3].bit_depth;
    for (size_t y = 0; y < plane.h; ++y) {
      for (size_t x = 0; x < plane.w; ++x) {
        const uint32_t code = static_cast<uint32_t>(
            (x * 311 + y * 997 + x * y * 53 + c * 4013) % 65521);
        uint32_t word;
        if (depth.floating_point_sample) {
          const float value = (static_cast<int>(code % 97) - 43) / 16.0f;
          std::memcpy(&word, &value, 4);
        } else {
          word = code & ((uint32_t{1} << depth.bits_per_sample) - 1);
        }
        std::memcpy(&plane.Row(y)[x], &word, 4);
        Word(source_file, word);
      }
    }
  }
  if (!source_file)
    return JXL_FAILURE("Cannot write source words");
  jxl::BitWriter writer(&memory);
  JXL_RETURN_IF_ERROR(jxl::WriteCodestreamHeaders(&metadata, &writer, nullptr));
  writer.ZeroPadToByte();
  JXL_RETURN_IF_ERROR(Header(frame, writer));
  const bool single = dimensions.num_groups == 1;
  const size_t count =
      single ? 1 : 2 + dimensions.num_dc_groups + dimensions.num_groups;
  std::vector<std::unique_ptr<jxl::BitWriter>> sections;
  for (size_t i = 0; i < count; ++i)
    sections.push_back(std::make_unique<jxl::BitWriter>(&memory));
  auto &global = *sections[0];
  JXL_RETURN_IF_ERROR(
      global.WithMaxBits(2, jxl::LayerType::Header, nullptr, [&] {
        global.Write(1, 1); // Default LF dequantization.
        global.Write(1, 0); // Local trees.
        return true;
      }));
  jxl::ModularOptions options;
  options.predictor = jxl::Predictor::Zero;
  options.tree_kind = jxl::ModularOptions::TreeKind::kTrivialTreeNoPredictor;
  auto global_options = options;
  global_options.max_chan_size = dimensions.group_dim;
  JXL_RETURN_IF_ERROR(
      jxl::ModularGenericCompress(image, global_options, global));
  size_t first = 0;
  while (first < image.channel.size() &&
         image.channel[first].w <= dimensions.group_dim &&
         image.channel[first].h <= dimensions.group_dim)
    ++first;
  for (size_t group = 0; group < dimensions.num_dc_groups; ++group) {
    JXL_RETURN_IF_ERROR(Group(
        image, first, dimensions.dc_group_dim,
        group % dimensions.xsize_dc_groups, group / dimensions.xsize_dc_groups,
        3, 30, options, jxl::ModularStreamId::ModularDC(group).ID(dimensions),
        *sections[single ? 0 : 1 + group]));
  }
  for (size_t group = 0; group < dimensions.num_groups; ++group) {
    JXL_RETURN_IF_ERROR(Group(
        image, first, dimensions.group_dim, group % dimensions.xsize_groups,
        group / dimensions.xsize_groups, 0, 2, options,
        jxl::ModularStreamId::ModularAC(group, 0).ID(dimensions),
        *sections[single ? 0 : 2 + dimensions.num_dc_groups + group]));
  }
  std::vector<size_t> sizes;
  for (auto &section : sections) {
    section->ZeroPadToByte();
    sizes.push_back(section->BitsWritten() / 8);
  }
  JXL_RETURN_IF_ERROR(jxl::WriteTocPermutation({}, &writer, nullptr));
  JXL_RETURN_IF_ERROR(jxl::WriteTocSizes(sizes, &writer, nullptr));
  JXL_RETURN_IF_ERROR(writer.AppendByteAligned(sections));
  const auto bytes = std::move(writer).TakeBytes();
  std::ofstream file(directory / (name + ".jxl"), std::ios::binary);
  file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  return file ? jxl::Status(true) : JXL_FAILURE("Cannot write stream");
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  const std::filesystem::path directory(argv[1]);
  if (std::filesystem::exists(directory))
    return 2;
  std::filesystem::create_directories(directory);
  std::ofstream manifest(directory / "manifest.tsv");
  for (uint32_t extra : {8, 16, 32, 64}) {
    for (uint32_t color : {1, 2, 4, 8}) {
      for (auto [width, height] :
           {std::pair{129, 97}, {1, 65}, {65, 1}, {7, 5}, {2051, 33}}) {
        if (width == 2051 && color != 1)
          continue;
        const auto name =
            "modular_" + std::to_string(width) + "x" + std::to_string(height) +
            "_color" + std::to_string(color) + "_extra" + std::to_string(extra);
        if (!Generate(directory, width, height, color, extra, name))
          return 1;
        manifest << name << '\t' << width << '\t' << height << '\t' << color
                 << '\t' << extra << '\n';
      }
    }
  }
  const auto vardct = directory / "vardct";
  std::filesystem::create_directory(vardct);
  VarDct(vardct, 24, 16);
  VarDct(vardct, 272, 24);
  return manifest ? 0 : 1;
}
