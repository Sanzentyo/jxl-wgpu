// Offline independent fixtures: native libjxl serializes every header and Modular stream.
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/enc_toc.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/transform.h"

namespace {
struct Case {
  std::string name;
  std::array<int, 3> selectors = {0, 1, 0};
  size_t width = 37, height = 19;
  uint32_t bits = 16, exponent_bits = 0;
  bool gray = false, gaborish = false, associated = false;
  uint32_t epf = 0, upsampling = 1, orientation = 1;
  uint32_t group_size_shift = 1, passes = 1;
  std::vector<uint32_t> extra_factors;
};

std::vector<Case> Cases() {
  std::vector<Case> cases;
  for (int cb = 0; cb < 4; ++cb) {
    for (int y = 0; y < 4; ++y) {
      for (int cr = 0; cr < 4; ++cr) {
        Case test;
        test.name = "sampling_" + std::to_string(cb) + std::to_string(y) + std::to_string(cr);
        test.selectors = {cb, y, cr};
        cases.push_back(test);
      }
    }
  }
  for (uint32_t bits : {8, 12, 31}) {
    Case test;
    test.name = "integer_" + std::to_string(bits);
    test.bits = bits;
    cases.push_back(test);
  }
  for (uint32_t bits : {16, 24, 32}) {
    Case test;
    test.name = "float_" + std::to_string(bits);
    test.bits = bits;
    test.exponent_bits = bits == 16 ? 5 : bits == 24 ? 7 : 8;
    cases.push_back(test);
  }
  for (bool vertical : {false, true}) {
    Case test;
    test.name = vertical ? "thin_vertical" : "thin_horizontal";
    if (vertical) test.width = 1; else test.height = 1;
    cases.push_back(test);
  }
  Case gray;
  gray.name = "gray";
  gray.gray = true;
  cases.push_back(gray);
  for (uint32_t epf : {1, 2, 3}) {
    Case test;
    test.name = "restoration_" + std::to_string(epf);
    test.selectors = {1, 2, 3};
    test.gaborish = true;
    test.epf = epf;
    cases.push_back(test);
  }
  for (uint32_t factor : {2, 4, 8}) {
    Case test;
    test.name = "resampling_" + std::to_string(factor);
    test.width = 53;
    test.height = 35;
    test.bits = 32;
    test.exponent_bits = 8;
    test.upsampling = factor;
    test.extra_factors = {factor, 8};
    test.gaborish = true;
    test.epf = 2;
    cases.push_back(test);
  }
  Case associated;
  associated.name = "associated";
  associated.bits = 32;
  associated.exponent_bits = 8;
  associated.associated = true;
  associated.extra_factors = {1, 1};
  cases.push_back(associated);
  for (uint32_t orientation = 2; orientation <= 8; ++orientation) {
    Case test = associated;
    test.name = "orientation_" + std::to_string(orientation);
    test.orientation = orientation;
    cases.push_back(test);
  }
  for (uint32_t shift = 0; shift <= 3; ++shift) {
    for (bool vertical : {false, true}) {
      Case test;
      test.name = "groups_" + std::to_string(128u << shift) + (vertical ? "_vertical" : "_horizontal");
      test.group_size_shift = shift;
      if (vertical) test.height = (256u << shift) + 3; else test.width = (256u << shift) + 3;
      cases.push_back(test);
    }
  }
  Case prefix;
  prefix.name = "global_prefix";
  prefix.width = 257;
  cases.push_back(prefix);
  prefix.name = "passes_global_prefix";
  prefix.passes = 2;
  cases.push_back(prefix);
  for (bool directional : {false, true}) {
    Case test;
    test.name = directional ? "passes_directional" : "passes_420";
    test.width = 259;
    test.group_size_shift = 0;
    test.passes = 2;
    if (directional) test.selectors = {1, 2, 3};
    cases.push_back(test);
  }
  Case lf;
  lf.name = "lf_extras";
  lf.width = 2051;
  lf.group_size_shift = 0;
  lf.bits = 32;
  lf.exponent_bits = 8;
  lf.extra_factors = {8, 8};
  cases.push_back(lf);
  return cases;
}

int32_t EncodeSample(float value, const jxl::BitDepth& depth) {
  if (!depth.floating_point_sample) {
    return static_cast<int32_t>(std::llround(static_cast<double>(value) * ((uint64_t{1} << depth.bits_per_sample) - 1)));
  }
  int32_t encoded;
  std::memcpy(&encoded, &value, sizeof(encoded));
  if (depth.bits_per_sample == 32) return encoded;
  uint32_t word;
  std::memcpy(&word, &value, sizeof(word));
  const uint32_t sign = (word >> 31) << (depth.bits_per_sample - 1);
  if ((word & 0x7fffffff) == 0) return sign;
  const int exponent = static_cast<int>((word >> 23) & 255) - 127 + (1 << (depth.exponent_bits_per_sample - 1)) - 1;
  if (exponent <= 0 || exponent >= (1 << depth.exponent_bits_per_sample) - 1) {
    throw std::runtime_error("Fixture values must be normal in their declared float format");
  }
  const uint32_t mantissa_bits = depth.bits_per_sample - depth.exponent_bits_per_sample - 1;
  return sign | (static_cast<uint32_t>(exponent) << mantissa_bits) | ((word & 0x7fffff) >> (23 - mantissa_bits));
}

jxl::Status EncodeGroup(const jxl::Image& image, size_t first, size_t tile, size_t gx, size_t gy,
                        int minimum_shift, int maximum_shift, const jxl::ModularOptions& options,
                        size_t stream_id, jxl::BitWriter& writer) {
  JXL_ASSIGN_OR_RETURN(auto group, jxl::Image::Create(image.memory_manager(), tile, tile, image.bitdepth, 0));
  for (size_t c = first; c < image.channel.size(); ++c) {
    const auto& source = image.channel[c];
    const int shift = std::min(source.hshift, source.vshift);
    if (shift < minimum_shift || shift > maximum_shift) continue;
    const size_t width = tile >> source.hshift, height = tile >> source.vshift;
    const size_t x0 = gx * width, y0 = gy * height;
    if (x0 >= source.w || y0 >= source.h) continue;
    JXL_ASSIGN_OR_RETURN(auto channel, jxl::Channel::Create(image.memory_manager(),
        std::min(width, source.w - x0), std::min(height, source.h - y0), source.hshift, source.vshift));
    for (size_t y = 0; y < channel.h; ++y) std::copy_n(source.Row(y0 + y) + x0, channel.w, channel.Row(y));
    group.channel.push_back(std::move(channel));
  }
  return jxl::ModularGenericCompress(group, options, writer, nullptr, jxl::LayerType::ModularGlobal, stream_id);
}

jxl::Status Generate(const std::filesystem::path& output, const Case& test, bool expanded_reference = false) {
  JxlMemoryManager memory = {nullptr,
    [](void*, size_t size) { return std::malloc(size); },
    [](void*, void* address) { std::free(address); }};
  jxl::CodecMetadata metadata;
  JXL_RETURN_IF_ERROR(metadata.size.Set(test.width, test.height));
  metadata.m.color_encoding = jxl::ColorEncoding::SRGB(test.gray);
  metadata.m.xyb_encoded = false;
  metadata.m.SetUintSamples(test.bits);
  metadata.m.bit_depth.floating_point_sample = test.exponent_bits != 0;
  metadata.m.bit_depth.exponent_bits_per_sample = test.exponent_bits;
  metadata.m.orientation = test.orientation;
  if (!test.extra_factors.empty()) {
    jxl::ExtraChannelInfo alpha, depth;
    alpha.type = jxl::ExtraChannel::kAlpha;
    alpha.bit_depth.bits_per_sample = 12;
    alpha.alpha_associated = test.associated;
    depth.type = jxl::ExtraChannel::kDepth;
    depth.bit_depth.bits_per_sample = 32;
    depth.bit_depth.floating_point_sample = true;
    depth.bit_depth.exponent_bits_per_sample = 8;
    metadata.m.extra_channel_info = {alpha, depth};
  }
  metadata.m.num_extra_channels = metadata.m.extra_channel_info.size();
  jxl::FrameHeader frame(&metadata);
  frame.encoding = jxl::FrameEncoding::kModular;
  frame.color_transform = jxl::ColorTransform::kYCbCr;
  frame.loop_filter.gab = test.gaborish;
  frame.loop_filter.epf_iters = test.epf;
  frame.upsampling = test.upsampling;
  frame.extra_channel_upsampling = test.extra_factors;
  frame.group_size_shift = test.group_size_shift;
  frame.passes.num_passes = test.passes;
  if (test.passes == 2) {
    frame.passes.num_downsample = 1;
    frame.passes.downsample[0] = 2;
    frame.passes.last_pass[0] = 0;
    frame.passes.shift[0] = 0;
  }
  constexpr int horizontal[4] = {0, 1, 1, 0};
  constexpr int vertical[4] = {0, 1, 0, 1};
  uint8_t hsample[3], vsample[3];
  for (size_t c = 0; c < 3; ++c) {
    const size_t jpeg_order = c < 2 ? c ^ 1 : c;
    hsample[jpeg_order] = 1 << horizontal[test.selectors[c]];
    vsample[jpeg_order] = 1 << vertical[test.selectors[c]];
  }
  JXL_RETURN_IF_ERROR(frame.chroma_subsampling.Set(hsample, vsample));
  const auto dimensions = frame.ToFrameDimensions();
  JXL_ASSIGN_OR_RETURN(jxl::Image image, jxl::Image::Create(&memory,
    dimensions.xsize, dimensions.ysize, test.bits, 3 + test.extra_factors.size()));
  for (size_t c = 0; c < image.channel.size(); ++c) {
    auto& plane = image.channel[c];
    const jxl::BitDepth* depth;
    if (c < 3) {
      depth = &metadata.m.bit_depth;
      plane.hshift = frame.chroma_subsampling.HShift(c);
      plane.vshift = frame.chroma_subsampling.VShift(c);
      JXL_RETURN_IF_ERROR(plane.shrink(jxl::DivCeil(dimensions.xsize, 1 << plane.hshift),
                                     jxl::DivCeil(dimensions.ysize, 1 << plane.vshift)));
    } else {
      depth = &metadata.m.extra_channel_info[c - 3].bit_depth;
      const auto factor = test.extra_factors[c - 3];
      plane.hshift = plane.vshift = jxl::CeilLog2Nonzero(factor) - jxl::CeilLog2Nonzero(test.upsampling);
      JXL_RETURN_IF_ERROR(plane.shrink(jxl::DivCeil(test.width, factor), jxl::DivCeil(test.height, factor)));
    }
    for (size_t y = 0; y < plane.h; ++y) {
      for (size_t x = 0; x < plane.w; ++x) {
        const int32_t sample = static_cast<int32_t>((x * (311 + c * 73) + y * (997 - c * 97)
                                                    + x * y * 53 + c * 4013) % 40001) - 20000;
        float value = sample / 65535.0f;
        if (depth->floating_point_sample && depth->bits_per_sample != 32) value = std::round(value * 1024.0f) / 1024.0f;
        if (c == 3) value = static_cast<float>((x * 17 + y * 37) % 65) / 64.0f;
        if (c == 4) value = static_cast<float>(static_cast<int>((x * 11 + y * 7) % 57) - 23) / 16.0f;
        plane.Row(y)[x] = EncodeSample(value, *depth);
      }
    }
  }
  if (expanded_reference) {
    // An independent scalar expansion creates an equivalent 4:4:4 stream. It isolates
    // restoration from libjxl's fast-renderer bug for vertically subsampled components.
    // These cases use signed normalized uint16 words or binary32 working words.
    if (test.bits != 16 && test.bits != 32) return JXL_FAILURE("Unsupported reference sample format");
    for (size_t c = 0; c < 3; ++c) {
      const auto& source = image.channel[c];
      JXL_ASSIGN_OR_RETURN(auto full, jxl::Channel::Create(&memory, dimensions.xsize, dimensions.ysize));
      const auto sample = [&](int x, int y) {
        x = std::max(0, std::min(x, static_cast<int>(source.w) - 1));
        y = std::max(0, std::min(y, static_cast<int>(source.h) - 1));
        const int32_t word = source.Row(y)[x];
        if (test.bits == 16) return static_cast<double>(word) / 65535.0;
        float value;
        std::memcpy(&value, &word, sizeof(value));
        return static_cast<double>(value);
      };
      for (size_t y = 0; y < full.h; ++y) {
        for (size_t x = 0; x < full.w; ++x) {
          const int sx = x >> source.hshift, sy = y >> source.vshift;
          const auto horizontal_value = [&](int row) {
            return source.hshift ? 0.75 * sample(sx, row) + 0.25 * sample(sx + (x % 2 ? 1 : -1), row)
                                 : sample(sx, row);
          };
          const float value = source.vshift ? 0.75 * horizontal_value(sy) + 0.25 * horizontal_value(sy + (y % 2 ? 1 : -1))
                                           : horizontal_value(sy);
          std::memcpy(&full.Row(y)[x], &value, sizeof(value));
        }
      }
      image.channel[c] = std::move(full);
    }
    metadata.m.SetFloat32Samples();
    image.bitdepth = 32;
    const uint8_t equal[3] = {1, 1, 1};
    JXL_RETURN_IF_ERROR(frame.chroma_subsampling.Set(equal, equal));
  }
  jxl::BitWriter writer(&memory);
  JXL_RETURN_IF_ERROR(jxl::WriteCodestreamHeaders(&metadata, &writer, nullptr));
  writer.ZeroPadToByte();
  JXL_RETURN_IF_ERROR(jxl::WriteFrameHeader(frame, &writer, nullptr));
  std::vector<std::unique_ptr<jxl::BitWriter>> sections;
  const bool single_section = dimensions.num_groups == 1 && test.passes == 1;
  const size_t count = single_section ? 1 : 2 + dimensions.num_dc_groups + dimensions.num_groups * test.passes;
  for (size_t i = 0; i < count; ++i) sections.push_back(std::make_unique<jxl::BitWriter>(&memory));
  auto& payload = *sections[0];
  JXL_RETURN_IF_ERROR(payload.WithMaxBits(2, jxl::LayerType::Header, nullptr, [&] {
    payload.Write(1, 1);  // Default LF channel dequantization.
    payload.Write(1, 0);  // Each Modular stream carries its native local tree.
    return true;
  }));
  jxl::ModularOptions options;
  options.predictor = jxl::Predictor::Weighted;
  if (test.bits == 32 || !test.extra_factors.empty() || expanded_reference) {
    // Binary32 sign transitions can exceed libjxl's signed predictive-residual range.
    // A native zero-predictor tree represents every original working word exactly.
    options.predictor = jxl::Predictor::Zero;
    options.tree_kind = jxl::ModularOptions::TreeKind::kTrivialTreeNoPredictor;
  }
  auto global_options = options;
  global_options.max_chan_size = dimensions.group_dim;
  JXL_RETURN_IF_ERROR(jxl::ModularGenericCompress(image, global_options, payload));
  size_t first = 0;
  while (first < image.channel.size() && image.channel[first].w <= dimensions.group_dim && image.channel[first].h <= dimensions.group_dim) ++first;
  for (size_t group = 0; group < dimensions.num_dc_groups; ++group) {
    auto& section = *sections[single_section ? 0 : 1 + group];
    JXL_RETURN_IF_ERROR(EncodeGroup(image, first, dimensions.dc_group_dim,
        group % dimensions.xsize_dc_groups, group / dimensions.xsize_dc_groups, 3, 30, options,
        jxl::ModularStreamId::ModularDC(group).ID(dimensions), section));
  }
  for (size_t pass = 0; pass < test.passes; ++pass) {
    int minimum, maximum;
    frame.passes.GetDownsamplingBracket(pass, minimum, maximum);
    for (size_t group = 0; group < dimensions.num_groups; ++group) {
      auto& section = *sections[single_section ? 0 : 2 + dimensions.num_dc_groups + pass * dimensions.num_groups + group];
      JXL_RETURN_IF_ERROR(EncodeGroup(image, first, dimensions.group_dim,
          group % dimensions.xsize_groups, group / dimensions.xsize_groups, minimum, maximum, options,
          jxl::ModularStreamId::ModularAC(group, pass).ID(dimensions), section));
    }
  }
  std::vector<size_t> sizes;
  for (auto& section : sections) {
    section->ZeroPadToByte();
    sizes.push_back(section->BitsWritten() / 8);
  }
  JXL_RETURN_IF_ERROR(jxl::WriteTocPermutation({}, &writer, nullptr));
  JXL_RETURN_IF_ERROR(jxl::WriteTocSizes(sizes, &writer, nullptr));
  JXL_RETURN_IF_ERROR(writer.AppendByteAligned(sections));
  const auto bytes = std::move(writer).TakeBytes();
  std::ofstream file(output / (test.name + (expanded_reference ? ".expanded.jxl" : ".jxl")), std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!file) return JXL_FAILURE("Cannot write fixture");
  return true;
}
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::filesystem::create_directories(argv[1]);
  for (const auto& test : Cases()) {
    std::fprintf(stderr, "%s\n", test.name.c_str());
    if (!Generate(argv[1], test)) return 1;
    if ((test.gaborish || test.epf != 0) && !Generate(argv[1], test, true)) return 1;
  }
}
