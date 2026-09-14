#include "references.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/transform.h"

namespace {
using Extent = std::array<uint32_t, 2>;

jxl::Status WriteCase(JxlMemoryManager* memory, std::ostream& out,
                     uint32_t kind, uint32_t width, uint32_t height,
                     uint32_t first_blocks, const std::array<Extent, 3>& lf) {
  const uint32_t bx = (width + 7) / 8;
  const uint32_t by = (height + 7) / 8;
  JXL_ASSIGN_OR_RETURN(auto image, jxl::Image::Create(memory, bx, by, 32, 0));
  for (uint32_t c = 0; c < (kind == 0 ? 3u : 4u); ++c) {
    Extent extent = {bx, by};
    int shift = 0;
    if (kind == 0) {
      extent = lf[c];
    } else if (c < 2) {
      extent = {(width + 63) / 64, (height + 63) / 64};
      shift = 3;
    } else if (c == 2) {
      extent = {first_blocks, 2};
    }
    JXL_ASSIGN_OR_RETURN(auto channel,
                         jxl::Channel::Create(memory, extent[0], extent[1], shift, shift));
    // Include negative values, zero, and both signed endpoints. The reference
    // kernel computes differences in pixel_type_w before narrowing to int32.
    constexpr std::array<int32_t, 13> values = {
        0, -7, 13, -31, 49, 2, -11, 97, -111,
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max(), 65535, -65536};
    for (size_t y = 0; y < channel.h; ++y) {
      for (size_t x = 0; x < channel.w; ++x) {
        channel.Row(y)[x] = values[(x * 7 + y * 3 + c * 5) % values.size()];
      }
    }
    image.channel.emplace_back(std::move(channel));
  }
  out << "{\"kind\":" << kind << ",\"width\":" << width
      << ",\"height\":" << height << ",\"first_blocks\":" << first_blocks
      << ",\"lf\":[";
  for (size_t c = 0; c < 3; ++c) {
    if (c) out << ',';
    out << '[' << lf[c][0] << ',' << lf[c][1] << ']';
  }
  out << "],\"channels\":[";
  for (uint32_t c = 0; c < image.channel.size(); ++c) {
    const auto& channel = image.channel[c];
    if (c) out << ',';
    out << "{\"width\":" << channel.w << ",\"height\":" << channel.h
        << ",\"samples\":[";
    for (size_t y = 0; y < channel.h; ++y) {
      for (size_t x = 0; x < channel.w; ++x) {
        if (x || y) out << ',';
        out << channel.Row(y)[x];
      }
    }
    out << "],\"properties\":[";
    JXL_ASSIGN_OR_RETURN(auto references, jxl::Channel::Create(memory, 16, channel.w));
    for (size_t y = 0; y < channel.h; ++y) {
      jxl::PrecomputeReferences(channel, y, image, c, &references);
      for (size_t x = 0; x < channel.w; ++x) {
        for (size_t p = 0; p < 16; ++p) {
          if (x || y || p) out << ',';
          out << references.Row(x)[p];
        }
      }
    }
    out << "]}";
  }
  out << "]}";
  return true;
}
}  // namespace

jxl::Status WriteReferences(JxlMemoryManager* memory,
                            const std::filesystem::path& directory) {
  std::ofstream out(directory / "references.json");
  out << '[';
  constexpr std::array<Extent, 4> extents = {{{4, 4}, {2, 4}, {4, 2}, {2, 2}}};
  for (size_t i = 0; i < 64; ++i) {
    if (i) out << ",\n";
    JXL_RETURN_IF_ERROR(WriteCase(memory, out, 0, 32, 32, 16,
                                 {extents[i / 16], extents[i / 4 % 4], extents[i % 4]}));
  }
  // Coincident dimensions with different shifts, eligible sharpness/strategy
  // pairs, padded strategy row strides, and ordinary multi-cell correlation.
  for (const auto& shape : std::vector<std::array<uint32_t, 3>>{
           {8, 8, 1}, {16, 16, 2}, {64, 16, 8}, {64, 128, 1},
           {128, 128, 2}, {129, 73, 120}}) {
    out << ",\n";
    const Extent blocks = {(shape[0] + 7) / 8, (shape[1] + 7) / 8};
    JXL_RETURN_IF_ERROR(WriteCase(memory, out, 1, shape[0], shape[1], shape[2],
                                 {blocks, blocks, blocks}));
  }
  out << "]\n";
  return bool(out);
}
