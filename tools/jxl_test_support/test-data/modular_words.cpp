// Development-only export of libjxl's integer Modular planes before float conversion.
// The pinned scalar library owns header/entropy parsing, prediction and all transforms.
// This reader accepts original-color one-pass encoder frames with optional global RCT.
// The channel-words mode includes independently sampled scalar LF/pass planes; the original
// raw-word mode retains its equal-grid/common-precision contract. Sampling-header
// and presentation inspection accept either frame codec without decoding image samples.
#include <jxl/decode.h>
#include <jxl/version.h>

#if !defined(HWY_COMPILE_ONLY_SCALAR) || JPEGXL_MAJOR_VERSION != 0 || JPEGXL_MINOR_VERSION != 12 || JPEGXL_PATCH_VERSION != 0
#error "Requires the pinned scalar libjxl 0.12.0 build"
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/icc_codec.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/transform/palette.h"
#include "lib/jxl/toc.h"

namespace {
struct Reader : jxl::BitReader {
  explicit Reader(jxl::Bytes bytes) : BitReader(bytes) {}
  ~Reader() { if (!Close()) std::abort(); }
};

struct Frame {
  uint32_t width, height, channels, bits, exponent;
  std::vector<uint32_t> words;
  std::vector<std::array<uint32_t, 2>> extents;
};

uint32_t Be32(const uint8_t* p) {
  return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3];
}

// Transport extraction only; unsupported transports fail instead of guessing boundaries.
jxl::Status Codestream(const std::vector<uint8_t>& file, std::vector<uint8_t>* raw) {
  JXL_ENSURE(file.size() >= 2);
  if (file[0] == 0xff && file[1] == 0x0a) { *raw = file; return true; }
  const uint8_t signature[] = {0, 0, 0, 12, 'J', 'X', 'L', ' ', 13, 10, 135, 10};
  JXL_ENSURE(file.size() >= sizeof(signature) && std::equal(std::begin(signature), std::end(signature), file.begin()));
  size_t pos = 12;
  while (pos < file.size()) {
    JXL_ENSURE(file.size() - pos >= 8);
    const size_t size = Be32(file.data() + pos);
    JXL_ENSURE(size >= 8 && size <= file.size() - pos);
    if (Be32(file.data() + pos + 4) == 0x6a786c63) {
      JXL_ENSURE(raw->empty());
      raw->assign(file.begin() + pos + 8, file.begin() + pos + size);
    }
    pos += size;
  }
  JXL_ENSURE(!raw->empty());
  return true;
}

jxl::Status FinishSection(Reader& reader) {
  JXL_RETURN_IF_ERROR(reader.JumpToByteBoundary());
  JXL_ENSURE(reader.TotalBitsConsumed() == reader.TotalBytes() * 8);
  return true;
}

// The pinned native ICC reader owns entropy decoding and exact bit consumption.
// Sample-word inspection does not evaluate the profile or replace it with RGB.
jxl::Status ReadICC(JxlMemoryManager* memory, Reader* reader, jxl::CodecMetadata* metadata) {
  if (!metadata->m.color_encoding.WantICC()) return true;
  jxl::ICCReader decoder(memory);
  jxl::PaddedBytes decoded(memory);
  JXL_RETURN_IF_ERROR(decoder.Init(reader));
  JXL_RETURN_IF_ERROR(decoder.Process(reader, &decoded));
  JXL_ENSURE(!decoded.empty());
  jxl::IccBytes icc;
  jxl::Bytes(decoded).AppendTo(icc);
  metadata->m.color_encoding.SetICCRaw(std::move(icc));
  return true;
}

jxl::Status AuditPalette(const jxl::Image& image, const jxl::GroupHeader& header,
                         std::array<uint32_t, 3>* counts) {
  JXL_ENSURE(!header.transforms.empty());
  const auto& transform = header.transforms.back();
  JXL_ENSURE(transform.id == jxl::TransformId::kPalette);
  JXL_ENSURE(image.nb_meta_channels == 1 && transform.begin_c + 1 < image.channel.size());
  const auto& indices = image.channel[transform.begin_c + 1];
  for (size_t y = 0; y < indices.h; ++y) for (size_t x = 0; x < indices.w; ++x) {
    const int32_t index = indices.Row(y)[x];
    ++(*counts)[index < 0 ? 0 : uint32_t(index) >= transform.nb_colors + transform.nb_deltas ? 1 : 2];
  }
  return true;
}

jxl::Status Decode(const std::vector<uint8_t>& raw, std::vector<Frame>* frames,
                   std::array<uint32_t, 3>* audit = nullptr, bool preview = false, bool independent = false) {
  JxlMemoryManager memory{nullptr, [](void*, size_t size) -> void* { return std::malloc(size); }, [](void*, void* p) { std::free(p); }};
  jxl::CodecMetadata metadata;
  size_t pos;
  {
    Reader reader(jxl::Bytes(raw.data(), raw.size()));
    JXL_ENSURE(reader.ReadBits(16) == 0x0aff);
    JXL_RETURN_IF_ERROR(jxl::ReadSizeHeader(&reader, &metadata.size));
    JXL_RETURN_IF_ERROR(jxl::ReadImageMetadata(&reader, &metadata.m));
    metadata.transform_data.nonserialized_xyb_encoded = metadata.m.xyb_encoded;
    JXL_RETURN_IF_ERROR(jxl::Bundle::Read(&reader, &metadata.transform_data));
    JXL_ENSURE(!metadata.m.xyb_encoded && metadata.m.have_preview == preview);
    JXL_RETURN_IF_ERROR(ReadICC(&memory, &reader, &metadata));
    if (!independent) for (const auto& extra : metadata.m.extra_channel_info) {
      JXL_ENSURE(extra.dim_shift == 0);
      JXL_ENSURE(extra.bit_depth.bits_per_sample == metadata.m.bit_depth.bits_per_sample);
      JXL_ENSURE(extra.bit_depth.floating_point_sample == metadata.m.bit_depth.floating_point_sample);
      if (extra.bit_depth.floating_point_sample) JXL_ENSURE(extra.bit_depth.exponent_bits_per_sample == metadata.m.bit_depth.exponent_bits_per_sample);
    }
    JXL_RETURN_IF_ERROR(reader.JumpToByteBoundary());
    pos = reader.TotalBitsConsumed() / 8;
  }
  bool last = false;
  while (!last) {
    JXL_ENSURE(pos < raw.size() && frames->size() < 64);
    jxl::FrameHeader header(&metadata);
    header.nonserialized_is_preview = preview;
    std::vector<uint32_t> sizes;
    std::vector<jxl::coeff_order_t> permutation;
    jxl::FrameDimensions dim;
    {
      Reader reader(jxl::Bytes(raw.data() + pos, raw.size() - pos));
      JXL_RETURN_IF_ERROR(jxl::ReadFrameHeader(&reader, &header));
      dim = header.ToFrameDimensions();
      JXL_ENSURE(header.encoding == jxl::FrameEncoding::kModular && header.color_transform == jxl::ColorTransform::kNone);
      JXL_ENSURE(header.passes.num_passes == 1 && header.dc_level == 0);
      // Native FrameDimensions supplies the coded grid, before presentation resampling.
      // This helper's equal-geometry planes still exclude independently sampled extras.
      JXL_ENSURE(header.upsampling == 1 || header.upsampling == 2 || header.upsampling == 4 || header.upsampling == 8);
      if (!independent) for (auto up : header.extra_channel_upsampling) JXL_ENSURE(up == header.upsampling);
      JXL_ENSURE(dim.xsize != 0 && dim.ysize != 0 && uint64_t(dim.xsize) * dim.ysize <= (1u << 24));
      JXL_RETURN_IF_ERROR(jxl::ReadToc(&memory, jxl::NumTocEntries(dim.num_groups, dim.num_dc_groups, 1), &reader, &sizes, &permutation));
      JXL_ENSURE(permutation.empty());
      JXL_RETURN_IF_ERROR(reader.JumpToByteBoundary());
      pos += reader.TotalBitsConsumed() / 8;
    }
    std::vector<size_t> offsets;
    for (auto size : sizes) {
      JXL_ENSURE(size <= raw.size() - pos);
      offsets.push_back(pos);
      pos += size;
    }
    const size_t channels = (metadata.m.color_encoding.IsGray() ? 1 : 3) + metadata.m.extra_channel_info.size();
    JXL_ENSURE(channels >= 1 && channels <= (independent ? 259u : 4u));
    const int bits = metadata.m.bit_depth.bits_per_sample;
    JXL_ASSIGN_OR_RETURN(jxl::Image image, jxl::Image::Create(&memory, dim.xsize, dim.ysize, bits, 0));
    const size_t color_channels = metadata.m.color_encoding.IsGray() ? 1 : 3;
    uint64_t samples = 0;
    for (size_t c = 0; c < channels; ++c) {
      const size_t factor = c < color_channels ? header.upsampling : header.extra_channel_upsampling[c - color_channels];
      JXL_ENSURE(factor >= header.upsampling && factor <= 64);
      const size_t width = jxl::DivCeil(dim.xsize_upsampled, factor);
      const size_t height = jxl::DivCeil(dim.ysize_upsampled, factor);
      samples += uint64_t(width) * height;
      JXL_ENSURE(samples <= (1u << 26));
      JXL_ASSIGN_OR_RETURN(jxl::Channel channel, jxl::Channel::Create(&memory, width, height));
      channel.hshift = channel.vshift = jxl::CeilLog2Nonzero(factor) - jxl::CeilLog2Nonzero(header.upsampling);
      image.channel.push_back(std::move(channel));
    }
    jxl::Tree tree;
    jxl::ANSCode code;
    std::vector<uint8_t> contexts;
    jxl::GroupHeader global;
    {
      Reader reader(jxl::Bytes(raw.data() + offsets[0], sizes[0]));
      JXL_ENSURE(reader.ReadBits(1) == 1); // default LF dequantization
      JXL_ENSURE(reader.ReadBits(1) == 1); // global tree present
      JXL_RETURN_IF_ERROR(jxl::DecodeTree(&memory, &reader, &tree, 1u << 22));
      JXL_RETURN_IF_ERROR(jxl::DecodeHistograms(&memory, &reader, (tree.size() + 1) / 2, &code, &contexts));
      jxl::ModularOptions options;
      options.max_chan_size = options.group_dim = dim.group_dim;
      JXL_RETURN_IF_ERROR(jxl::ModularGenericDecompress(&reader, image, &global, 0, &options, false, &tree, &code, &contexts));
      // ModularDecode returns before constructing its symbol reader when every channel
      // belongs to later groups. EncodeStream/WriteTokens can nevertheless retain a
      // zero-symbol ANS state. Validate it with libjxl, rather than ignoring trailing bytes.
      bool has_global_samples = false;
      for (size_t c = 0; c < image.channel.size(); ++c) {
        const auto& channel = image.channel[c];
        if (c >= image.nb_meta_channels &&
            (channel.w > options.max_chan_size || channel.h > options.max_chan_size)) break;
        has_global_samples |= channel.w != 0 && channel.h != 0;
      }
      if (!has_global_samples && global.use_global_tree && !code.use_prefix_code &&
          reader.TotalBitsConsumed() + 7 < reader.TotalBytes() * 8) {
        JXL_ASSIGN_OR_RETURN(jxl::ANSSymbolReader empty, jxl::ANSSymbolReader::Create(&code, &reader, 0));
        JXL_ENSURE(reader.AllReadsWithinBounds() && empty.CheckANSFinalState());
      }
      JXL_RETURN_IF_ERROR(FinishSection(reader));
    }
    if (dim.num_groups > 1) {
      JXL_ENSURE(image.nb_meta_channels == 0 && image.channel.size() == channels);
      for (const auto& transform : global.transforms) JXL_ENSURE(transform.id == jxl::TransformId::kRCT);
      const size_t begin = jxl::AcGroupIndex(0, 0, dim.num_groups, dim.num_dc_groups);
      JXL_ENSURE(sizes[begin - 1] == 0); // no HF-global in Modular
      auto decode_group = [&](size_t toc, const jxl::Rect& rect, size_t edge,
                              int min_shift, int max_shift, jxl::ModularStreamId id) -> jxl::Status {
        JXL_ASSIGN_OR_RETURN(jxl::Image part, jxl::Image::Create(&memory, rect.xsize(), rect.ysize(), bits, 0));
        std::vector<size_t> selected;
        bool global_prefix = true;
        for (size_t c = 0; c < channels; ++c) {
          const auto& channel = image.channel[c];
          global_prefix &= channel.w <= dim.group_dim && channel.h <= dim.group_dim;
          const int shift = std::min(channel.hshift, channel.vshift);
          if (global_prefix || shift < min_shift || shift > max_shift) continue;
          const size_t x = rect.x0() >> channel.hshift;
          const size_t y = rect.y0() >> channel.vshift;
          JXL_ENSURE(x < channel.w && y < channel.h);
          const size_t width = std::min(edge >> channel.hshift, channel.w - x);
          const size_t height = std::min(edge >> channel.vshift, channel.h - y);
          JXL_ASSIGN_OR_RETURN(jxl::Channel local, jxl::Channel::Create(&memory, width, height));
          local.hshift = channel.hshift;
          local.vshift = channel.vshift;
          part.channel.push_back(std::move(local));
          selected.push_back(c);
        }
        if (selected.empty()) { JXL_ENSURE(sizes[toc] == 0); return true; }
        Reader reader(jxl::Bytes(raw.data() + offsets[toc], sizes[toc]));
        jxl::ModularOptions options;
        options.group_dim = dim.group_dim;
        jxl::GroupHeader local;
        JXL_RETURN_IF_ERROR(jxl::ModularGenericDecompress(&reader, part, audit ? &local : nullptr,
          id.ID(dim), &options, audit == nullptr, &tree, &code, &contexts));
        JXL_RETURN_IF_ERROR(FinishSection(reader));
        if (audit) {
          JXL_RETURN_IF_ERROR(AuditPalette(part, local, audit));
          part.undo_transforms(local.wp_header);
        }
        JXL_ENSURE(!part.error && part.channel.size() == selected.size() && part.nb_meta_channels == 0);
        for (size_t i = 0; i < selected.size(); ++i) {
          auto& destination = image.channel[selected[i]];
          const auto& decoded = part.channel[i];
          const size_t x = rect.x0() >> destination.hshift;
          const size_t y = rect.y0() >> destination.vshift;
          JXL_ENSURE(decoded.w == std::min(edge >> destination.hshift, destination.w - x));
          JXL_ENSURE(decoded.h == std::min(edge >> destination.vshift, destination.h - y));
          for (size_t row = 0; row < decoded.h; ++row) std::copy_n(decoded.Row(row), decoded.w, destination.Row(y + row) + x);
        }
        return true;
      };
      for (size_t group = 0; group < dim.num_dc_groups; ++group) {
        const jxl::Rect rect((group % dim.xsize_dc_groups) * dim.dc_group_dim,
                             (group / dim.xsize_dc_groups) * dim.dc_group_dim,
                             dim.dc_group_dim, dim.dc_group_dim);
        JXL_RETURN_IF_ERROR(decode_group(1 + group, rect, dim.dc_group_dim,
                                        3, 30, jxl::ModularStreamId::ModularDC(group)));
      }
      for (size_t group = 0; group < dim.num_groups; ++group) {
        JXL_RETURN_IF_ERROR(decode_group(begin + group, dim.GroupRect(group), dim.group_dim,
                                        0, 2, jxl::ModularStreamId::ModularAC(group, 0)));
      }
    }
    if (audit && dim.num_groups == 1) JXL_RETURN_IF_ERROR(AuditPalette(image, global, audit));
    image.undo_transforms(global.wp_header);
    JXL_ENSURE(!image.error && image.nb_meta_channels == 0 && image.channel.size() == channels);
    Frame frame{uint32_t(dim.xsize), uint32_t(dim.ysize), uint32_t(channels), uint32_t(bits), metadata.m.bit_depth.floating_point_sample ? metadata.m.bit_depth.exponent_bits_per_sample : 0, {}, {}};
    for (const auto& channel : image.channel) {
      if (!independent) JXL_ENSURE(channel.w == dim.xsize && channel.h == dim.ysize);
      frame.extents.push_back({uint32_t(channel.w), uint32_t(channel.h)});
      for (size_t y = 0; y < channel.h; ++y) for (size_t x = 0; x < channel.w; ++x) frame.words.push_back(uint32_t(channel.Row(y)[x]));
    }
    frames->push_back(std::move(frame));
    last = header.is_last;
  }
  JXL_ENSURE(preview ? (frames->size() == 1 && pos < raw.size()) : pos == raw.size());
  return true;
}

void Word(uint32_t value) {
  const uint8_t bytes[] = {uint8_t(value), uint8_t(value >> 8), uint8_t(value >> 16), uint8_t(value >> 24)};
  if (fwrite(bytes, 1, 4, stdout) != 4) std::abort();
}

jxl::Status InspectHeaders(const std::vector<uint8_t>& raw,
                           std::vector<std::vector<uint32_t>>* frames,
                           bool presentation) {
  JxlMemoryManager memory{nullptr, [](void*, size_t size) -> void* { return std::malloc(size); }, [](void*, void* p) { std::free(p); }};
  jxl::CodecMetadata metadata;
  size_t pos;
  {
    Reader reader(jxl::Bytes(raw.data(), raw.size()));
    JXL_ENSURE(reader.ReadBits(16) == 0x0aff);
    JXL_RETURN_IF_ERROR(jxl::ReadSizeHeader(&reader, &metadata.size));
    JXL_RETURN_IF_ERROR(jxl::ReadImageMetadata(&reader, &metadata.m));
    JXL_ENSURE(!metadata.m.have_preview);
    metadata.transform_data.nonserialized_xyb_encoded = metadata.m.xyb_encoded;
    JXL_RETURN_IF_ERROR(jxl::Bundle::Read(&reader, &metadata.transform_data));
    JXL_RETURN_IF_ERROR(ReadICC(&memory, &reader, &metadata));
    JXL_RETURN_IF_ERROR(reader.JumpToByteBoundary());
    pos = reader.TotalBitsConsumed() / 8;
  }
  bool last = false;
  while (!last) {
    JXL_ENSURE(pos < raw.size() && frames->size() < 64);
    jxl::FrameHeader header(&metadata);
    std::vector<uint32_t> sizes;
    std::vector<jxl::coeff_order_t> permutation;
    {
      Reader reader(jxl::Bytes(raw.data() + pos, raw.size() - pos));
      JXL_RETURN_IF_ERROR(jxl::ReadFrameHeader(&reader, &header));
      const auto dim = header.ToFrameDimensions();
      JXL_ENSURE(header.dc_level == 0 && header.extra_channel_upsampling.size() <= 256);
      if (presentation) {
        frames->push_back({metadata.m.orientation, uint32_t(header.name.size())});
        for (unsigned char byte : header.name) frames->back().push_back(byte);
      } else {
        frames->push_back({uint32_t(dim.xsize_upsampled), uint32_t(dim.ysize_upsampled),
                         uint32_t(dim.xsize), uint32_t(dim.ysize), header.upsampling,
                         header.passes.num_passes, uint32_t(header.extra_channel_upsampling.size())});
        for (uint32_t factor : header.extra_channel_upsampling) frames->back().push_back(factor);
      }
      JXL_RETURN_IF_ERROR(jxl::ReadToc(&memory, jxl::NumTocEntries(dim.num_groups, dim.num_dc_groups, header.passes.num_passes), &reader, &sizes, &permutation));
      JXL_RETURN_IF_ERROR(reader.JumpToByteBoundary());
      pos += reader.TotalBitsConsumed() / 8;
    }
    for (uint32_t size : sizes) {
      JXL_ENSURE(size <= raw.size() - pos);
      pos += size;
    }
    last = header.is_last;
  }
  JXL_ENSURE(pos == raw.size());
  return true;
}
} // namespace

int main(int argc, char** argv) {
  if (JPEGXL_MAJOR_VERSION != 0 || JPEGXL_MINOR_VERSION != 12 || JPEGXL_PATCH_VERSION != 0 || JxlDecoderVersion() != 12000) return 2;
  if (argc == 3 && std::strcmp(argv[1], "--implicit-entries") == 0) {
    char* end = nullptr;
    const long bits = std::strtol(argv[2], &end, 10);
    if (!end || *end || bits < 1 || bits > 32) return 2;
    if (fwrite("JXLIMP12", 1, 8, stdout) != 8) return 2;
    Word(JxlDecoderVersion());
    for (int index = -143; index < 189; ++index) for (size_t c = 0; c < 4; ++c) {
      Word(uint32_t(jxl::palette_internal::GetPaletteValue(nullptr, index, c, 0, 0, std::min<long>(bits, 24))));
    }
    return 0;
  }
  const bool independent = argc == 3 && std::strcmp(argv[1], "--channel-words") == 0;
  const bool audit = argc == 3 && std::strcmp(argv[1], "--palette-audit") == 0;
  const bool headers = argc == 3 && std::strcmp(argv[1], "--sampling-headers") == 0;
  const bool presentation = argc == 3 && std::strcmp(argv[1], "--presentation-headers") == 0;
  const bool preview = argc == 3 && std::strcmp(argv[1], "--preview-words") == 0;
  if (argc != 2 && !audit && !headers && !presentation && !preview && !independent) return 2;
  std::ifstream input(argv[(audit || headers || presentation || preview || independent) ? 2 : 1], std::ios::binary);
  if (!input) return 2;
  std::vector<uint8_t> file{std::istreambuf_iterator<char>(input), {}};
  if (file.size() > (1u << 26)) return 2;
  std::vector<uint8_t> raw;
  std::vector<Frame> frames;
  std::array<uint32_t, 3> counts{};
  if (!Codestream(file, &raw)) return 1;
  if (headers || presentation) {
    std::vector<std::vector<uint32_t>> sampling;
    if (!InspectHeaders(raw, &sampling, presentation)) return 1;
    if (fwrite(presentation ? "JXLMET12" : "JXLSMP12", 1, 8, stdout) != 8) return 2;
    Word(JxlDecoderVersion());
    Word(sampling.size());
    for (const auto& fields : sampling) for (uint32_t value : fields) Word(value);
    return 0;
  }
  if (!Decode(raw, &frames, audit ? &counts : nullptr, preview, independent)) return 1;
  if (audit) {
    if (fwrite("JXLPAL12", 1, 8, stdout) != 8) return 2;
    Word(JxlDecoderVersion());
    for (uint32_t count : counts) Word(count);
    return 0;
  }
  if (fwrite(independent ? "JXLCHN12" : "JXLRAW12", 1, 8, stdout) != 8) return 2;
  Word(JxlDecoderVersion());
  Word(frames.size());
  for (const auto& frame : frames) {
    for (uint32_t field : {frame.width, frame.height, frame.channels, frame.bits, frame.exponent}) Word(field);
    if (independent) for (auto extent : frame.extents) { Word(extent[0]); Word(extent[1]); }
    for (uint32_t word : frame.words) Word(word);
  }
  return 0;
}
