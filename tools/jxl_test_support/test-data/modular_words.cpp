// Development-only export of libjxl's integer Modular planes before float conversion.
// The pinned scalar library owns header/entropy parsing, prediction and all transforms.
// This reader intentionally accepts only original-color one-pass encoder frames, with
// uniform unresampled components, empty LF/HF groups and optional global RCT.
#include <jxl/decode.h>
#include <jxl/version.h>

#if !defined(HWY_COMPILE_ONLY_SCALAR) || JPEGXL_MAJOR_VERSION != 0 || JPEGXL_MINOR_VERSION != 12 || JPEGXL_PATCH_VERSION != 0
#error "Requires the pinned scalar libjxl 0.12.0 build"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/toc.h"

namespace {
struct Reader : jxl::BitReader {
  explicit Reader(jxl::Bytes bytes) : BitReader(bytes) {}
  ~Reader() { if (!Close()) std::abort(); }
};

struct Frame {
  uint32_t width, height, channels, bits, exponent;
  std::vector<uint32_t> words;
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

jxl::Status Decode(const std::vector<uint8_t>& raw, std::vector<Frame>* frames) {
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
    JXL_ENSURE(!metadata.m.xyb_encoded && !metadata.m.have_preview && !metadata.m.color_encoding.WantICC());
    for (const auto& extra : metadata.m.extra_channel_info) {
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
    std::vector<uint32_t> sizes;
    std::vector<jxl::coeff_order_t> permutation;
    jxl::FrameDimensions dim;
    {
      Reader reader(jxl::Bytes(raw.data() + pos, raw.size() - pos));
      JXL_RETURN_IF_ERROR(jxl::ReadFrameHeader(&reader, &header));
      dim = header.ToFrameDimensions();
      JXL_ENSURE(header.encoding == jxl::FrameEncoding::kModular && header.color_transform == jxl::ColorTransform::kNone);
      JXL_ENSURE(header.passes.num_passes == 1 && header.upsampling == 1 && header.dc_level == 0);
      for (auto up : header.extra_channel_upsampling) JXL_ENSURE(up == 1);
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
    JXL_ENSURE(channels >= 1 && channels <= 4);
    const int bits = metadata.m.bit_depth.bits_per_sample;
    JXL_ASSIGN_OR_RETURN(jxl::Image image, jxl::Image::Create(&memory, dim.xsize, dim.ysize, bits, channels));
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
      JXL_RETURN_IF_ERROR(FinishSection(reader));
    }
    if (dim.num_groups > 1) {
      JXL_ENSURE(image.nb_meta_channels == 0 && image.channel.size() == channels);
      for (const auto& transform : global.transforms) JXL_ENSURE(transform.id == jxl::TransformId::kRCT);
      const size_t begin = jxl::AcGroupIndex(0, 0, dim.num_groups, dim.num_dc_groups);
      for (size_t i = 1; i < begin; ++i) JXL_ENSURE(sizes[i] == 0);
      for (size_t group = 0; group < dim.num_groups; ++group) {
        const auto rect = dim.GroupRect(group);
        JXL_ASSIGN_OR_RETURN(jxl::Image part, jxl::Image::Create(&memory, rect.xsize(), rect.ysize(), bits, channels));
        Reader reader(jxl::Bytes(raw.data() + offsets[begin + group], sizes[begin + group]));
        jxl::ModularOptions options;
        options.group_dim = dim.group_dim;
        JXL_RETURN_IF_ERROR(jxl::ModularGenericDecompress(&reader, part, nullptr,
          jxl::ModularStreamId::ModularAC(group, 0).ID(dim), &options, true, &tree, &code, &contexts));
        JXL_RETURN_IF_ERROR(FinishSection(reader));
        JXL_ENSURE(!part.error && part.channel.size() == channels && part.nb_meta_channels == 0);
        for (size_t c = 0; c < channels; ++c) {
          JXL_ENSURE(part.channel[c].w == rect.xsize() && part.channel[c].h == rect.ysize());
          for (size_t y = 0; y < rect.ysize(); ++y) std::copy_n(part.channel[c].Row(y), rect.xsize(), image.channel[c].Row(rect.y0() + y) + rect.x0());
        }
      }
    }
    image.undo_transforms(global.wp_header);
    JXL_ENSURE(!image.error && image.nb_meta_channels == 0 && image.channel.size() == channels);
    Frame frame{uint32_t(dim.xsize), uint32_t(dim.ysize), uint32_t(channels), uint32_t(bits), metadata.m.bit_depth.floating_point_sample ? metadata.m.bit_depth.exponent_bits_per_sample : 0, {}};
    for (const auto& channel : image.channel) {
      JXL_ENSURE(channel.w == dim.xsize && channel.h == dim.ysize);
      for (size_t y = 0; y < dim.ysize; ++y) for (size_t x = 0; x < dim.xsize; ++x) frame.words.push_back(uint32_t(channel.Row(y)[x]));
    }
    frames->push_back(std::move(frame));
    last = header.is_last;
  }
  JXL_ENSURE(pos == raw.size());
  return true;
}

void Word(uint32_t value) {
  const uint8_t bytes[] = {uint8_t(value), uint8_t(value >> 8), uint8_t(value >> 16), uint8_t(value >> 24)};
  if (fwrite(bytes, 1, 4, stdout) != 4) std::abort();
}
} // namespace

int main(int argc, char** argv) {
  if (JPEGXL_MAJOR_VERSION != 0 || JPEGXL_MINOR_VERSION != 12 || JPEGXL_PATCH_VERSION != 0 || JxlDecoderVersion() != 12000) return 2;
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  if (!input) return 2;
  std::vector<uint8_t> file{std::istreambuf_iterator<char>(input), {}};
  if (file.size() > (1u << 26)) return 2;
  std::vector<uint8_t> raw;
  std::vector<Frame> frames;
  if (!Codestream(file, &raw) || !Decode(raw, &frames)) return 1;
  if (fwrite("JXLRAW12", 1, 8, stdout) != 8) return 2;
  Word(JxlDecoderVersion());
  Word(frames.size());
  for (const auto& frame : frames) {
    for (uint32_t field : {frame.width, frame.height, frame.channels, frame.bits, frame.exponent}) Word(field);
    for (uint32_t word : frame.words) Word(word);
  }
  return 0;
}
