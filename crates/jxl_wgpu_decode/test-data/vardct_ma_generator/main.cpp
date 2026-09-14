// Independent offline libjxl fixtures; never linked into the production codec.
#include <jxl/cms.h>
#include <jxl/memory_manager.h>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/image_bundle.h"
#include "references.h"

namespace {
jxl::Status Encode(JxlMemoryManager* memory, const std::filesystem::path& output,
                   uint32_t width, uint32_t height, int property, bool weighted) {
  jxl::CodecMetadata metadata;
  JXL_RETURN_IF_ERROR(metadata.size.Set(width, height));
  metadata.m.color_encoding = jxl::ColorEncoding::SRGB();
  metadata.m.xyb_encoded = true;
  metadata.m.bit_depth.bits_per_sample = 8;
  metadata.m.bit_depth.floating_point_sample = false;
  JXL_ASSIGN_OR_RETURN(auto image, jxl::Image3F::Create(memory, width, height));
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        image.PlaneRow(c, y)[x] =
            static_cast<float>((x * (3 + c * 7) + y * (11 + c * 13) +
                                (x / 8) * (y / 8) * 17 + c * 71) % 256) / 255;
      }
    }
  }
  jxl::ImageBundle bundle(memory, &metadata.m);
  JXL_RETURN_IF_ERROR(bundle.SetFromImage(std::move(image), metadata.m.color_encoding));
  jxl::CompressParams params;
  params.SetCms(*JxlGetDefaultCms());
  params.speed_tier = jxl::SpeedTier::kSquirrel;
  params.butteraugli_distance = 2;
  params.patches = jxl::Override::kOff;
  params.noise = jxl::Override::kOff;
  params.gaborish = jxl::Override::kOff;
  params.epf = 0;
  params.palette_colors = 0;
  params.channel_colors_percent = 0;
  params.channel_colors_pre_transform_percent = 0;
  params.custom_fixed_tree = {
      jxl::PropertyDecisionNode::Split(property, 0, 1, 2),
      jxl::PropertyDecisionNode::Leaf(weighted ? jxl::Predictor::Weighted : jxl::Predictor::Gradient),
      jxl::PropertyDecisionNode::Leaf(jxl::Predictor::Zero, -3)};
  jxl::BitWriter writer(memory);
  JXL_RETURN_IF_ERROR(jxl::WriteCodestreamHeaders(&metadata, &writer, nullptr));
  writer.ZeroPadToByte();
  JXL_RETURN_IF_ERROR(jxl::EncodeFrame(memory, params, jxl::FrameInfo(), &metadata,
                                      bundle, *JxlGetDefaultCms(), nullptr, &writer, nullptr));
  writer.ZeroPadToByte();
  std::ofstream out(output);
  size_t index = 0;
  for (uint8_t byte : writer.GetSpan()) {
    out << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(byte);
    if (++index % 40 == 0) out << '\n';
  }
  out << '\n';
  return bool(out);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || std::filesystem::exists(argv[1])) {
    std::cerr << "Usage: generate_vardct_ma NEW_OUTPUT_DIRECTORY\n";
    return 1;
  }
  const std::filesystem::path output(argv[1]);
  std::filesystem::create_directories(output);
  JxlMemoryManager memory = {nullptr,
      [](void*, size_t size) -> void* { return std::malloc(size); },
      [](void*, void* ptr) { std::free(ptr); }};
  if (!WriteReferences(&memory, output)) return 1;
  std::ofstream manifest(output / "manifest.txt");
  for (const auto& extent : std::array<std::array<uint32_t, 2>, 3>{{
           {8, 8}, {129, 73}, {272, 32}}}) {
    for (int property = 16; property < 28; ++property) {
      for (bool weighted : {false, true}) {
        const std::string name = std::to_string(extent[0]) + "x" + std::to_string(extent[1]) +
            "_p" + std::to_string(property) + (weighted ? "_weighted" : "_gradient");
        if (!Encode(&memory, output / (name + ".jxl.hex"), extent[0], extent[1], property, weighted)) {
          return 1;
        }
        manifest << name << ' ' << extent[0] << ' ' << extent[1] << ' ' << property
                 << ' ' << weighted << '\n';
      }
    }
  }
  return manifest ? 0 : 1;
}
