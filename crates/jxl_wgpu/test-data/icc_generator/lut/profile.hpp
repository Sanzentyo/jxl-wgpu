#pragma once
#include "curve.hpp"
#include "stages.hpp"

namespace lut {
enum class Format { Eight, Sixteen, AB };
struct Recipe {
  std::string name;
  Format format;
  bool lab;
  unsigned channels;
  unsigned version = 4;
  bool matrix = true, clut = true, shared = false;
};
struct Pipeline {
  Bytes bytes;
  std::vector<Stage> stages;
};

inline Pipeline Tables(const Recipe &recipe, unsigned intent, bool reverse) {
  const unsigned p = reverse ? 3 : recipe.channels,
                 q = reverse ? recipe.channels : 3;
  const unsigned precision = recipe.format == Format::Eight ? 1 : 2;
  const unsigned n = precision == 1 ? 256 : 17, m = precision == 1 ? 256 : 33;
  const unsigned points = p > 5 ? 2 : 3;
  std::vector<Curve> input, output;
  for (unsigned c = 0; c < p; ++c)
    input.push_back(Table(n, precision, [=](double x) {
      return std::pow(x, .75 + (intent + c) / 8.0);
    }));
  for (unsigned c = 0; c < q; ++c)
    output.push_back(Table(m, precision, [=](double x) {
      return .0625 + (.875 - intent / 64.0) * std::pow(x, 1.1 + c / 16.0);
    }));
  const auto matrix =
      Matrix(reverse && !recipe.lab
                 ? std::array<double, 9>{1.125, -.0625, .03125, -.03125, .875,
                                         .0625, .03125, .125, 1.0625}
                 : std::array<double, 9>{1, 0, 0, 0, 1, 0, 0, 0, 1},
             {0, 0, 0}, true, true);
  const auto clut = Clut(std::vector<unsigned>(p, points), q, precision,
                         reverse && recipe.lab);
  auto bytes = Header(precision == 1 ? "mft1" : "mft2");
  for (unsigned v : {p, q, points, 0u})
    bytes.push_back(static_cast<uint8_t>(v));
  bytes.insert(bytes.end(), matrix.bytes.begin(), matrix.bytes.begin() + 36);
  if (precision == 2) {
    U16(bytes, static_cast<uint16_t>(n));
    U16(bytes, static_cast<uint16_t>(m));
  }
  auto append_tables = [&](const std::vector<Curve> &curves) {
    for (const auto &curve : curves)
      for (auto value : curve.table) {
        if (precision == 1)
          bytes.push_back(static_cast<uint8_t>(value / 257));
        else
          U16(bytes, value);
      }
  };
  append_tables(input);
  bytes.insert(bytes.end(), clut.bytes.begin() + 20, clut.bytes.end());
  append_tables(output);
  std::vector<Stage> stages;
  if (reverse && !recipe.lab)
    stages.push_back(matrix);
  stages.push_back(Curves(input));
  stages.push_back(clut);
  stages.push_back(Curves(output));
  return {bytes, stages};
}

inline Pipeline AB(const Recipe &recipe, unsigned intent, bool reverse) {
  const unsigned p = reverse ? 3 : recipe.channels,
                 q = reverse ? recipe.channels : 3;
  std::vector<Curve> b, m, a;
  for (unsigned c = 0; c < 3; ++c) {
    b.push_back(Shape(1 + intent + c));
    m.push_back(Shape(recipe.shared ? 1 + intent + c : 3 + intent + c));
  }
  for (unsigned c = 0; c < recipe.channels; ++c)
    a.push_back(Shape(recipe.shared
                          ? 1 + intent + c + (recipe.channels == 1 ? 2 : 0)
                          : 5 + intent + c));
  const auto bc = Curves(b), mc = Curves(m), ac = Curves(a);
  // Keep the synthetic RGB primaries' Y nonzero so libjxl can represent their
  // chromaticities. Negative X and clipped intermediate curves remain
  // exercised.
  const auto matrix =
      Matrix({1.125, -.125, .0625, .0625, .75, .125, -.0625, .125, 1.0625},
             {-.0625, .25, .0625}, true, true);
  std::vector<unsigned> grid(p);
  for (unsigned i = 0; i < p; ++i)
    grid[i] = p > 5 ? 2 : 2 + i % 2;
  const auto clut = Clut(grid, q, 1 + intent % 2, reverse && recipe.lab);
  auto bytes = Header(reverse ? "mBA " : "mAB ");
  bytes.resize(32);
  bytes[8] = static_cast<uint8_t>(p);
  bytes[9] = static_cast<uint8_t>(q);
  std::vector<std::pair<unsigned, Bytes>> pieces{{12, bc.bytes}};
  if (recipe.matrix) {
    pieces.push_back({16, matrix.bytes});
    if (!recipe.shared)
      pieces.push_back({20, mc.bytes});
  }
  if (recipe.clut) {
    pieces.push_back({24, clut.bytes});
    if (!recipe.shared)
      pieces.push_back({28, ac.bytes});
  }
  // The byte order is intentionally unrelated to the stage order, and shared
  // curve groups can start at an individual curve inside the stored B group.
  unsigned b_offset = 0;
  for (auto piece = pieces.rbegin(); piece != pieces.rend(); ++piece) {
    const auto offset = static_cast<unsigned>(bytes.size());
    Put(bytes, piece->first, offset);
    if (piece->first == 12)
      b_offset = offset;
    bytes.insert(bytes.end(), piece->second.begin(), piece->second.end());
    Pad(bytes);
  }
  if (recipe.shared) {
    if (recipe.matrix)
      Put(bytes, 20, b_offset);
    if (recipe.clut) {
      size_t skip = 0;
      if (recipe.channels == 1)
        for (unsigned c = 0; c < 2; ++c)
          skip += (b[c].bytes.size() == 8 ? 12 : b[c].bytes.size());
      Put(bytes, 28, b_offset + static_cast<unsigned>(skip));
    }
  }
  std::vector<Stage> stages;
  if (recipe.clut) {
    stages.push_back(ac);
    stages.push_back(clut);
  }
  if (recipe.matrix) {
    stages.push_back(mc);
    stages.push_back(matrix);
  }
  stages.push_back(bc);
  if (reverse)
    std::reverse(stages.begin(), stages.end());
  return {bytes, stages};
}

inline std::string DeviceSpace(unsigned channels) {
  if (channels == 1)
    return "GRAY";
  if (channels == 3)
    return "RGB ";
  if (channels == 4)
    return "CMYK";
  return std::string(1,
                     static_cast<char>(channels < 10 ? '0' + channels
                                                     : 'A' + channels - 10)) +
         "CLR";
}
struct Profile {
  Recipe recipe;
  std::array<Pipeline, 6> pipelines;
  Bytes bytes;
};
inline Profile Build(const Recipe &recipe) {
  Profile result{recipe, {}, {}};
  std::vector<std::pair<std::string, Bytes>> tags;
  auto white = Header("XYZ ");
  U32(white, 0xf6d6);
  U32(white, 65536);
  U32(white, 0xd32d);
  tags.push_back({"wtpt", white});
  for (unsigned intent = 0; intent < 3; ++intent)
    for (bool reverse : {false, true}) {
      auto pipeline = recipe.format == Format::AB
                          ? AB(recipe, intent, reverse)
                          : Tables(recipe, intent, reverse);
      tags.push_back(
          {std::string(reverse ? "B2A" : "A2B") + std::to_string(intent),
           pipeline.bytes});
      auto pcs = Pcs(recipe.lab, recipe.format == Format::Sixteen, reverse);
      if (reverse)
        pipeline.stages.insert(pipeline.stages.begin(), pcs.begin(), pcs.end());
      else
        pipeline.stages.insert(pipeline.stages.end(), pcs.begin(), pcs.end());
      result.pipelines[intent * 2 + reverse] = std::move(pipeline);
    }
  auto &bytes = result.bytes;
  bytes.resize(132 + tags.size() * 12);
  Put(bytes, 8, recipe.version == 4 ? 0x04400000 : 0x02400000);
  std::copy_n("mntr", 4, bytes.begin() + 12);
  const auto space = DeviceSpace(recipe.channels);
  std::copy_n(space.begin(), 4, bytes.begin() + 16);
  std::copy_n(recipe.lab ? "Lab " : "XYZ ", 4, bytes.begin() + 20);
  std::copy_n("acsp", 4, bytes.begin() + 36);
  Put(bytes, 64, 1);
  Put(bytes, 68, 0xf6d6);
  Put(bytes, 72, 65536);
  Put(bytes, 76, 0xd32d);
  Put(bytes, 128, static_cast<uint32_t>(tags.size()));
  for (size_t i = 0; i < tags.size(); ++i) {
    const auto &[name, data] = tags[i];
    std::copy_n(name.begin(), 4, bytes.begin() + 132 + i * 12);
    Put(bytes, 136 + i * 12, static_cast<uint32_t>(bytes.size()));
    Put(bytes, 140 + i * 12, static_cast<uint32_t>(data.size()));
    bytes.insert(bytes.end(), data.begin(), data.end());
    Pad(bytes);
  }
  Put(bytes, 0, static_cast<uint32_t>(bytes.size()));
  return result;
}
} // namespace lut
