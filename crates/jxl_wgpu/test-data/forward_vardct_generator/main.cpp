// Offline, pinned native forward-transform oracle. No production code links it.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/enc_transforms.h"
#include "lib/jxl/quant_weights.h"

namespace {
void Word(std::ostream &out, uint32_t word) {
  for (int i = 0; i < 4; ++i)
    out.put(static_cast<char>(word >> (8 * i)));
}

float Sample(size_t width, size_t height, uint32_t test, size_t channel,
             size_t x, size_t y) {
  if (test != 0) {
    const size_t position = test - 1;
    const size_t target = channel == 0   ? position
                          : channel == 1 ? (position + 17) % 64
                                         : 63 - position;
    const float amplitude = channel == 0 ? 1.0f : channel == 1 ? -0.5f : 0.25f;
    return y * width + x == target ? amplitude : 0.0f;
  }
  if (channel == 1)
    return 0.375f;
  if (channel == 2)
    return x == width - 1 && y == height / 2 ? -0.75f : 0.0f;
  const int code = (x * 37 + y * 101 + x * y * 3) % 509;
  return (code - 254) / 256.0f;
}

void Record(std::ostream &out, uint32_t raw, uint32_t test) {
  const auto strategy =
      jxl::AcStrategy::FromRawStrategy(static_cast<uint8_t>(raw));
  const size_t lw = strategy.covered_blocks_x();
  const size_t lh = strategy.covered_blocks_y();
  const size_t width = lw * 8, height = lh * 8, area = width * height;
  Word(out, raw);
  Word(out, test);
  Word(out, width);
  Word(out, height);
  // Fixed alignment and stride meet every native transform's SIMD contract.
  alignas(64) float pixels[jxl::AcStrategy::kMaxCoeffArea];
  alignas(64) float coefficients[jxl::AcStrategy::kMaxCoeffArea];
  alignas(64) float scratch[2 * jxl::AcStrategy::kMaxCoeffArea];
  alignas(64) float
      lf[jxl::AcStrategy::kMaxCoeffBlocks * jxl::AcStrategy::kMaxCoeffBlocks];
  std::vector<float> result;
  std::vector<float> low_frequency;
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < height; ++y)
      for (size_t x = 0; x < width; ++x)
        pixels[y * width + x] = Sample(width, height, test, channel, x, y);
    const auto kind = static_cast<jxl::AcStrategyType>(raw);
    jxl::TransformFromPixels(kind, pixels, width, coefficients, scratch);
    jxl::DCFromLowestFrequencies(kind, coefficients, lf, lw, scratch);
    result.insert(result.end(), coefficients, coefficients + area);
    low_frequency.insert(low_frequency.end(), lf, lf + lw * lh);
  }
  result.insert(result.end(), low_frequency.begin(), low_frequency.end());
  for (float value : result) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    Word(out, bits);
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3 || std::filesystem::exists(argv[1]) ||
      std::filesystem::exists(argv[2]))
    return 2;
  std::ofstream out(argv[1], std::ios::binary);
  out.write("JXLFWD01", 8);
  Word(out, 667); // 27 patterns plus all 64 impulses for ten 8x8 strategies.
  for (uint32_t raw = 0; raw < 27; ++raw) {
    Record(out, raw, 0);
    const auto strategy =
        jxl::AcStrategy::FromRawStrategy(static_cast<uint8_t>(raw));
    if (strategy.covered_blocks_x() * strategy.covered_blocks_y() == 1)
      for (uint32_t test = 1; test <= 64; ++test)
        Record(out, raw, test);
  }
  std::printf("667 native forward VarDCT records\n");
  JxlMemoryManager memory = {
      nullptr, [](void *, size_t size) { return std::malloc(size); },
      [](void *, void *address) { std::free(address); }};
  jxl::DequantMatrices matrices;
  if (!matrices.EnsureComputed(&memory, (1u << 27) - 1))
    return 1;
  std::ofstream metadata(argv[2], std::ios::binary);
  metadata.write("JXLQNT01", 8);
  Word(metadata, 27);
  for (uint32_t raw = 0; raw < 27; ++raw) {
    const auto strategy =
        jxl::AcStrategy::FromRawStrategy(static_cast<uint8_t>(raw));
    const size_t area =
        64 * strategy.covered_blocks_x() * strategy.covered_blocks_y();
    Word(metadata, raw);
    Word(metadata, area);
    std::vector<jxl::coeff_order_t> order(area);
    strategy.ComputeNaturalCoeffOrder(order.data());
    for (auto position : order)
      Word(metadata, position);
    for (size_t channel = 0; channel < 3; ++channel) {
      const float *matrix =
          matrices.Matrix(static_cast<jxl::AcStrategyType>(raw), channel);
      for (size_t position = 0; position < area; ++position) {
        uint32_t bits;
        std::memcpy(&bits, matrix + position, sizeof(bits));
        Word(metadata, bits);
      }
    }
  }
  std::printf(
      "27 native dequantization matrices and natural coefficient orders\n");
  return out && metadata ? 0 : 1;
}
