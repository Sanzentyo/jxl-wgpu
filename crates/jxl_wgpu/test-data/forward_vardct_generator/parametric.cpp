// Offline native oracle for the wire-unit scaling of matrix modes 1 and 2.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/quant_weights.h"

namespace {
void Word(std::ostream& out, uint32_t word) {
  for (int i = 0; i < 4; ++i) out.put(static_cast<char>(word >> (8 * i)));
}
void Bits(std::vector<uint8_t>& bytes, size_t& cursor, uint32_t value, size_t count) {
  for (size_t i = 0; i < count; ++i, ++cursor) {
    if (cursor / 8 == bytes.size()) bytes.push_back(0);
    bytes[cursor / 8] |= ((value >> i) & 1) << (cursor % 8);
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || std::filesystem::exists(argv[1])) return 1;
  std::ofstream out(argv[1], std::ios::binary);
  out.write("JXLPQM01", 8);
  Word(out, 4);
  JxlMemoryManager memory = {
      nullptr, [](void*, size_t size) { return std::malloc(size); },
      [](void*, void* address) { std::free(address); }};
  for (uint32_t mode : {1u, 2u}) {
    for (uint32_t variant : {0u, 1u}) {
      std::vector<uint8_t> bytes;
      size_t cursor = 0;
      Bits(bytes, cursor, 0, 1);  // custom matrices
      Bits(bytes, cursor, mode, 3);  // DCT8's matrix family
      const size_t count = mode == 1 ? 3 : 6;
      Word(out, mode);
      Word(out, variant);
      Word(out, count);
      for (size_t c = 0; c < 3; ++c) {
        for (size_t i = 0; i < count; ++i) {
          const uint32_t parameter = variant == 0 ? 0x3c00 :
              0x4000 + static_cast<uint32_t>((c + i) % 3) * 0x400;
          Word(out, parameter);
          Bits(bytes, cursor, parameter, 16);
        }
      }
      for (size_t i = 1; i < 17; ++i) Bits(bytes, cursor, 0, 3);
      jxl::BitReader reader(bytes);
      jxl::DequantMatrices matrices;
      if (!matrices.Decode(&memory, &reader, nullptr) || !reader.Close() ||
          !matrices.EnsureComputed(&memory, 1)) return 1;
      for (size_t c = 0; c < 3; ++c) {
        const float* values = matrices.Matrix(jxl::AcStrategyType::DCT, c);
        for (size_t i = 0; i < 64; ++i) {
          uint32_t bits;
          std::memcpy(&bits, values + i, sizeof(bits));
          Word(out, bits);
        }
      }
    }
  }
  return out ? 0 : 1;
}
