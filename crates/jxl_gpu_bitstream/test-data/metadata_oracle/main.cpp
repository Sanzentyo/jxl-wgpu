// Raw libjxl box API oracle. Unlike djxl image export, this never rewrites Exif orientation.
#include <jxl/decode.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::ifstream file(argv[1], std::ios::binary);
  const std::vector<uint8_t> input{std::istreambuf_iterator<char>(file), {}};
  if (input.empty()) return 3;
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> decoder(
      JxlDecoderCreate(nullptr), &JxlDecoderDestroy);
  if (!decoder || JxlDecoderSubscribeEvents(decoder.get(), JXL_DEC_BOX) != JXL_DEC_SUCCESS ||
      JxlDecoderSetDecompressBoxes(decoder.get(), JXL_TRUE) != JXL_DEC_SUCCESS ||
      JxlDecoderSetInput(decoder.get(), input.data(), input.size()) != JXL_DEC_SUCCESS) return 4;
  JxlDecoderCloseInput(decoder.get());
  std::array<uint8_t, 4096> buffer{};
  std::vector<uint8_t> payload;
  std::string name;
  auto flush = [&]() {
    const size_t remaining = JxlDecoderReleaseBoxBuffer(decoder.get());
    if (remaining > buffer.size()) return false;
    payload.insert(payload.end(), buffer.begin(), buffer.end() - remaining);
    return payload.size() <= (1 << 20);
  };
  for (;;) {
    const auto status = JxlDecoderProcessInput(decoder.get());
    if (status == JXL_DEC_BOX_NEED_MORE_OUTPUT) {
      if (name.empty() || !flush() || JxlDecoderSetBoxBuffer(decoder.get(), buffer.data(), buffer.size()) != JXL_DEC_SUCCESS) return 5;
      continue;
    }
    if (status != JXL_DEC_BOX && status != JXL_DEC_SUCCESS) {
      std::cerr << "unexpected decoder status " << status << '\n';
      return 6;
    }
    if (!name.empty()) {
      if (!flush()) return 7;
      std::ofstream output(std::string(argv[2]) + "/" + name + ".bin", std::ios::binary);
      output.write(reinterpret_cast<const char*>(payload.data()), payload.size());
      if (!output) return 8;
      name.clear();
      payload.clear();
    }
    if (status == JXL_DEC_SUCCESS) return 0;
    JxlBoxType type;
    if (JxlDecoderGetBoxType(decoder.get(), type, JXL_TRUE) != JXL_DEC_SUCCESS) return 9;
    const std::string kind(type, 4);
    if (kind == "Exif") name = "exif";
    if (kind == "xml ") name = "xmp";
    if (kind == "jumb") name = "jumbf";
    if (!name.empty() && JxlDecoderSetBoxBuffer(decoder.get(), buffer.data(), buffer.size()) != JXL_DEC_SUCCESS) return 10;
  }
}
