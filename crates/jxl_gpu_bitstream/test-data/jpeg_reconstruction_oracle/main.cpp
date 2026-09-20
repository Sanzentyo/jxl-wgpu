#include <jxl/decode.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("input.jxl output.jpg max-output-bytes");
    std::ifstream source(argv[1], std::ios::binary);
    std::vector<uint8_t> input((std::istreambuf_iterator<char>(source)), {});
    if (!source.eof() && source.fail()) throw std::runtime_error("read input");
    const auto limit = std::stoull(argv[3]);
    if (limit == 0 || limit > 8 * 1024 * 1024) throw std::runtime_error("output limit");
    std::vector<uint8_t> output(limit);
    std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> decoder(JxlDecoderCreate(nullptr), JxlDecoderDestroy);
    if (!decoder) throw std::runtime_error("create decoder");
    if (JxlDecoderSubscribeEvents(decoder.get(), JXL_DEC_JPEG_RECONSTRUCTION | JXL_DEC_FULL_IMAGE) != JXL_DEC_SUCCESS) throw std::runtime_error("subscribe");
    if (JxlDecoderSetInput(decoder.get(), input.data(), input.size()) != JXL_DEC_SUCCESS) throw std::runtime_error("set input");
    JxlDecoderCloseInput(decoder.get());
    bool reconstructed = false;
    bool complete = false;
    for (;;) {
      const auto status = JxlDecoderProcessInput(decoder.get());
      if (status == JXL_DEC_JPEG_RECONSTRUCTION) {
        if (reconstructed) throw std::runtime_error("duplicate JPEG event");
        if (JxlDecoderSetJPEGBuffer(decoder.get(), output.data(), output.size()) != JXL_DEC_SUCCESS) throw std::runtime_error("set JPEG output");
        reconstructed = true;
      } else if (status == JXL_DEC_FULL_IMAGE) {
        if (!reconstructed || complete) throw std::runtime_error("invalid JPEG completion");
        complete = true;
      } else if (status == JXL_DEC_SUCCESS) {
        if (!complete) throw std::runtime_error("JPEG reconstruction unavailable");
        const auto unused = JxlDecoderReleaseJPEGBuffer(decoder.get());
        output.resize(output.size() - unused);
        std::ofstream destination(argv[2], std::ios::binary);
        destination.write(reinterpret_cast<const char*>(output.data()), output.size());
        if (!destination) throw std::runtime_error("write JPEG");
        std::cout << "libjxl " << JxlDecoderVersion() << "; reconstructed " << output.size() << " bytes without pixel fallback\n";
        return 0;
      } else {
        std::cerr << "JPEG-only decoder status " << static_cast<int>(status) << '\n';
        return 1;
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
