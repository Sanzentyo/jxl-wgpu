// Development-only libjpeg-turbo 3.2.0 oracle. Its src/jdcoefct.c allocates and
// consumes whole sampling-aligned MCUs, including blocks outside width/height_in_blocks.
// These original JPEG integers are independent of every production JXL/GPU operation.
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
extern "C" {
#include <jpeglib.h>
}
static void word(uint32_t value) {
  for (unsigned i = 0; i < 4; ++i) {
    if (std::fputc((value >> (8 * i)) & 255, stdout) == EOF) std::exit(3);
  }
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  FILE* input = std::fopen(argv[1], "rb");
  if (!input) return 3;
  jpeg_decompress_struct jpeg{};
  jpeg_error_mgr error{};
  jpeg.err = jpeg_std_error(&error);
  jpeg_create_decompress(&jpeg);
  jpeg_stdio_src(&jpeg, input);
  if (jpeg_read_header(&jpeg, TRUE) != JPEG_HEADER_OK) return 4;
  if (uint64_t(jpeg.image_width) * jpeg.image_height > (1u << 24) || jpeg.data_precision != 8 || jpeg.num_components > 3) return 5;
  jvirt_barray_ptr* arrays = jpeg_read_coefficients(&jpeg);
  if (!arrays) return 6;
  word(0x314f434a); word(jpeg.image_width); word(jpeg.image_height); word(jpeg.num_components);
  for (int c = 0; c < jpeg.num_components; ++c) {
    const auto& info = jpeg.comp_info[c];
    const JDIMENSION width = (info.width_in_blocks + info.h_samp_factor - 1) / info.h_samp_factor * info.h_samp_factor;
    const JDIMENSION height = (info.height_in_blocks + info.v_samp_factor - 1) / info.v_samp_factor * info.v_samp_factor;
    word(info.component_id); word(info.h_samp_factor); word(info.v_samp_factor);
    word(width); word(height); word(info.width_in_blocks); word(info.height_in_blocks);
    const auto* quant = jpeg.quant_tbl_ptrs[info.quant_tbl_no];
    if (!quant) return 7;
    for (unsigned k = 0; k < DCTSIZE2; ++k) word(quant->quantval[k]);
    for (JDIMENSION y = 0; y < height; ++y) {
      JBLOCKARRAY row = (*jpeg.mem->access_virt_barray)(reinterpret_cast<j_common_ptr>(&jpeg), arrays[c], y, 1, FALSE);
      for (JDIMENSION x = 0; x < width; ++x) {
        for (unsigned k = 0; k < DCTSIZE2; ++k) word(static_cast<uint32_t>(static_cast<int32_t>(row[0][x][k])));
      }
    }
  }
  if (!jpeg_finish_decompress(&jpeg)) return 8;
  jpeg_destroy_decompress(&jpeg);
  if (std::fclose(input) != 0 || std::fflush(stdout) != 0) return 3;
  return 0;
}
