# Third-party notices

The production workspace does not vendor a CPU JPEG XL codec. The following implementation
reference was adapted into focused GPU-codec control-plane code:

- `zune-jpegxl` 0.5.2 fast-lossless prefix-code and JPEG XL header construction. Adapted portions
  are identified in `crates/jxl_gpu_bitstream/src/acceleration.rs`,
  `crates/jxl_wgpu_encode/src/prefix.rs`, and
  `crates/jxl_wgpu_encode/src/lossless_gray8.rs`. The original is copyright the zune-image
  developers and used under the MIT license reproduced in
  `LICENSES/zune-jpegxl-MIT.txt`. It is not linked as a production dependency and is not a CPU
  codec or pixel fallback in this workspace.

Reference implementations and command-line tools used only for conformance testing retain their
own licenses and are not redistributed by this repository.
