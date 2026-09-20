# Third-party notices

The production workspace does not vendor a CPU JPEG XL codec. The following implementation
references were adapted into focused GPU-codec code and bounded metadata expansion:

- libjxl 0.12.0 default quantization constants, natural coefficient orders and forward-transform
  equations at commit `a7a9c787341cf703dede03c2009fa460cae5e5df`. The affected modules and
  copyright notices are recorded in `crates/jxl_gpu_protocol/THIRD_PARTY.md` and
  `crates/jxl_wgpu/THIRD_PARTY.md`; BSD-3-Clause terms are included in each crate's `LICENSE`.
  The same revision's bounded ICC metadata serialization is adapted in
  `crates/jxl_gpu_bitstream/src/icc_profile.rs` and its child modules, with source attribution
  and BSD-3-Clause terms in that crate's `THIRD_PARTY.md` and `LICENSE`.

- `zune-jpegxl` 0.5.2 fast-lossless prefix-code and JPEG XL header construction. Adapted portions
  are identified in `crates/jxl_gpu_bitstream/src/acceleration.rs`,
  `crates/jxl_wgpu_encode/src/prefix.rs`, and
  `crates/jxl_wgpu_encode/src/lossless_gray8.rs`. The original is copyright the zune-image
  developers and used under the MIT license reproduced in
  `LICENSES/zune-jpegxl-MIT.txt`. It is not linked as a production dependency and is not a CPU
  codec or pixel fallback in this workspace.

Reference implementations and command-line tools used only for conformance testing retain their
own licenses and are not redistributed by this repository.
