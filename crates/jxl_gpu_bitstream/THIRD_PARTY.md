# Third-party notices

Parts of `src/acceleration.rs` are adapted from the `zune-jpegxl` 0.5.2 fast-lossless
encoder. The original is copyright the zune-image developers and is used under the MIT
license reproduced in `LICENSES/zune-jpegxl-MIT.txt`.

The locally owned image/color-header grammar reuses primitive bundles from `jxl-image` 0.13.0 and its public
`jxl-bitstream` 1.0.0 / `jxl-oxide-common` 1.0.0 interfaces. Entropy-coded TOC permutation metadata
uses `jxl-coding` 1.0.1, and bounded embedded ICC reconstruction uses `jxl-color` 0.11.0. Those
crates are part of jxl-oxide and are licensed under
`MIT OR Apache-2.0`; their source distributions contain the corresponding license texts. No
jxl-oxide frame, Modular, VarDCT, or pixel decoder is linked into this crate.

`src/icc_profile.rs` and its child modules adapt the ICC metadata serialization in libjxl
0.12.0, commit `a7a9c787341cf703dede03c2009fa460cae5e5df`: `lib/jxl/cms/jxl_cms_internal.h`,
`color_encoding_cms.h`, `opsin_params.h`, `tone_mapping.h`, `transfer_functions.h` and
`lib/jxl/base/matrix_ops.h`. Copyright the JPEG XL Project Authors; BSD-3-Clause terms are
reproduced in this crate's `LICENSE`. This bounded metadata writer processes no image samples.
The `md-5` 0.11.0 dependency computes the ICC profile ID and is licensed under
`MIT OR Apache-2.0`, with license texts in its source distribution.

`test-data/basic.jxl.hex`, `oddsize_ups.jxl.hex`, `green_queen_vardct_e3.jxl.hex`, and
`animation_spline.jxl.hex`, `has_permutation.jxl.hex`, and `with_icc.jxl.hex` are byte-for-byte
hexadecimal copies of the corresponding JPEG XL fixtures in `libjxl/jxl-rs` commit
`f37283edbac13f47e03e79db393438a4a2b82e07` (`jxl` 0.6.0).
They are used under that project's BSD-3-Clause license and are stored package-locally so the
published crate's tests do not depend on workspace-root files.
