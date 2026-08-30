# Third-party notices

Parts of `src/acceleration.rs` are adapted from the `zune-jpegxl` 0.5.2 fast-lossless
encoder. The original is copyright the zune-image developers and is used under the MIT
license reproduced in `LICENSES/zune-jpegxl-MIT.txt`.

The production image-header grammar is provided by `jxl-image` 0.13.0 and its public
`jxl-bitstream` 1.0.0 / `jxl-oxide-common` 1.0.0 interfaces. Entropy-coded TOC permutation metadata
uses `jxl-coding` 1.0.1. Those crates are part of jxl-oxide and are licensed under
`MIT OR Apache-2.0`; their source distributions contain the corresponding license texts. No
jxl-oxide frame, Modular, VarDCT, or pixel decoder is linked into this crate.

`test-data/basic.jxl.hex`, `oddsize_ups.jxl.hex`, `green_queen_vardct_e3.jxl.hex`, and
`animation_spline.jxl.hex`, and `has_permutation.jxl.hex` are byte-for-byte hexadecimal copies of
the corresponding JPEG XL fixtures in `libjxl/jxl-rs` commit
`f37283edbac13f47e03e79db393438a4a2b82e07` (`jxl` 0.6.0).
They are used under that project's BSD-3-Clause license and are stored package-locally so the
published crate's tests do not depend on workspace-root files.
