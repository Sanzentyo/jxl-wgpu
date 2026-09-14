# Third-party notices

## JPEG XL forward transforms

`src/forward_vardct/basis.rs` derives bounded strategy constants from the forward-transform
equations in libjxl 0.12.0 `enc_transforms-inl.h` at commit
`a7a9c787341cf703dede03c2009fa460cae5e5df`. AFV reuses the inverse basis described below.
All source-dependent arithmetic executes in `shaders/forward_vardct.wgsl`. The independent
fixture generator calls the pinned native implementation offline.

Copyright (c) the JPEG XL Project Authors. The BSD-3-Clause terms are reproduced in this
crate's `LICENSE` file.

## JPEG XL frame upsampling

`src/resident_upsample.rs` expands compact image-header weights using the normative symmetry
contract implemented by libjxl's BSD-3-Clause
[`stage_upsampling.cc`](https://github.com/libjxl/libjxl/blob/main/lib/jxl/render_pipeline/stage_upsampling.cc).
It reuses the existing `shaders/upsample.wgsl` filter for GPU sample processing. Copyright (c) the
JPEG XL Project Authors. The BSD-3-Clause terms are reproduced in this crate's `LICENSE` file.

## `jxl_transforms` 0.6.0

The normative 4x4 AFV inverse basis in `src/vardct_general.rs` and the transform ordering,
normalization, and scalar test oracles are derived from the BSD-3-Clause
[`jxl_transforms`](https://github.com/libjxl/jxl-rs/tree/main/crates/jxl_transforms)
implementation. The crate is used only as a development dependency; production VarDCT execution
remains in WGSL.

Copyright (c) the JPEG XL Project Authors. All rights reserved.

The BSD-3-Clause terms are reproduced in this crate's `LICENSE` file.
