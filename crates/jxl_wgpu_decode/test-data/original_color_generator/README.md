# Original SDR color corpus

`main.cpp` uses the public encoder and decoder APIs of libjxl **0.12.0** (version 12000).
The source audited for generation is tag v0.12.0, commit
`a7a9c787341cf703dede03c2009fa460cae5e5df`. CPU codecs are offline test dependencies;
production reconstruction, color conversion and blending stay on the GPU.

```sh
c++ -std=c++17 -Wall -Wextra -Werror main.cpp $(pkg-config --cflags --libs libjxl libjxl_cms) -o generate-original-color
./generate-original-color ../original_color
```

Then, from the workspace root:

```sh
cargo run -p jxl_wgpu_decode --example regenerate_original_color
cargo test -p jxl_wgpu_decode --test original_color -- --test-threads=1
```

The Rust completion tool also requires the native extra-channel oracle documented in
`tools/jxl_test_support/src/oracles/extra_channels.rs`. Case classification, color declarations,
source precision and YCbCr recipes are explicit in
`tools/jxl_test_support/src/fixtures/original_color.rs`; filenames are generated labels.

| Dimension | Cases |
| --- | --- |
| Original Modular / VarDCT RGB, and Modular / VarDCT XYB | Each mode: D65 BT.709, BT.2020, Display-P3, gray × Linear, sRGB, BT.709 × still/sequence |
| Modular / VarDCT YCbCr | Each mode: three RGB primaries × three transfers × still/sequence |
| F32 sources | Four RGB/XYB modes × BT.2020 Linear or P3 BT.709 × still/sequence |
| Total | 148 streams: 74 stills and 74 sequences; 332 files including 36 RGB component sources and 148 original RGBA F32 references |

Images are 37×19, with independent 10-bit alpha. Color is 12-bit (8-bit for YCbCr component
sources) or F32. Each six-frame sequence presents four images and uses all five blend modes,
independent alpha Replace, cropped and oversized layers, a hidden layer and overwritten reference
slots 1 and 2. It does not claim coverage of all four reference slots. Alpha is preserved,
orientation is kept, and spot rendering is disabled in the native reference decoder.

The public libjxl encoder overrides its `COLOR_TRANSFORM` option from the image metadata
(`lib/jxl/encode.cc`, lines 924–930 at the pinned revision). Its YCbCr requests therefore emit
explicit `.rgb-source.jxl.hex` files. Rust changes only each frame's `do_ycbcr` flag and inserts
three full-resolution sampling selectors; it preserves the 8-bit component entropy, image
metadata, transforms and all other frame fields. It reparses every resulting stream, asserts
actual YCbCr metadata, and independently decodes it with native libjxl before saving a reference.
No source precision is transplanted and no GPU result is used as a reference.

## Precision and independent references

The original RGBA test checks normalized absolute color error `abs(gpu-reference)/(1+abs(reference))`:
`1e-5` for original Modular and `1/1024` for XYB/VarDCT. Alpha has a separate `2e-6` bound.
Whole input, 256-byte GPU windows with 43-byte async fragments, every held progressive image,
final-only equality, final frame count and zero remaining reservations are checked.

An independent f64 oracle derives primary matrices from CIE xy coordinates and the shared D65
white `(0.3127, 0.3290)`. It propagates the fixed reconstruction interval through EOTFs and signed
primary matrices, including BT.709's inverse breakpoint discontinuity. Conversion allows a
separate `5e-6*(1+abs(reference))` packing error and one integer code where quantized. It checks
linear BT.709 F32, sRGB RGBA8, original RGBA12, original numeric color and independent numeric alpha.

The numeric regression test additionally re-encodes only the image color metadata of three existing
exact-word integer fixtures across all nine RGB profiles. The resulting 27 variants retain
byte-identical physical frames. Native RGB and each scalar selection must preserve 17/31-bit color
and independent 5/24/31-bit alpha exactly, both for whole input and bounded fragmented input. This
guards against metadata-only color conversion rounding native words through F32.

jxl-oxide 0.12.6 independently decodes all 74 original still streams. Original Gray is requested
as Gray and replicated for RGBA comparison, avoiding a Gray-to-RGB CMS round trip. For XYB,
jxl-color 0.11.0 `src/convert.rs` inserts `Clip` or `GamutMap` before primary/Gray conversion.
The test therefore requests unbounded linear BT.709 from jxl-oxide, then uses the independent f64
matrix, gray luminance projection and original transfer. It never derives expected pixels from
the GPU or silently clips the native reference.
Its jxl-frame 0.13.3 dependency uses the color blend mode when deciding whether an extra channel
has a source selector (`src/header.rs`, `ec_blending_info` context and `BlendingInfo::source`).
The full-canvas Mul/Blend frames with alpha Replace omit that selector, as required by the extra's
own mode; jxl-oxide loses alignment and reports `Frame(Bitstream(NonZeroPadding))` for the first
sequence. Native libjxl uses each extra's own mode in `BlendingInfo::VisitFields` and decodes all
74 original sequences. The sequence references remain native; this is a recorded second-oracle
limitation, not a change to the original corpus.

For BT.709, native libjxl `TF_709` (`lib/jxl/cms/transfer_functions-inl.h`) and jxl-oxide
`jxl-color 0.11.0/src/tf/bt709.rs` extend the linear toe below zero. Rust `jxl 0.6.0/src/color/tf.rs`
instead reflects the positive power curve. GPU image conversion, display and render transfer
now use the linear negative extension for interoperability. BT.709 itself specifies the nominal
nonnegative range; the negative extension is an explicit implementation contract. sRGB, PQ,
HLG and BT.2020 retain their existing sign-reflected extensions.

This corpus covers original color plumbing and frame composition. ICC, custom chromaticities,
gamma/DCI, HDR luminance mapping, wide-gamut spot/patch/spline/LF combinations and full ISO
conformance remain separate roadmap work.
