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
exact-word integer fixtures across the nine original and four additional analytic RGB profiles.
The resulting 39 variants retain
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

This corpus covers original color plumbing and frame composition. ICC, broader rendering policies,
HDR luminance mapping, wide-gamut spot/patch/spline/LF combinations and full ISO
conformance remain separate roadmap work.

## Analytic profiles

The additional `--analytic` case family preserves every original fixture and adds 80 streams:
40 stills and 40 six-frame sequences, 200 presentations and 168 files (including eight explicit
YCbCr component sources). The combined corpus has 228 streams, 570 presentations and 500 files.
Run the native generator and Rust completion tool with `--analytic`:

```sh
./generate-original-color ../original_color --analytic
cargo run -p jxl_wgpu_decode --example regenerate_original_color -- --analytic
```

`tools/jxl_test_support/src/fixtures/original_color/analytic.rs` declares each profile explicitly:
BT.709/E/sRGB; P3/DCI/DCI; custom Adobe-RGB primaries/custom D50/gamma 0.4545455; custom wide primaries/
D65/Linear; Gray/E/gamma 0.5; Gray/DCI/DCI. The four RGB/XYB codec modes cover all six profiles and
both still/sequence forms. P3/DCI, Adobe/D50 and Gray/E also have F32 sources above one. Both YCbCr
modes cover the first two RGB profiles. Color is 12-bit, F32, or 8-bit YCbCr components; independent
alpha is 10-bit. The original reconstruction and output error bounds above are unchanged.

The native decoder is configured with its CMS for the new RGB/XYB family. libjxl v0.12.0
`dec_xyb.cc::CanOutputToColorEncoding` rejects an explicit non-D65 Gray request for a non-XYB image,
even when it is already in that original profile. The generator keeps the decoder's original output
for those Gray inputs, asserts the actual original metadata and checks every resulting reference.

`stage_from_linear.cc::OpGamma` zeros values at or below `1e-5` before the original Gamma/DCI OETF.
This includes negative values and occurs before blending and reference storage. The general native
CMS is different: ICC Gamma para-0 clamps negatives, whereas DCI para-3 has a unit-slope negative
branch. `probe.cpp` independently reproduces both CMS directions with negative, zero and extended
F32 inputs using the public libjxl API. Compile it with the same pinned libraries and strict flags
as `main.cpp`. The GPU keeps these reconstruction and general color-conversion contracts explicit.

For XYB original RGB profiles requiring a primary/white conversion, libjxl's
`ColorEncoding::GetPrimaries(kSRGB)` uses the ICC-calibrated coordinates
`(.639998686,.330010138), (.300003784,.600003357), (.150002046,.059997204)`.
`OutputEncodingInfo::SetColorEncoding` uses those coordinates to derive its matrix; direct sRGB
and Gray skip that calibration. The inverse-opsin producer now declares its linear RGB space
explicitly. Generic BT.709 matrices continue to use the standard coordinates.

All 40 additional stills have an independent jxl-oxide comparison. XYB uses unbounded linear output,
independent F64 colorimetry, and the original OETF/black floor. Requested linear or sRGB output of
unreferenced Gamma/DCI XYB stills uses independent pre-OETF values: the original zeroed samples
cannot reconstruct their negative linear values. Sequence references remain native due to the
extra-channel source-selector defect described above. An additional absolute-XYZ output is checked
for all 80 new cases; this is an explicit output policy. The intent variants below extend original
metadata admission. Wider ICC/LUT/CMYK, HDR luminance mapping and wide-gamut
feature/LF combinations remain open in the full JPEG XL roadmap.

## Original intent variants

`tests/original_color/intents.rs` changes only the rendering-intent enum in each of the 80 analytic
sources to Perceptual, Relative, Saturation and Absolute (320 declarations). The shared helper
reparses the image, checks every other field, preserves the exact physical frame bytes and checks
the new enum with jxl-oxide's independent header reader. No checked-in stream or reference is rewritten.

The native extra-channel oracle's `--original` option requires runtime version 12000, installs
the native CMS, requests the original encoding and verifies the resulting color fields. As in
the generator above, original non-D65 Gray keeps its already-original default; XYB Gray receives
an explicit original request. Every resulting F32 word must equal the frozen original reference,
including alpha. The GPU then runs the existing whole/bounded progressive, retained-image and
final-only checks with their unchanged bounds. Missing native tools fail the test.

This follows pinned libjxl's
[`OutputEncodingInfo::SetColorEncoding`](https://github.com/libjxl/libjxl/blob/a7a9c787341cf703dede03c2009fa460cae5e5df/lib/jxl/dec_xyb.cc):
analytic original XYB reconstruction derives its matrix from primaries and Bradford-adapted white,
independently of the enum intent. Requested output remains a separate policy. Another test checks
all 320 variants against independent F64 conversion for both Bradford and absolute XYZ linear
BT.709 output, including pre-OETF references where Gamma/DCI loses information. Each selected
policy must also produce identical words across the four original intents. Existing source
intervals and `5e-6*(1+abs(reference))` output packing error remain unchanged.

The exact-word integer test applies all four intents to its 39 profile/source combinations,
yielding 156 variants with unchanged 17/31-bit color and independent 5/24/31-bit alpha through
native RGB and scalar output under whole/bounded input. This is enumerated intent admission and
reconstruction evidence; arbitrary ICC gamut methods and full JPEG XL conformance remain open.
