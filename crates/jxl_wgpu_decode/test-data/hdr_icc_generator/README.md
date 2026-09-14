# HDR and ICC connections

Image-relative linear RGB uses `intensity_target` nits for unit white. The ICC connection maps
that image white to PCS Y=1, then applies the selected ICC rendering intent and profile method.
PQ converts absolute light using the 10000-nit scale; HLG uses its primary-dependent display OOTF.
This is an explicit image-white connection, not display adaptation or automatic tone/gamut mapping.
ICC-absolute intent still applies the existing media-white connection; it does not infer a new
physical display peak from an optional ICC `lumi` tag. ICC.1:2022 distinguishes media-relative PCS
white in D.5 and emissive-device luminance metadata in 9.2.33:
[ICC specification](https://www.color.org/specification/ICC.1-2022-05.pdf).

## Corpus and independent precision

The 225 files contain one explicit manifest and 224 references. They reuse all 56 native HDR
streams and 80 presentations without changing their bytes. Each source is paired with one of
nine existing RGB/Gray profiles: matrix/TRC, tinted-white/chromatic-black, identity MPE, v4
8-/16-bit XYZ/Lab LUTs, Gray mBA and a v2 LUT. Every pair covers all four rendering intents;
this is a pairing table, not a Cartesian product of streams and profiles.

`export_hdr_icc_sources` computes F64 display-relative light and adapted PCS from the pinned
libjxl 0.12.0 references. Original RGB and composed sources use native original values; unreferenced
XYB uses native linear reconstruction. The shared HDR oracle applies the existing negative-luminance
HLG extension. Input intervals retain normalized `1e-5` for original Modular and `1/1024` for
XYB/VarDCT. They propagate through the nonlinear transfer, coupled OOTF and signed primary/Bradford
matrix, adding the existing `5e-5*(1+abs(PCS))` transfer allowance. No GPU values set these bounds.

The C++ generator shares the original SDR corpus's independent ICC evaluator in
`tools/jxl_test_support/native/icc/rgb.hpp`. Little CMS 2.19 consumes the PCS double values with
`NOOPTIMIZE | NOCACHE`; the primary F64 evaluator independently applies profile curves, matrices,
LUT interpolation, Lab conversion, rendering intent and error propagation. The unchanged profile
recipes must reproduce the exact LUT bytes. Each 28-byte record stores native/primary values,
primary bounds, native bounds and the established native-semantics mask. Every native component
must meet its own bounds, and every GPU component must meet the primary bounds; no mask skips a
GPU assertion. See the [SDR connection recipe](../rgb_icc_generator/README.md) for CMM boundary policy.

There are 489,536 independent/native color components. GPU tests check them 1,958,144 times
through 1,280 presentations: planar/interleaved output, whole input and 256-byte GPU windows with
43-byte fragments. Interleaved requests also retain progressive updates. Final words agree across
all four configurations, held updates remain immutable, alpha equals original GPU alpha bit-for-bit,
and releasing the image returns both input and GPU byte budgets to zero.

A separate full-stream test checks every HDR source directly against independent PCS equations.
The reverse connection uses all eight existing embedded-ICC RGB/Gray × Modular/VarDCT × RGB/XYB
sources at 100/255/1000/4000 nits. Only the tone-mapping header changes; compressed ICC and all
physical frame bytes are asserted unchanged, and jxl-oxide independently parses the new intensity.
XYB's existing native linear reference and uncertainty scale by old/new intensity before the
output matrix/OOTF/OETF. Original ICC uses the existing independent scalar linear references and
their absolute `2e-4` bound. PQ/HLG, BT.2020/Display-P3, both layouts and both input modes produce
512 checked outputs, retaining exact alpha and final words across input/layout configurations.

The resident ICC primitive independently checks 11,664 components across Scalar/Lanes32/Tile16x16,
three primary sets, nine intensities, both HDR transfers and both directions, including black,
signed values and HLG's near-unity threshold. These checks retain the predeclared `5e-5` transfer
and `5e-7` matrix arithmetic allowances before nonlinear error propagation.

## Reproduction

From the workspace root with Rust 1.98+, C++17 and Little CMS 2.19:

```sh
mkdir -p .git/hdr-icc-regenerate
cargo run --locked -p jxl_wgpu_decode --example export_hdr_icc_sources -- \
  .git/hdr-icc-regenerate/sources
clang++ -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off \
  -Itools/jxl_test_support/native -Icrates/jxl_wgpu/test-data/icc_generator \
  crates/jxl_wgpu_decode/test-data/hdr_icc_generator/main.cpp \
  $(pkg-config --cflags --libs lcms2) -o .git/hdr-icc-regenerate/generator
.git/hdr-icc-regenerate/generator crates/jxl_wgpu/test-data/icc \
  .git/hdr-icc-regenerate/sources .git/hdr-icc-regenerate/references
diff -rq crates/jxl_wgpu_decode/test-data/hdr_icc .git/hdr-icc-regenerate/references
cargo test --locked -p jxl_wgpu --test icc hdr:: -- --test-threads=1
cargo test --locked -p jxl_wgpu_decode --test hdr -- --test-threads=1
```

All generation and GPU jobs use frozen source. The shared evaluator must also reproduce the
existing 913-file SDR connection corpus unchanged. Physical display adaptation, ICC `lumi` policy,
tone/gamut mapping, wider HDR/CMYK/feature/LF combinations and full-range/ISO precision conformance
remain separate requirements of the full JPEG XL goal.
