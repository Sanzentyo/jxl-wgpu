# Enumerated RGB to requested ICC

This offline corpus reuses all 228 `original_color` streams: 114 stills and 114 six-physical-frame
sequences, producing 570 presentations. Both codecs, original RGB, XYB and YCbCr, RGB/Gray,
integer/F32, standard/custom primaries, D65/E/DCI/custom whites, and Linear/sRGB/BT.709/Gamma/DCI
are included. Sequences retain the existing crops, hidden frames, all five blend modes, independent
alpha and overwritten reference slots. No source stream, original reference or ICC profile changes.

The manifest explicitly pairs each source with one of nine target profiles; this is not their
Cartesian product. Every pair uses all four ICC intents. Targets are RGB/Gray matrix/TRC,
tinted-media-white and chromatic-black profiles, identity MPE, v4 8-/16-bit XYZ/Lab LUTs,
Gray mBA and a v2 XYZ LUT. Native generator method selection uses a typed definition table.
The independently constructed LUT recipes must reproduce the exact existing profile bytes.

The 913 files contain one manifest and 912 references, with 4,049,280 color components.
The decoder checks these through interleaved progressive and planar final-only output, each
with whole input and 43-byte async fragments under a 256-byte GPU entropy window. The resulting
9,120 presentations check 16,197,120 components. Final images agree exactly across those modes;
alpha agrees bit-for-bit with original-color GPU output. Every retained update is reread after
session release, and dropping outputs returns the complete byte budget.

## Independent references

`export_rgb_icc_sources` uses the shared test-only `oracles::color` module. Original-device images
and composed sequences use the existing libjxl 0.12.0 references. Unreferenced XYB stills use
jxl-oxide 0.12.6's unbounded linear reconstruction followed by independent f64 geometry. This
preserves values lost by the native Gamma/DCI original-OETF black floor. The existing calibrated
XYB primary and Gray-luma conventions remain unchanged; see the
[original-color recipe](../original_color_generator/README.md).

Input uncertainty stays `1e-5 * (1 + abs(sample))` for original Modular and
`(1 + abs(sample)) / 1024` otherwise. Independent piecewise EOTFs and signed CIE/Bradford matrices
propagate it to exact encoded ICC D50 PCS. The PCS interval additionally includes
`5e-6 * (1 + abs(PCS))` for the existing F32 color-output arithmetic. These bounds precede target
selection and all GPU execution. Each scratch input pixel stores three pairs of little-endian
f64 center/radius values (48 bytes).

The C++ generator uses Little CMS 2.19's XYZ-double input, `NOOPTIMIZE | NOCACHE`, and the exact
target bytes. Matrix/TRC native output retains the established unit-output contract; LUT native
stages use the separately modeled unbounded CMM convention. Independent shared f64 ICC equations
propagate source-box corners through monotone inverses, or radii through each LUT matrix, curve,
Lab boundary and interpolation gradient. Native precision is derived separately with zero codec
input uncertainty. No production parser, production color matrix, GPU pixels or measured GPU error
sets any reference or interval.

Each 28-byte reference contains six F32 values (native, primary center, primary lower/upper,
native lower/upper) and a u32 native-semantics mask. Bit 5 retains native LUT extensions from the
[resident LUT oracle](../../../jxl_wgpu/test-data/icc_generator/README.md).
Bit 6 records Little CMS 2.19's `cmsPERCEPTUAL_BLACK_Z = 0.00287`, separately from the primary
reference-black Z `0.0028646`. The X/Y constants agree. The pinned
[public header](https://github.com/mm2/Little-CMS/blob/lcms2.19/include/lcms2.h) and
[black detection](https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmssamp.c) define this
native convention. Every component, including both mask bits together, must satisfy both its
primary/GPU and native interval. No mask skips an assertion.

Compared with relative intent, disjoint primary intervals identify 562,786 perceptual,
558,471 saturation and 83,309 absolute components. The original and bounded-input comparisons
therefore test effective intent execution, beyond enum selection.

## Reproduction

From the workspace root, with Rust 1.98+, C++17 and Little CMS 2.19:

```sh
mkdir -p .git/rgb-icc-regenerate
cargo run --locked -p jxl_wgpu_decode --example export_rgb_icc_sources -- \
  .git/rgb-icc-regenerate/sources
clang++ -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off \
  -Itools/jxl_test_support/native -Icrates/jxl_wgpu/test-data/icc_generator \
  crates/jxl_wgpu_decode/test-data/rgb_icc_generator/main.cpp \
  $(pkg-config --cflags --libs lcms2) -o .git/rgb-icc-regenerate/generator
.git/rgb-icc-regenerate/generator crates/jxl_wgpu/test-data/icc \
  .git/rgb-icc-regenerate/sources .git/rgb-icc-regenerate/references
diff -rq crates/jxl_wgpu_decode/test-data/rgb_icc .git/rgb-icc-regenerate/references
cargo test --locked -p jxl_wgpu_decode --test original_color icc:: -- --test-threads=2
```

Both output directories must be new. Two independent source exports and native generations
reproduce every file exactly. Fixture generation and GPU tests run against fixed source trees.
The independent ICC evaluator is shared in `tools/jxl_test_support/native/icc/rgb.hpp` with the
[HDR/ICC corpus](../hdr_icc_generator/README.md); extracting it preserves all 913 SDR files exactly.
This corpus establishes SDR connection coverage. Later corpora separately cover image-relative
HDR, ICC spot rendering and CMYK image plumbing. Physical display adaptation, arbitrary MPE
range/conditioning and full JPEG XL remain open.
