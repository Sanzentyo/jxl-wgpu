# CMYK-suggested XYB output

This offline generator requires libjxl **0.12.0**, Little CMS **2.19**, and C++17.
It reuses the six unchanged CMYK profiles in `jxl_wgpu/test-data/icc/lut`: 8-bit tables,
16-bit tables and A/B methods, each with XYZ and Lab PCS. No production GPU output supplies
input samples, reference values, or acceptance bounds.

From the workspace root, using a new output directory:

```sh
mkdir -p .git/cmyk-xyb-regenerate
c++ -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off \
  crates/jxl_wgpu_decode/test-data/cmyk_xyb_generator/main.cpp \
  $(pkg-config --cflags --libs libjxl libjxl_cms lcms2) \
  -o .git/cmyk-xyb-regenerate/generator
.git/cmyk-xyb-regenerate/generator crates/jxl_wgpu/test-data/icc/lut \
  .git/cmyk-xyb-regenerate/references
diff -rq crates/jxl_wgpu_decode/test-data/cmyk_xyb .git/cmyk-xyb-regenerate/references
cargo test --locked -p jxl_wgpu_decode --test embedded_icc xyb::cmyk:: -- --test-threads=1
```

The **73 files** contain 12 three-frame streams, 12 native F32 sample files, 48 ICC reference
files and a manifest. Each 17×9 stream has F32 Black/Alpha/spot extras. Black alternates between
indices 0 and 2, and Alpha is index 1. All frames replace the full canvas, have positive durations
1/2/3, and save no references. Both Modular and VarDCT encode XYB. These inputs satisfy the
[ICC XYB reference restriction](../../../../docs/ICC_COLOR.md#xyb-reference-validity).

Native decoding verifies original profile bytes, geometry, Black index, presentation durations,
and the actual DATA encoding: unbounded linear D65 BT.709. Spot rendering and alpha unassociation
are disabled. Every extra sample must equal the encoded dyadic input exactly. The `.f32` files
interleave three linear RGB components followed by all three extras, in frame/pixel order.

Independent F64 CIE/Bradford geometry connects that linear basis to PCS. It applies v4 reference
black compensation for Perceptual/Saturation, or decimal-D50-to-stored-media-white scaling for
Absolute, then the selected reverse LUT stages. Little CMS uses XYZ doubles and CMYK float
percentages with `NOOPTIMIZE | NOCACHE`; stored references use unit ink amounts. Each intent has
more than 153 pixels where generated K differs from the complemented stored Black by more than
the complete primary uncertainty plus 0.01. The two quantities cannot substitute for each other.

The primary input interval is the existing native XYB contract `(1 + abs(linear)) / 1024`.
Matrix coefficients propagate it into PCS; the existing independent LUT/curve/CLUT/Lab stages
propagate it onward. F32 primitive uncertainty and the separate native CMM interval retain their
existing definitions. Each 28-byte record stores native, scalar, primary lower/upper, native
lower/upper and the native semantics mask. All 88,128 native components must satisfy their
independent native intervals during generation. A final numeric complement adds one F32
subtraction's rounding allowance; no GPU measurement sets these intervals.

Public GPU tests check 352,512 device components in 576 presentations across four intents,
planar/interleaved F32, and whole/43-byte fragments with 256-byte entropy windows. Alpha is exact.
Linear output separately checks the native reconstruction basis. Numeric tests compare 66,096
samples: reconstructed complemented CMY and exact encoded Black/Alpha/spot. Presentation intent,
white adaptation, spot and alpha options cannot change numeric values. All three outputs remain
held and are reread after session destruction, and the byte budget must return to zero.

Numeric reconstruction uses complete ICC device storage, including generated K, followed by
independent extras. Separate private tests cover exact storage/program admission, retry,
concurrent completion and cancellation for RGB/Gray/CMYK. Existing RGB/Gray numeric and patched
LF alpha/depth substitution tests cover the shared output boundary.

This establishes these F32 full-canvas sequences. Broader source precision and sampling,
legal ICC crop/blend combinations, profile methods/ranges, HDR and the remaining full JPEG XL
requirements remain open.
