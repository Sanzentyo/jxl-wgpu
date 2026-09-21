# CMYK composition and ICC conversion

The offline generator requires libjxl **0.12.0** and Little CMS **2.19**. It reuses the six
unchanged CMYK profiles from `jxl_wgpu/test-data/icc/lut`: `mft1`, `mft2`, and A/B methods,
each with XYZ and Lab PCS. Target profiles are existing RGB/Gray LUTs with the opposite PCS.
No production GPU pixel output participates in generation.

The manifest explicitly identifies each source, target, target channel count, coding mode,
and Black extra-channel index. Mode 0 is original Modular; mode 1 is original VarDCT;
mode 2 is VarDCT with 4:4:4 YCbCr components reconstructing complemented CMY (YCCK).
Every stream has three 17 × 9 presentations, independent F32 Black/Alpha/spot extras,
and reference slots 1 and 2. Alpha is extra 1; Black alternates between extra 0 and 2.
Frame 1 adds Black to reference 1. Frame 2 multiplies Black by reference 1 while its color
blend reads reference 2. Spot coverage replaces independently. Black addition includes
values above one, so the ICC complement includes values below zero.

The native public encoder overrides YCbCr to None in `encode.cc`. Generation therefore
has three explicit steps: native component seeds, metadata assembly, then independent native
decode and ICC evaluation. `assemble_cmyk` uses the shared `as_ycbcr_444` fixture assembler.
It preserves image metadata, blend contracts and all entropy sections, and verifies the actual
transform flag. Original-color fixture regeneration uses this same assembly function and is
checked against its existing frozen bytes. Container-relative offsets are resolved through
the parsed codestream before comparing sections.

From the workspace root, using new output directories:

```sh
mkdir -p .git/cmyk-regenerate
c++ -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off \
  crates/jxl_wgpu_decode/test-data/cmyk_generator/main.cpp \
  $(pkg-config --cflags --libs libjxl lcms2) -o .git/cmyk-regenerate/generator
.git/cmyk-regenerate/generator seeds crates/jxl_wgpu/test-data/icc/lut \
  .git/cmyk-regenerate/seeds
cargo run --locked -p jxl_wgpu_decode --example assemble_cmyk -- \
  .git/cmyk-regenerate/seeds .git/cmyk-regenerate/assembled
.git/cmyk-regenerate/generator references crates/jxl_wgpu/test-data/icc/lut \
  .git/cmyk-regenerate/assembled .git/cmyk-regenerate/references
diff -rq crates/jxl_wgpu_decode/test-data/cmyk/generated .git/cmyk-regenerate/references
cargo test --locked -p jxl_wgpu_decode --test embedded_icc cmyk:: -- --test-threads=2
```

The 181 files contain 18 `.jxl` streams, 18 native `.f32` images, 144 `.reference` files,
and `manifest.json`. Native images interleave three CMY components and all three extras per
pixel, for all three coalesced frames. Native original and data ICC queries must equal the
source bytes. Spot rendering and alpha unassociation are disabled during this reconstruction.
The reference filename suffix is `_<render_spots:0|1>_<intent:0..3>.reference`.

Independent C++ F64 equations apply the source-domain spot ink, complement all four CMYK
components, and evaluate the selected forward/PCS/reverse LUT stages. Little CMS consumes
the same independently prepared amounts through its percentage-valued CMYK float interface,
with `NOOPTIMIZE | NOCACHE`, for all four intents. Native and primary intervals remain separate;
the existing six-F32/u32 record stores both, including native semantics mask 32 where needed.
Every native value must fall inside its independently derived native interval during generation.

The existing LUT stage intervals use `epsilon = 4e-7`. Source component uncertainty remains
zero for original Modular and `2e-5` for VarDCT, including YCbCr reconstruction. The exact dyadic
extra-channel data and Add/Multiply operations introduce no source uncertainty. A spot stage
propagates `(1 - strength) * source_error` and adds `8*epsilon*(1 + abs(value))`; the complement
adds `epsilon*(1 + abs(1 - value))`. Subsequent LUT/curve/CLUT/Lab stages propagate these bounds.
No interval is fitted to GPU output or widened for a native CMM difference.

There are 132,192 independent/native color components. The GPU test checks 528,768 components
across 1,728 presentations: both F32 layouts, whole/fragmented input, all intents, and spot
Render/Preserve. Alpha is bit-exact. All six original numeric channels are also compared with
native output without inks; Modular and extra words are exact, VarDCT color uses its source
bound. Three simultaneous frame slots allow all outputs to remain held and reread after session
release; the transient budget must return to zero. The separate official case retains the
upstream limits. Complete device layouts and CMYK-suggested XYB have separate
[output evidence](../../../../docs/ICC_COLOR.md#cmyk-suggested-xyb-and-numeric-output).
Wider sampling/profile/range coverage and the full JPEG XL goal remain open.
