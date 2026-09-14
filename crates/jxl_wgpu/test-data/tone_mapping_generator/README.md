# Native luminance references

`main.cpp` includes and invokes `jxl::Rec2408ToneMapperBase` from libjxl 0.12.0,
commit `a7a9c787341cf703dede03c2009fa460cae5e5df`. It does not copy the tone equations.
The fixed 18 ranges use source whites 100/1000/4000/10000 nits, lower display whites
80/255/1000 nits, and black pairs 0→0 or 1→0.01 nit. Each has 257 ramp samples and 20
boundary/extended-chromaticity samples, for 4,986 rows. Columns are source black/white,
target black/white, input XYZ and native output XYZ. The primitive uses Y directly; only its
neutral black cap is re-expressed as D50 XYZ. Input luminance is nonnegative; signed chromatic
components and above-white inputs remain covered. Negative luminance and protected/degenerate
range policies use the separate F64/GPU tests; the native primitive is not a reference for
those extensions or JPEG XL's `linear_below` constraint.

The independent Rust oracle uses F64 ST 2084 and Bernstein cubic evaluation, without production
coefficients. Both native and GPU results must meet `8e-5 * (1 + abs(reference))`, declared before
execution for the two PQ evaluations and F32 coefficients. Every native row and every GPU color
component is asserted. GPU Scalar/Lanes32/Tile16x16 also check padded source/output planes and
guard regions. The [policy and integration evidence](../../../../docs/TONE_MAPPING.md) distinguish
this explicit luminance mapping from native automatic HLG adaptation and gamut mapping.

From the workspace root, with the pinned libjxl source checked out in `$LIBJXL_SOURCE`:

```sh
mkdir -p .git/tone-regenerate
clang++ -std=c++17 -O2 -Wall -Wextra -Werror -DNDEBUG -ffp-contract=off \
  -isystem "$LIBJXL_SOURCE" crates/jxl_wgpu/test-data/tone_mapping_generator/main.cpp \
  -o .git/tone-regenerate/generator
.git/tone-regenerate/generator .git/tone-regenerate/native.txt
cmp crates/jxl_wgpu/test-data/tone_mapping/native.txt .git/tone-regenerate/native.txt
cargo test --locked -p jxl_wgpu --test icc tone_mapping:: -- --test-threads=1
cargo test --locked -p jxl_wgpu_decode --test hdr tone_mapping:: -- --test-threads=1
```

The external headers are system includes so their existing unused-parameter warnings do not
mask warnings in this generator. Source remains frozen while compilation, generation or tests run.
