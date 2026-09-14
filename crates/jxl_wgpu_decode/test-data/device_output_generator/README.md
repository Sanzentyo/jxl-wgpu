# ICC device output references

The generator requires libjxl **0.12.0** and Little CMS **2.19**. It reuses the frozen
`jxl_wgpu/test-data/icc/lut` profiles and its 24 RGB/Gray decoder sources, plus the 18
three-frame sources in `cmyk/generated`. It decodes every stream with libjxl and requires
exact reproduction of the existing native component bytes and both queried ICC profiles.
No production Rust/WGSL output participates in reference generation.

Each source selects a target with 1, 2, 3, 4, 5 or 15 device components. The target rotation
covers all six counts through original Modular, original VarDCT and YCbCr CMYK. An equal
source/target name selects a different PCS profile so every reference exercises a connection.
Same-profile tests separately compare original samples without evaluating ICC curves.

The independently rebuilt profiles must match their frozen bytes. C++ F64 equations evaluate
the selected source LUT, PCS conversion and target LUT for all four intents. CMYK spot
presentation and sample complements use the shared original-CMYK oracle. Little CMS uses
`NOOPTIMIZE | NOCACHE` and explicit component formatters. Its CMYK/multicolor percentages are
converted to unit ink amounts. Alpha is excluded from every color transform.

Source uncertainty is zero for original Modular and `2e-5` for VarDCT color, including YCbCr;
independent Black/alpha/spot samples remain exact. The shared CMYK oracle propagates ink and
complement error, and the existing LUT equations propagate source bounds through every stage.
Native reconstruction and native-CMM intervals remain distinct. The seven-word reference
record retains native/exact/lower/upper/native-lower/native-upper and the native semantics mask.
Every native value is checked during generation; no bound is fitted to GPU output.

From the workspace root, using new output directories:

```sh
mkdir -p .git/device-output-regenerate
c++ -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off \
  crates/jxl_wgpu_decode/test-data/device_output_generator/main.cpp \
  $(pkg-config --cflags --libs libjxl lcms2) -o .git/device-output-regenerate/generator
.git/device-output-regenerate/generator crates/jxl_wgpu/test-data/icc/lut \
  crates/jxl_wgpu_decode/test-data/cmyk/generated .git/device-output-regenerate/references
diff -rq crates/jxl_wgpu_decode/test-data/device_output .git/device-output-regenerate/references
cargo test --locked -p jxl_wgpu_decode --test embedded_icc device:: -- --test-threads=1
```

The 241 files contain the manifest and 240 reference files, totaling 403,920 independent/native
components. Decoder checks cover 4,224 converted presentations and 624 same-profile presentations,
U8/F32, planar/interleaved output, reversed physical component order, alpha presence/association,
spot Render/Preserve and whole/256-byte bounded input. Outputs remain held through session release
and are reread; the transient budget must return to zero. The additional association interval
accounts for one binary32 multiplication before applying the independent U8 quantizer.

The separate resident packer test covers 1,152 guarded dispatches with 1/2/3/4/5/15 components,
all eight orientations, thin/non-square images, padded input, unaligned output rows/planes,
explicit component permutations and all alpha conversions. Wider device-space/profile/range
conformance, XYB-to-original CMYK reconstruction and HDR remain full JPEG XL requirements.
