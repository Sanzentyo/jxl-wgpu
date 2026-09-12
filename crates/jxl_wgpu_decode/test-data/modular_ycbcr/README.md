# Modular YCbCr conformance corpus

The offline generator in `../modular_ycbcr_generator` uses libjxl to serialize the image/frame/
TOC and Modular headers and entropy of 100 original streams. The outer two-bit LF-global prefix
selects default LF quantization and local MA trees. The shared Rust case manifest verifies actual
metadata. The production GPU codec supplies neither fixture bytes nor reference pixels.

Coverage includes all 64 two-bit component-selector triples; 8/12/16/31-bit integers;
16/24/32-bit floating samples; odd dimensions and one-pixel axes; grayscale presentation;
Gaborish with one, two and three EPF iterations; 2×/4×/8× frame resampling; independently
resampled 12-bit alpha and binary32 depth; associated alpha; orientations 1–8; horizontal
and vertical groups at every 128/256/512/1024-pixel group dimension; a global color prefix;
two-pass channel ownership with empty leading passes; and three LF groups carrying extras.

`*.f32.hex` stores little-endian IEEE-754 words as eight-digit hexadecimal values: interleaved
RGBA followed by each full-resolution extra plane. Color keeps the original sRGB encoding,
unclipped range, alpha association and codestream orientation. `*.oriented.f32.hex` applies
the declared orientation. `*.progressive.f32.hex` concatenates native snapshots after zero,
one and two completed passes, including empty boundaries the GPU intentionally does not publish.

All original streams must decode with native libjxl. Its fast renderer has an existing
vertical-component restoration defect: `restoration_1` and `restoration_3` differ by about
0.075 in F32. For the six filtered cases, the generator independently expands normalized
components with scalar f64 quarter/three-quarter interpolation and edge replication, then
uses native libjxl to serialize equivalent binary32 4:4:4 streams (`*.expanded.jxl.hex`).
Native libjxl and pinned jxl-oxide 0.12.6 must agree within maxAE 2e-6 on these equivalent
streams. The GPU decodes the original subsampled streams against that independently verified
reference with the same bound. No GPU output generates a reference, and no tolerance is widened.

The alternative native simple renderer cannot serve as an odd-extent reference: its temporary
height allocation rejects a vertically expanded odd-height Modular plane. Pinned jxl-oxide
also fails to consume the original asymmetric Modular component streams, so it is used only
for the independently expanded equivalents. These oracle limitations do not reject GPU input.

Binary32 working words use a native zero-predictor tree because their sign transitions can
overflow libjxl's signed predictive-residual range. Other cases use its learned weighted tree.
The generator explicitly emits global, LF and pass streams with their native Modular stream IDs.
Scalar input patterns and equivalent-stream expansion belong solely to offline fixture generation.

Reproduce with CMake, a C++17 compiler, pkg-config and system Highway, Brotli, LCMS2 and libjxl.
Use a libjxl checkout at `a7a9c787341cf703dede03c2009fa460cae5e5df` (v0.12.0); the CMake project
rejects a different commit. From the workspace root:

```sh
cmake -S crates/jxl_wgpu_decode/test-data/modular_ycbcr_generator \
  -B /tmp/jxl-modular-ycbcr-build -DCMAKE_BUILD_TYPE=Release \
  -DJXL_SOURCE=/absolute/path/to/libjxl
cmake --build /tmp/jxl-modular-ycbcr-build --target generate_modular_ycbcr --parallel 8
cargo run -p jxl_wgpu_decode --example regenerate_modular_ycbcr -- \
  /tmp/jxl-modular-ycbcr-build/generate_modular_ycbcr
cargo test -p jxl_wgpu_decode --test modular_ycbcr -- --test-threads=1
```

The GPU tests require an actual adapter and compare native color, selected RGB/extra components,
40-byte GPU windows, 43-byte transport fragments and immutable progressive snapshots. Separate
tests check exact component-expansion admission, a one-byte budget shortfall, retry and cancellation.
This corpus does not yet cover Modular YCbCr transforms, patch/spline/noise dictionaries or mixed
frame composition. Those remain part of the active full JPEG XL goal.
