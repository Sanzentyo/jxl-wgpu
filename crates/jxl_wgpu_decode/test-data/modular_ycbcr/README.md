# Modular YCbCr conformance corpus

The offline generator in `../modular_ycbcr_generator` uses libjxl to serialize the image/frame/
TOC and Modular headers and entropy of 244 original streams. The outer two-bit LF-global prefix
selects default LF quantization and local MA trees. The shared Rust case manifest verifies actual
metadata. The production GPU codec supplies neither fixture bytes nor reference pixels.

Coverage includes all 64 two-bit component-selector triples; 8/12/16/31-bit integers;
16/24/32-bit floating samples; odd dimensions and one-pixel axes; grayscale presentation;
Gaborish with one, two and three EPF iterations; 2×/4×/8× frame resampling; independently
resampled 12-bit alpha and binary32 depth; associated alpha; orientations 1–8; horizontal
and vertical groups at every 128/256/512/1024-pixel group dimension; a global color prefix;
two-pass channel ownership with empty leading passes; and three LF groups carrying extras.

The 144 transformed streams add all 42 RCT types; default Squeeze with every selector triple;
explicit append/in-place horizontal and vertical Squeeze; one-pixel axes with empty residuals;
all group dimensions; three LF groups and progressive residual passes; integer and floating
working words; component and three-channel palettes; and ordered RCT/Palette/Squeeze stacks.
Separate cases transform residual channels, resample independent extras and restore subsampled
components after Palette/Squeeze. These are frame-level transforms; local per-group transform
combinations remain outside this corpus.

Each transformed stream has a native `*.topology` sidecar. Its header contains the meta-channel
count, total channel count, transform count and number of inverse Squeeze channel operations.
Each following row contains width, height, horizontal shift and vertical shift for one channel,
including empty residuals. The Rust metadata test compares every channel and transform before
planning the full decoder; the sidecars are never production inputs.

`*.f32.hex` stores little-endian IEEE-754 words as eight-digit hexadecimal values: interleaved
RGBA followed by each full-resolution extra plane. Color keeps the original sRGB encoding,
unclipped range, alpha association and codestream orientation. `*.oriented.f32.hex` applies
the declared orientation. `*.progressive.f32.hex` concatenates native snapshots after zero,
one and two completed passes, including empty boundaries the GPU intentionally does not publish.

All original streams must decode with native libjxl. Its fast renderer has an existing
vertical-component restoration defect: `restoration_1` and `restoration_3` differ by about
0.075 in F32. For the seven filtered cases, the generator independently expands normalized
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
The binary32 Squeeze case uses positive source samples to avoid signed overflow in native
forward Squeeze; it does not establish arbitrary signed binary32 transform conformance.
Negative binary32 samples remain covered by the untransformed cases.
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
cargo test -p jxl_wgpu_decode --lib transformed_ycbcr_topology -- --test-threads=1
cargo test -p jxl_wgpu_decode --test modular_ycbcr -- --test-threads=1
```

The GPU tests require an actual adapter and compare native color, selected RGB/extra components,
40-byte GPU windows, 43-byte transport fragments and immutable progressive snapshots. Separate
tests check 19 exact-budget admission cases, a one-byte budget shortfall, retry and cancellation.
Short whole-stream uploads are admitted by actual allocation size, including those below the
40-byte limit needed for segmented entropy streams. This corpus does not yet cover per-group
transform stacks, patch/spline/noise dictionaries or mixed frame composition. Those remain part
of the active full JPEG XL goal.
