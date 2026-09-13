# Modular YCbCr conformance corpus

The offline generator in `../modular_ycbcr_generator` uses libjxl to serialize the image/frame/
TOC and Modular headers and entropy of 464 original streams. The outer two-bit LF-global prefix
selects default LF quantization and local MA trees. The shared Rust case manifest verifies actual
metadata. The production GPU codec supplies neither fixture bytes nor reference pixels.

Coverage includes all 64 two-bit component-selector triples; 8/12/16/31-bit integers;
16/24/32-bit floating samples; odd dimensions and one-pixel axes; grayscale presentation;
Gaborish with one, two and three EPF iterations; 2×/4×/8× frame resampling; independently
resampled 12-bit alpha and binary32 depth; associated alpha; orientations 1–8; horizontal
and vertical groups at every 128/256/512/1024-pixel group dimension; a global color prefix;
two-pass channel ownership with empty leading passes; and three LF groups carrying extras.

The 364 streams with transforms include 232 global-transform cases and 136 cases with local
LF/pass transforms (four use both). They cover all 42 RCT types; default Squeeze with every
selector triple; append/in-place horizontal and vertical Squeeze; one-pixel axes with empty
residuals; all group dimensions; integer and floating working words; component and three-channel
palettes; ordered RCT/Palette/Squeeze stacks; independent extras; resampling; and restoration.
Eighty-four cases apply every RCT type to empty horizontal or vertical Squeeze residuals. Local
stacks include singleton edge palettes, empty residual RCT and three LF groups followed by
progressive residual passes. The manifests explicitly separate global, LF and pass transforms.

Each global-transform case has a native `*.topology` sidecar. Its header contains the meta-channel
count, total channel count, transform count and number of inverse Squeeze channel operations.
Each following row contains width, height, horizontal shift and vertical shift for one channel,
including empty residuals. Local cases have `*.local` sidecars with 652 nonempty substream records:
`stream ID`, `source COUNT` and its channel rows, then `transformed` and the same topology format.
Rust compares source/target geometry and shifts, transform headers, stream order, entropy cursor,
MA/predictor configuration, packed descriptors and the actual resident inverse plan. Sidecars are
never production inputs.

Native forward selection elides identity RCT, default Squeeze without applicable steps and
singleton palettes. The generator preserves these explicit legal headers through native
meta-application. A singleton palette stores its original color and a zero index; native libjxl
serializes its header and entropy and independently decodes the resulting stream.

`*.f32.hex` stores little-endian IEEE-754 words as eight-digit hexadecimal values: interleaved
RGBA followed by each full-resolution extra plane. Color keeps the original sRGB encoding,
unclipped range, alpha association and codestream orientation. `*.oriented.f32.hex` applies
the declared orientation. `*.progressive.f32.hex` concatenates native snapshots after zero,
one and two completed passes, including empty boundaries the GPU intentionally does not publish.

All original streams must decode with native libjxl. Its fast renderer has an existing
vertical-component restoration defect: `restoration_1` and `restoration_3` differ by about
0.075 in F32. For the eight filtered cases, the generator independently expands normalized
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
The binary32 Squeeze cases use positive source samples to avoid signed overflow in native
forward Squeeze; it does not establish arbitrary signed binary32 transform conformance.
Negative binary32 samples remain covered by the untransformed cases.

`local_squeeze_integer_31` explicitly selects the pinned scalar libjxl oracle. Its SIMD inverse
Squeeze overflows on wide working words: 42 F32 output samples differ by more than 2e-6, with
maxAE 0.440161824. The scalar build matches the GPU for color and every selected component,
including bounded execution. The original stream is unchanged, and the 2e-6 bound is unchanged.
The optional CMake `JXL_SCALAR_ONLY` configuration applies `HWY_COMPILE_ONLY_SCALAR` consistently
to the complete native build and exposes `decode_modular_ycbcr_scalar`; this is an offline oracle.
Merely selecting a Highway scalar target at runtime is insufficient when scalar code was not
compiled into the installed library. The scalar-reference case has one unfiltered final frame;
other progressive and oriented references use their existing native entry points.

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
cmake -S crates/jxl_wgpu_decode/test-data/modular_ycbcr_generator \
  -B /tmp/jxl-modular-ycbcr-scalar -DCMAKE_BUILD_TYPE=Release \
  -DJXL_SOURCE=/absolute/path/to/libjxl -DJXL_SCALAR_ONLY=ON
cmake --build /tmp/jxl-modular-ycbcr-scalar --target decode_modular_ycbcr_scalar --parallel 8
cargo run -p jxl_wgpu_decode --example regenerate_modular_ycbcr -- \
  /tmp/jxl-modular-ycbcr-build/generate_modular_ycbcr \
  /tmp/jxl-modular-ycbcr-scalar/decode_modular_ycbcr_scalar
cargo test -p jxl_wgpu_decode --lib profile::ycbcr_tests -- --test-threads=1
cargo test -p jxl_wgpu_decode --test modular_ycbcr -- --test-threads=1
```

The GPU tests require an actual adapter and compare native color, selected RGB/extra components,
40-byte GPU windows, 43-byte transport fragments and immutable progressive snapshots. Separate
tests check 29 exact-budget admission cases with one requested frame slot, a one-byte budget shortfall, retry and cancellation.
Short whole-stream uploads are admitted by actual allocation size, including those below the
40-byte limit needed for segmented entropy streams. The nine two-pass cases also compare 27 native
prefix snapshots and immutable GPU updates. Broader mixed MA/transform combinations, patch/spline/
noise dictionaries and mixed frame composition remain part of the active full JPEG XL goal.
