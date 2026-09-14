# Extended extra-channel upsampling

This offline generator pins libjxl 0.12.0 commit
`a7a9c787341cf703dede03c2009fa460cae5e5df`. C++17, CMake and system Highway, Brotli and
Little CMS are required. Production does not link or call these native codecs.

From the workspace root, with a clean checkout at the pinned commit and a new output directory:

```sh
cmake -S crates/jxl_wgpu_decode/test-data/extra_upsampling_generator \
  -B .git/extra-upsampling-build -DJXL_SOURCE=/absolute/path/to/libjxl \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build .git/extra-upsampling-build --target generate_extra_upsampling -j 4
.git/extra-upsampling-build/generate_extra_upsampling .git/extra-upsampling-regenerated
diff -rq crates/jxl_wgpu_decode/test-data/extra_upsampling .git/extra-upsampling-regenerated
cargo test --locked -p jxl_wgpu_decode --test extra_upsampling -- --test-threads=1
```

`CMAKE_PREFIX_PATH` locates the installed dependencies; adjust it for other hosts. For native
diagnostics, add `-DCMAKE_CXX_FLAGS="-DJXL_IS_DEBUG_BUILD=1 -DJXL_DEBUG_ON_ALL_ERROR=1"` when configuring.
No upstream source modification is required.

The 139 files contain 68 Modular codestreams, their 68 little-endian `.coded` source-word files,
`manifest.tsv`, and two native VarDCT controls. The manifest columns are name, width, height,
color factor and effective extra factor. Color is original sRGB with 12-bit integer samples.
Extras are F32 Depth, unassociated 12-bit Alpha, F32 Spot and 17-bit SelectionMask. Every extra
uses image-header dimension shift 3 and a frame factor 1/2/4/8, giving effective 8/16/32/64.
Color factors 1/2/4/8 cross 129×97, 1×65, 65×1 and 7×5; 2051×33 with color factor one adds LF/pass
group boundaries. Tests verify the declared grids, precision and every coded word against the
deterministic source formula, before constructing expected pixels.

libjxl encodes the native Modular substreams with zero prediction and local trees. The small
explicit frame-header writer is necessary because this libjxl version rejects effective extra
factors above eight. The legal cumulative limit is 64 in ISO/IEC 18181-1:2024 F.2 (FrameHeader).
Native decoding validates every 8× control. The 8-first-then-2/4/8 reconstruction order also
matches jxl-render 0.12.4; this corpus is not a claim of complete Part 1 conformance.

The two VarDCT controls are 24×16 and 272×24, with four F32 extras on 8× grids. Each input 8×8
tile is a constant small dyadic value, so native box downsampling preserves known coded samples
exactly. Zero prediction avoids signed floating-word residual overflow in the native encoder.
Tests scale nominal dimensions and every frame factor together by 1/2/4/8. They verify unchanged
color grids, group counts and every entropy section. This produces 8× through 64× extras with
actual VarDCT color entropy, across a group boundary. Standard and custom weights give 16 cases.

The primary scalar reference uses F64 sums, explicit repeated-edge mirror coordinates and a
separately expanded symmetric kernel. Each 5×5 filter propagates input intervals, bounds 25
products and 25 additions by `gamma(50) * sum(abs(weight) * input_magnitude)`, then monotonically
clamps the interval to the neighborhood range. The complete first-stage grid is retained.
An explicit negative comparison proves that prematurely cropping it changes more than 100
samples beyond the derived rounding interval. No GPU measurement determines a bound.

Rust CPU normalization uses one F32 division; libjxl uses a rounded reciprocal and multiplication.
Their independent bounds use round-to-nearest unit roundoff `2^-24`, with the native F64
reciprocal's rounding included. For WGSL, multiplication/addition may round in either direction
(`2^-23`) and division allows 2.5 ULP; integer operands here are exactly representable. A separate
smallest-normal allowance covers flushed filter results. See
[WGSL floating-point accuracy](https://www.w3.org/TR/WGSL/#floating-point-accuracy).

The Rust oracle cannot validate single-sample coded axes: jxl-render 0.12.4
`PaddedGrid::mirror_edges_padding` reads padding as source when the axis is smaller than its
two-sample padding. Tests explicitly require at least two coded samples on both axes for that
oracle (35 Modular cases including custom variants). All thin cases still use the independent
repeated-mirror reference on GPU, and the 21 native Modular 8× controls include them. Native
libjxl cannot decode the larger effective factors; no missing oracle silently substitutes for it.
All 16 VarDCT cases satisfy the Rust oracle's geometry requirement; four have native 8× controls.

GPU tests check 840 Modular outputs (four extras plus RGBA, whole and 43-byte fragmented input
with 256-byte GPU windows) and 128 VarDCT scalar outputs. Held outputs are reread after session
destruction; both transports must agree exactly and release their byte reservations. Separate
40-byte-window tests check exact intermediate admission, pressure/retry and cancellation.
Broader declared shifts, mixed factors between extras, native output quantization, frame features,
LF dependencies and composition at these extended factors remain separate conformance work.
