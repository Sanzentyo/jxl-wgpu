# ICC spot presentation corpus

The manifest explicitly marks **28 supported streams and four negative reference cases** with
`valid_reference_color`. The four ICC XYB sequences were previously counted as conformance
evidence, but the [F.2 audit](../../../../docs/ICC_COLOR.md#xyb-reference-validity) found forbidden
reference storage. Their source bytes and old independent calculations remain diagnostic records;
the decoder must reject them before starting a frame. Native encoder acceptance is insufficient
to establish codestream validity.

`main.cpp` uses
 libjxl **0.12.0** and Little CMS **2.19** to generate 32 JPEG XL streams,
native source planes and independent spot/color references. `reference.hpp` owns interval
propagation; shared `icc/scalar.hpp` and `icc/linear.hpp` provide independent curve, matrix
and CIE/Bradford equations. Production Rust/WGSL code does not generate reference pixels.

From the repository root:

```sh
clang++ -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off \
  -Itools/jxl_test_support/native \
  crates/jxl_wgpu_decode/test-data/icc_spots_generator/main.cpp \
  $(pkg-config --cflags --libs libjxl libjxl_cms lcms2) \
  -o /tmp/jxl-icc-spots-generator
spot_run_a=$(mktemp -d)
spot_run_b=$(mktemp -d)
/tmp/jxl-icc-spots-generator crates/jxl_wgpu_decode/test-data/embedded_icc "$spot_run_a"
/tmp/jxl-icc-spots-generator crates/jxl_wgpu_decode/test-data/embedded_icc "$spot_run_b"
diff -qr "$spot_run_a" "$spot_run_b"
diff -qr "$spot_run_a" crates/jxl_wgpu_decode/test-data/icc_spots
cargo test --locked -p jxl_wgpu_decode --test embedded_icc spots:: -- --nocapture --test-threads=1
```

The two runs reproduce all **485 files**. Profile inputs reuse the unchanged
embedded-ICC RGB per-channel gamma and Gray sampled profiles. No native encoder/decoder or CMS
is linked into the production GPU decoder.

## Cases and files

The TSV manifest explicitly declares ICC/enumerated, RGB/Gray, Modular/VarDCT, original/XYB,
still/sequence and reference validity; the Cartesian product contains 32 cases. Rust checks those fields against
the actual inventory. Enumerated RGB is Display-P3/sRGB; enumerated Gray is D65/sRGB. The
17×9 canvas has clockwise or counterclockwise orientation, and every color declaration is F32.
Sequences have three presentations with durations 1/2/3, two saved reference versions, source-over
color/extra composition and associated first alpha. Stills have unassociated alpha. VarDCT
also contains progressive AC updates.

Nine extras are Depth32F, Spot1, Alpha7, Spot12, Spot32F, Thermal8, Spot6, Spot10 and Alpha15.
Five noncommuting inks include zero/unit/fractional solidity and zero/full/fractional samples.
Composed spot planes can exceed one; rendered device colors include signed and above-one values.
The main alpha is declaration 2, separate from the later alpha and every ink.

- `.jxl`: the native encoded stream.
- `.source.f32`: untinted RGBA then nine extra planes, frame by frame, in codestream orientation.
  Original/composed output is in its original device/enum domain. An unreferenced XYB still
  retains linear RGB in BT.709 for ICC/Gray or Display-P3 for enumerated RGB.
- `.rendered.f32`: the same native spot-stage output, with the four ICC XYB sequences composed
  independently as described below. A native Gray CMS may discard the other colored components;
  the generator checks its first channel and every RGB channel against the independent intervals.
- `.uncoalesced-linear.f32`: native linear reconstruction for the four ICC XYB sequences.
- `.{preserve,render}.{rgb,gray,linear}.bounds`: a little-endian pair of F64 lower/upper bounds
  for every color component in frame/pixel/channel order, without alpha or a header.
- `.{preserve,render}.{rgb,gray,linear}.native.f32`: separate Little CMS color values, exact
  device bypass samples, or independent CIE values when both endpoints are enumerated.

The native decoder rejects an explicit request for some original per-channel/sampled profiles.
For original coding, the generator verifies that its default data profile is byte-identical to
the requested original and reads those values directly. Native XYB blending cannot insert a
general inverse ICC connection. For those four sequences, uncoalesced native linear frames
enter Little CMS, then independent F64 source-over and ink equations construct the references.
These four rendered sequences are diagnostic calculations for invalid streams, not decoded conformance references.

## Domain, precision and ownership

Rendering follows the existing
[libjxl stage placement](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/dec_cache.cc)
and [ordered ink equation](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/render_pipeline/stage_spot.cc).
The actual post-reference source domain determines where inks mix. Gray ICC consumes one
device component; linear/RGB has three components before any requested Gray projection.
References remain untinted. Same-profile output bypasses ICC curves and preserves extended
values. Actual ICC connections retain the unit-domain curve contract; the native float CMS
comparison applies that device boundary explicitly, while preserving signed linear RGB.

Original lossless Modular source intervals use `1e-5 * (1 + abs(value))`; other native
reconstruction uses the existing `1/1024 * (1 + abs(value))` codec bound. ICC XYB sequence
intervals start before their original-profile inverse and propagate through that inverse and
reference blending. Extra-channel uncertainty is `2e-6 * (1 + abs(value))`. Interval products
carry all coverage endpoints through ordered inks, including signed/above-one coverage.
Matrix error is `4e-7 * (1 + magnitude + coefficient_sum)` before inverse curves, with `2e-7`
output rounding. Each ink adds `2e-7 * (1 + maximum_magnitude)` arithmetic uncertainty.
This handles steep/plateaued curves without a fixed output-code tolerance.

The generator separately validates **86,904 native CMM components** against the shared
method-specific native intervals. All native checks require finite values and the independently
derived bounds; no mask skips a failing comparison. Production checks use their own curve and
codec intervals. The runtime color test compares **1,002,456 components / 2,808 final images**
and retains/rereads progressive and final updates. Three alpha policies, planar/interleaved layouts, Apply/Keep,
whole input and 43-byte fragments with a 256-byte GPU window produce identical canonical final
words. Alpha remains exact across all variants. Numeric extra tests add **71,604 native sample
comparisons** and Render/Preserve equality.

The private ICC spot stage adds one source-sized surface and 32 bytes per ink, with no uniform
or image readback. A unit test includes one-byte-short admission, rollback, retry, shared program
reuse, reverse completion and cancellation for all four original source/profile shapes.
Four additional compositor-only headers combine an ink with dynamic black-validation profiles
to check that the GPU status map callback also retains and releases every spot allocation.
Preserve omits the stage. Full profile/range/render and JPEG XL conformance remain open.
