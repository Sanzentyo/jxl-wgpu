# Embedded ICC XYB output and invalid reference fixtures

The `animation` and `alpha` outputs are **52 negative fixtures**. The
[F.2 audit](../../../../docs/ICC_COLOR.md#xyb-reference-validity) corrected their earlier
classification as conforming composition tests. The production decoder now rejects their
post-transform references. All existing bytes and independent calculations are retained for
diagnosis; the composition calculations below do not define conforming decoded pixels.
Still output and the seven legal LF/patch substitutions remain positive tests.

These offline programs require
 libjxl **0.12.0**, Little CMS **2.19**, a C++17 compiler,
and `pkg-config`. They are not production dependencies. From the repository root:

```sh
sh crates/jxl_wgpu_decode/test-data/embedded_icc_xyb_generator/regenerate.sh /tmp/icc-xyb
diff -rq -x README.md -x alpha crates/jxl_wgpu_decode/test-data/embedded_icc_xyb /tmp/icc-xyb
```

The output contains **68 files**. All F32 references are little-endian binary32 bytes.
The existing [embedded ICC corpus](../embedded_icc_generator/README.md) supplies its exact
RGB/Gray profiles and four 17×9 Modular/VarDCT XYB streams with independent F32 alpha.

`linear.cpp` decodes every still with native default, preferred-linear and explicit-linear
requests, both with and without an explicit native CMS. It verifies the original ICC bytes
and actual DATA encoding: linear D65 BT.709, or D65 Gray. All six pixel buffers must be
byte-identical; one buffer per input is retained. `convert.cpp` connects these linear values
to both RGB and Gray ICC targets, retaining native and independent f64 results. Gray is
replicated into linear RGB before the connection. Every native component must satisfy the
separate Little CMS precision interval; alpha passes through unchanged. These steps produce
20 still references.

`animation.cpp` encodes four two-frame sequences using the same profiles and codecs.
The first frame saves reference 1 after color transformation. The second adds to that
reference. It decodes physical linear layers with both CMS configurations and verifies the
exact original ICC and actual DATA encoding. libjxl 0.12.0's coalesced path fails on the
second frame; its diagnostic is expected and is not used as the composition oracle.
`compose.cpp` independently converts each physical layer into original ICC device values
before adding. It retains native and scalar layer values, scalar composition and propagated
bounds. Four codestreams and 44 reference files comprise the animation subdirectory.
Every sequence produces values above 1.1 and distinguishes original-device addition from
addition in linear RGB, including after applying the acceptance bounds.

The GPU still tests use independent jxl-oxide linear output with an absolute **2e-5** bound,
including separate Gray luminance projection. Native XYB is cross-checked using the existing
original-color corpus's normalized **1/1024** bound. For the RGB VarDCT still, GPU versus
jxl-oxide differs by less than 1.8e-6, while both differ from libjxl by about 3.8e-5.
The original non-XYB device bound is not a valid native XYB precision assumption. Still
ICC conversion retains the established **2e-4** end-to-end bound against independent f64.
Alpha words remain exact in every comparison.

Animation bounds start with `abs(error) <= (1 + abs(linear)) / 1024`, the unchanged native
XYB reconstruction contract. `bounds.hpp` evaluates all eight outward-rounded RGB interval
corners through the independent ICC equations. It includes the primitive's existing F32
matrix/curve uncertainty, then propagates the layer intervals through addition with outward
rounding. No GPU pixel determines a reference or acceptance bound. The native CMS's own
precision interval remains separate. Near-black inverse curves explain why a fixed device
error cannot replace propagation from the linear input.

`tests/embedded_icc/xyb` checks planar/interleaved output, real Gray/RGB plane counts,
whole input and 43-byte fragments with 256-byte entropy windows, and retained frames after
session destruction. The retained diagnostic composition model ignores unused presentation intents while
reconstruction uses the original profile's own intent. Private GPU tests verify exact
output/intermediate/program admission, retry, cache reuse and completion-owned cancellation;
direct XYB output does not select an unused unsupported original intent.

Seven further metadata-only substitutions cover RGB/Gray LF and coefficient previews,
custom upsampling weights, and patched or nested LF producers in both codecs. All physical
frame bytes remain unchanged. Their established non-ICC GPU outputs provide the substitution
invariant, including dependency identities, 40-byte windows and immutable retained updates.
The original fixture suites retain independent native/scalar coverage of those codec pixels.

## Alpha and independent blend operations

A separate generator requires a clean libjxl **v0.12.0** source checkout at
`a7a9c787341cf703dede03c2009fa460cae5e5df`, the installed libraries above, and `libhwy`:

```sh
sh crates/jxl_wgpu_decode/test-data/embedded_icc_xyb_generator/regenerate-alpha.sh /path/to/libjxl /tmp/icc-xyb-alpha
diff -rq -x README.md crates/jxl_wgpu_decode/test-data/embedded_icc_xyb/alpha /tmp/icc-xyb-alpha
```

`alpha.cpp` encodes **48 streams**: RGB/Gray × Modular/VarDCT × straight/associated alpha ×
Replace/Add/Blend/MultiplyAdd/Multiply, plus eight explicit alpha-reference Blend cases. Each has two F32 layers and saves original-device
reference 1. Alpha cycles through exact 0, 1/8, 1/4, 1/2, 3/4 and 1. All 96 physical outputs
are identical with built-in and explicit native CMS. Coalesced decoding remains a diagnostic:
non-Replace second frames fail, and explicit CMS changes first-frame pixels even while DATA
ICC stays unchanged. Those coalesced values are not used as references.

`alpha_compose.cpp` converts each physical layer to original ICC device values using the
independent f64 ICC oracle, then evaluates the blend equations independently. It links the
unmodified native [`alpha.cc`](https://github.com/libjxl/libjxl/blob/a7a9c787341cf703dede03c2009fa460cae5e5df/lib/jxl/alpha.cc)
(SHA-256 `0935796743e0f4524bf0f506883d0bbc605f344bd8c939c6ba501b10c7c85b52`) to
cross-check coverage and color algebra. The [`PerformBlending` ordering](https://github.com/libjxl/libjxl/blob/a7a9c787341cf703dede03c2009fa460cae5e5df/lib/jxl/blending.cc)
explains why color Blend also writes its selected alpha after the independent extra operation.
Background alpha comes from its own extra-channel selector. Full-frame Replace implicitly
selects slot 0, so the original 40 streams use transparent background alpha. Eight extra
`_alpha_ref1` cases encode alpha Blend from saved slot 1; native header reads and Rust
inventory assertions verify the difference. The oracle applies these selectors before blending.

Native blend versus f64 differs by at most 5.97e-8; native CMS uncertainty remains separate.

The original normalized XYB reconstruction interval is propagated through the existing ICC
corner bounds and each monotone blend operation, with 2e-7 outward rounding for blend
arithmetic. Alpha remains exact. All 28 Add/MultiplyAdd and nontrivial Blend cases exclude blending in linear
RGB before conversion, even with these bounds. Three Multiply cases also discriminate that
order; Replace and near-homogeneous Multiply cases need not. Eight composed outputs exceed
one. The **624 files** include native/scalar physical layers, native/scalar composition and
all lower/upper bounds. Neither GPU output nor an observed GPU error defines any reference.

The public alpha test rejects all 48 cases through whole input and 1-/43-byte fragments before any frame is started. The previous diagnostic model changed the working color
domain from original-device reference reconstruction to direct linear presentation. An
image-owned selection registry shares identical ICC programs across these two uses; a byte
budget test leaves no room for a duplicate program and verifies cancellation releases once.

The old sequence calculations do not establish ICC XYB conformance. Broader legal alpha-policy/blend/crop/reference
combinations, transformed ICC targets after composition, numeric sequence bypasses, full ICC
methods/intents, and HDR/display policies remain part of the full JPEG XL goal.
