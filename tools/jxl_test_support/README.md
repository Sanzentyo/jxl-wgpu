# JPEG XL test support

This unpublished workspace crate owns helpers shared by codec integration tests and offline
fixture generators. The codecs select it only as a development dependency. Its CPU references
and native oracle processes are absent from the production codec dependency graph.

The module tree follows the source tree:

- `corpus` reads checked-in image inputs and comparison data.
- `fixtures` describes reproducible families and assembles their bounded codestream metadata.
- `oracles` provides independent CPU, scalar and offline native comparisons.
- `gpu` drives whole or fragmented test input and reads explicitly requested test output.
- `offline` writes fixture files and manages native generator processes and hexadecimal formats.

`corpus::jpeg_reconstruction` owns the 36 pinned JPEG/JXL identities used by both metadata and
GPU coefficient tests. The bitstream crate also imports this crate only as a development
dependency. `oracles::jpeg_coefficients` compiles
[`test-data/jpeg_coefficients.cpp`](test-data/jpeg_coefficients.cpp) with C++17 and
`pkg-config libjpeg`, requiring libjpeg-turbo 3.2.0. It reads coefficients directly from original
JPEGs, including sampling-aligned virtual-array padding verified against upstream `src/jdcoefct.c`.
Missing or mismatched native tools fail the test. No JPEG pixel decode or JXL implementation
supplies these coefficient references.

`oracles::icc_profile` compiles [`test-data/icc_profile.cpp`](test-data/icc_profile.cpp) with
C++17 and `pkg-config libjxl`, requiring libjxl 0.12.0 at build and runtime. The public encoder
creates tiny color-declaration inputs; the public decoder exports `JXL_COLOR_PROFILE_TARGET_ORIGINAL`
at its color event without requesting pixel decode. Tests compare every profile byte, including
the ICC ID, against the production metadata writer and unchanged official inputs. Missing tools,
version mismatches or native failure fail the test. No production dependency links this oracle.

`oracles::color` owns independent f64 transfer, CIE/Bradford and interval calculations shared by
original-color tests, resident ICC tests and the offline RGB-to-ICC exporter. Its jxl-oxide path
provides unbounded pre-OETF XYB stills; the codec's Gamma/DCI black floor makes inversion of an
already encoded original image unsuitable for that reference. These helpers use ordinary module
imports and stay outside production dependencies.

`oracles::vardct_matrices` loads four hash-checked native Hornuss/DCT2 matrix records, including
their exact binary16 wire parameters. The pinned libjxl parser supplies their expected values;
the older `jxl-vardct` parser omits the required ×64 scale for these two modes. Encoder and
decoder tests share this independent reference. Its schema and reproduction command are in the
[native VarDCT recipe](../../crates/jxl_wgpu/test-data/forward_vardct_generator/README.md).

`oracles::modular_integer` requests the declared original encoding or original embedded ICC
from jxl-oxide 0.12.6 and
copies its retained integer planes through jxl-render 0.12.4. It rejects F32 planes rather than
rounding them back to integers, so exact 17–31-bit encoder checks include the low sample bits.
The [wide-integer encoder matrix](../../docs/CONFORMANCE_CORPUS.md#wide-integer-modular-encoding)
also requires the libjxl 0.12.0 original-component oracle for independent normalized output.
The same retained integer working planes expose raw IEEE binary16/binary32 words before
conversion to F32. The [floating encoder matrix](../../docs/CONFORMANCE_CORPUS.md#ieee-floating-point-modular-encoding)
compares those words exactly, including NaN payloads and signed zero, and independently checks
libjxl's original F32 output. Finite arithmetic composition uses the Rust `jxl` F32 oracle in
addition to libjxl; the older jxl-oxide header reader does not implement the per-channel
reference-field rule for mixed full-frame blend modes.
The [RCT encoder matrix](../../docs/CONFORMANCE_CORPUS.md#lossless-modular-rct-selection) also uses
jxl-oxide's frame/TOC reader and the independent `jxl-bitstream` U32 decoder to inspect every
separate pass-group header. It verifies the requested local transform number, MA-tree choice and default/custom Weighted coefficients
without consulting the production metadata parser or deriving expected source words from an inverse.
The extra-channel native oracle's `--original-icc` mode requires libjxl 0.12.0 and exact identity
between the original and actual DATA ICC bytes before returning original device samples. Original
profile streams use their native default, because libjxl rejects an explicit ICC request for them;
XYB streams explicitly request the original profile. This never substitutes a generated profile
or silently accepts a different output encoding. The encoder ICC matrix separately exports the
original profile through `oracles::icc_profile`, checks raw integer/IEEE words, and compares requested
color output with the unchanged independent scalar corpus and native-encoded fixture.
Neither oracle is a production dependency or a fallback.

`fixtures::icc_spots` reads a typed manifest for native ICC/enumerated RGB/Gray sources, both
codecs and original/XYB stills or reference sequences. Behavior follows the manifest fields and
is checked against the decoded inventory. Independent interval/CMM reference generation lives
with the [spot corpus recipe](../../crates/jxl_wgpu_decode/test-data/icc_spots_generator/README.md).

`native/icc` owns the shared C++ f64 ICC curve and CIE/Bradford reference equations used by the
resident-ICC and embedded-JPEG-XL generators. Both compile with `-Itools/jxl_test_support/native`
and include the named `icc` headers. Cargo and production decoding never compile or link this
offline oracle; its pinned native dependency and reproduction commands belong to each corpus.

`fixtures::frame_features` owns common frame headers, bounded prefix assembly and feature entropy
writing. `fixtures::patches` and `fixtures::splines` define their respective scenarios using that
shared contract. A frame explicitly declares its patch and spline programs; assembly preserves
their codestream order without reaching into another test target or example's private files.
`fixtures::patch_references` has ordinary `jpeg`, `mixed` and `modular_ycbcr` submodules. Each
case declares its family, source noise operation, oracle, comparison controls and precision metric.
Filename prefixes and suffixes do not select test behavior or oracle exceptions.

Each decoder integration target has a `tests/<target>/main.rs` entry point. Its private modules
live below that directory and use ordinary `mod` declarations. Examples with private helpers
follow the same `examples/<target>/main.rs` layout. Shared helpers are imported from this crate;
source files are never included through `#[path]` or compiled repeatedly under different names.
Existing `cargo test --test <target>` and `cargo run --example <target>` commands stay the same.

The workspace denies `dead_code` and `unused`. Remove obsolete helpers and imports instead of
silencing the diagnostics. Platform or ownership exceptions belong on the smallest applicable
item as `expect(..., reason = "...")`; a stale expectation must produce a warning. A resource
retained until GPU completion can be necessary even when it is never read explicitly. Such
ownership should be expressed in its containing type and explained at that boundary.

`fixtures::original_color::intents` rewrites only the final intent enum of explicit RGB/Gray
metadata. It checks unchanged image fields and physical frame bytes, and an independent header
reader verifies the new intent. The extra-channel native oracle's `--original` mode requires
libjxl 0.12.0, requests the declared encoding and verifies the actual output color fields. Tests
require this oracle; missing tools or version mismatches fail. Existing fixtures and references
are read without replacement. See the [intent recipe](../../crates/jxl_wgpu_decode/test-data/original_color_generator/README.md#original-intent-variants).

## Scalar progressive oracle

`oracles::progressive::scalar_linear_updates` requires libjxl 0.12.0 compiled entirely with
`HWY_COMPILE_ONLY_SCALAR`. Its inverse quantizer uses ordinary division rather than SIMD
reciprocal estimates. The [progressive encoder corpus](../../docs/CONFORMANCE_CORPUS.md#progressive-vardct-encoding)
records a raw-matrix image where that distinction exceeds the existing final linear-F32 bound.
The scalar reference keeps that bound and all inputs; normal native SIMD still checks every
pass's RGB8 precision independently. Other `native_updates*` entry points retain their existing
system-library behavior.

`scalar_linear_prefix_updates` uses the same pinned executable to flush incomplete input at
complete AC-pass boundaries. It requests the helper's existing `prefix` mode; coefficient
decoding and color arithmetic are unchanged. The encoder tests compare the flushed pixels
and intended resolution with the corresponding whole-input pass, including permuted TOCs.

Reuse the existing pinned libjxl checkout and scalar native build, when available. The existing
Modular YCbCr CMake project supplies the shared scalar library and now also builds
`decode_progressive_scalar` from the unchanged pass-decoding algorithm in `decode_progressive.c`.
No fixture regeneration or extra Cargo target directory is required:

```sh
JXL_NATIVE_SOURCE=/absolute/path/to/pinned/libjxl
JXL_SCALAR_BUILD=/absolute/path/to/existing/scalar-build
cmake -S crates/jxl_wgpu_decode/test-data/modular_ycbcr_generator \
  -B "$JXL_SCALAR_BUILD" -DCMAKE_BUILD_TYPE=Release -DJXL_SCALAR_ONLY=ON \
  -DJXL_SOURCE="$JXL_NATIVE_SOURCE"
cmake --build "$JXL_SCALAR_BUILD" --target decode_progressive_scalar --parallel 4
export JXL_PROGRESSIVE_SCALAR_ORACLE="$JXL_SCALAR_BUILD/decode_progressive_scalar"
cargo test --locked -p jxl_wgpu_encode --lib vardct_encoder::tests::progressive -- --test-threads=2
```

CMake rejects revisions other than `a7a9c787341cf703dede03c2009fa460cae5e5df` and applies the
scalar definition to libjxl and its helper together. The helper checks build/runtime version
0.12.0; Rust verifies its scalar identity before using it. Missing or mismatched helpers fail
instead of selecting another reference or skipping. This is development-only CPU oracle work.

Run the workspace checks from the repository root with Rust 1.98 or later, keeping that oracle
environment available:

```sh
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features --no-fail-fast -- --test-threads=2
```

GPU tests use exactly two libtest threads, with separate test executables run sequentially.
Fixture generators use the same manifests and helpers as the tests;
their commands, pinned oracle versions and observed errors are recorded in
[`CONFORMANCE_CORPUS.md`](../../docs/CONFORMANCE_CORPUS.md).
