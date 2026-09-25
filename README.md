# jxl-wgpu

Portable, GPU-required JPEG XL encode/decode building blocks for Rust.

## Project phase

This is a maintainer-directed implementation and conformance-development repository,
not a complete or production-ready JPEG XL codec. APIs and supported feature
combinations are still evolving. **Third-party contributions are not being accepted
at this stage.** Development notes describe the maintainer's work and authorized
agent tasks, not a public contribution process.

The immediate work is completing the GPU codec paths and their correctness,
resource-lifetime, and interoperability evidence. Encoder syntax coverage and
production encoder quality are separate goals in the
[full JPEG XL roadmap](docs/FULL_JPEG_XL_ROADMAP.md).

In the technical documentation, “production path” distinguishes the real library
codec from test oracles; it does not claim production readiness. That path requires
a compatible `wgpu` adapter and has no CPU image-codec fallback. Unsupported
features and device limits produce typed errors.

## Build and validate

Use **Rust 1.98 or later**, as declared in [Cargo.toml](Cargo.toml), and a compatible
GPU for codec execution. From a checkout of this repository:

```console
cargo run --locked -p jxl_gpu_harness -- adapters
cargo run --locked -p jxl_gpu_harness -- codec fixtures/gpu_gray8_lossless.jxl \
  --format u8 --output-target cpu-readback
```

The second command decodes a checked-in fixture on the GPU and explicitly reads
back its output. `cpu-readback` is transport, not CPU decoding. Adapter enumeration
alone does not establish codec compatibility.

The [decoder](crates/jxl_wgpu_decode/README.md) and
[encoder](crates/jxl_wgpu_encode/README.md) document the library APIs. The
[harness guide](tools/jxl_gpu_harness/README.md) describes the CLI's own supported
workloads; it is not a capability specification for every library API.
[Internal development notes](docs/DEVELOPMENT.md) contain validation commands and
evidence gates. GitHub Actions is [intentionally disabled](.github/workflows/README.md).

## Implemented codec slice

The [roadmap](docs/FULL_JPEG_XL_ROADMAP.md) owns supported variants, remaining work,
and the evidence required to mark a feature complete. This is only an overview;
individual kernels, fixtures, or passing tests do not establish full conformance.

| Area | Available building blocks | Important limits |
|---|---|---|
| Decode | A common GPU frontend for Modular and VarDCT; stills, animation/composition, embedded previews, and validated progressive output in supported paths. | Both coding modes and their feature combinations remain partially implemented or incompletely covered. Progressive output does not imply incomplete-main-input decoding. |
| Encode | Lossless Modular Gray/GrayAlpha/RGB/RGBA with 1–31-bit integers or all 154 legal floating precisions (2–8 exponent and 2–23 fraction bits) in packed, planar or split buffers, including RGB/BGR order, declared alpha association, enumerated or embedded RGB/Gray ICC source color, selectable 128/256/512/1024 groups, all 42 source and ordered local RCT types, exact local GPU color, delta, mixed or implicit Palette on selected contiguous components, explicit per-group Squeeze with named separable policies or up to 296 ordered current-channel steps, tail/in-place residual placement, ordered local RCT/Squeeze programs through 273 header entries, planned GPU intermediates and checked signed-word residuals, all 14 explicit predictors with custom Weighted coefficients, general GPU LZ77 matching, default Prefix or GPU ANS serialization with bounded histogram clustering and adaptive residual/distance/length hybrid integers and 64/128/256-symbol alphabets, and supported animation; experimental GPU VarDCT with all 27 transform strategies, all 13 caller-selected coefficient-order families, parametric/raw quantization matrices for all 17 matrix families, and 1–11 spectral/quantized AC passes with resolution stopping points and raster/center/explicit or GPU local-contrast group order, plus 1–31-bit integer or all 154 legal floating Gray/GrayAlpha/RGB/RGBA precisions in packed/planar/split buffers with shared checked addressing for stills and animation in XYB or original components with enumerated SDR/HDR or embedded RGB/Gray ICC color, all four intents and exact image white, with timed crops, all five blend modes, hidden frames and four post-color-transform references. VarDCT also accepts independently sized lossless scalar extras with separate precision, metadata and blend-alpha selection. Both encoders emit timed sequences and layered stills with hidden regular/reference-only frames through shared control planning. Mixed sequences explicitly select Modular or VarDCT per physical frame under one checked integer/floating Gray/GrayAlpha/RGB/RGBA original-component color contract, with cross-codec references and indexed presentation seeking. | General/global transform stacks and adaptive component/transform choices, CMYK, independent extras for Modular/mixed encoding, YUV and texture inputs remain; VarDCT rejects nonfinite color samples, nonfinite ICC conversion and forbidden XYB/ICC post-transform references, and still lacks general perceptual-quality guarantees, adaptive rate control, DC progressive encoding and broad perceptual saliency evidence. |
| Transport and metadata | Bounded raw/`jxlc`/`jxlp` scanning, fragmented input, opaque metadata retention/writing, plain `jxli` generation from assembled encoder frames with shared header/dependency validation and GPU seeking from contiguous or incrementally received input, bounded `jbrd` metadata parsing/emission, and validated GPU original-JPEG byte reconstruction for a qualified still profile. | Seeking requires complete input; full container policy, byte-range seeking and broader JPEG reconstruction variants remain incomplete. |
| Color and rendering | GPU restoration, resampling, composition, enumerated SDR/HDR and supported ICC connections; all four original RGB/Gray intents with standard/custom white points; explicit tone/gamut mapping and still gain-map reconstruction with enumerated or ICC output. | Profile, rendering, gain-map, and cross-feature conformance are not complete. |
| Output and scheduling | GPU-resident pitch-linear buffers, explicit readback, display textures, runtime-neutral async APIs, and budgeted resource leases. | Output support depends on the codec path and format. Host-thread concurrency is not coalesced codec GPU batching. |

The [official conformance corpus](crates/jxl_wgpu_decode/test-data/official_conformance/README.md)
checks the published per-channel pixel bounds of all 40 descriptors across 27 unique inputs
in its pinned upstream revision. It includes three animations, signed F32 samples, both alpha
associations, noise, splines, patches, original ICC components and a 4064×2704 progressive image.
A separate GPU byte-output target reproduces all three published original JPEGs exactly.
Bounded original ICC export preserves embedded bytes and generates enumerated RGB/Gray/XYB
profiles, with exact official-object and native comparisons. Broader conformance remains open.

## Execution contract

Image-domain prediction, transforms, coefficient/residual processing, filtering,
color conversion, and supported entropy jobs execute on the GPU. Bounded host
parsing, scheduling, validation, deterministic bit writing, and container assembly
are allowed. CPU image codecs and native oracle tools stay in development support;
see the [upstream boundary](docs/UPSTREAM_BOUNDARY.md).

Standard decoding does not require private acceleration metadata. The optional
single-group `jwgp` box is not a substitute for the ordinary JPEG XL codestream.
The [JPEG reconstruction API](crates/jxl_wgpu_decode/README.md#gpu-original-jpeg-byte-output)
restores quantizers and coefficients from actual GPU JXL decoding, generates sequential or
progressive JPEG entropy on the GPU, and publishes a validated, budget-owned byte lease.
All 36 pinned original JPEGs agree byte for byte, including the three official sources.
The bounded still DCT8 profile remains Partial; broader legal variants and GPU JPEG ingestion
remain required.

`GpuDecoder::open` and `stream(...).finish()` require complete input. A complete
embedded preview can be taken earlier with `take_preview`, while the same stream
continues receiving the main image. Opt-in progressive updates refine a presentation;
only its final update advances animation time, and `next_frame` remains final-only.
Preview/main selection and incremental-input ownership are specified in the
[decoder guide](crates/jxl_wgpu_decode/README.md#executable-profile); pass and LF
updates have their own [completion contract](crates/jxl_wgpu_decode/README.md#intermediate-lf-and-pass-images).

[Bounded GPU seeking](docs/FRAME_SEEKING.md) validates index offsets and timing against real headers,
restores required reference versions, and exposes the target with its original metadata. Skipped
entropy is not claimed validated. The API does not yet acquire input by byte range.

Output is authoritative only after the applicable codec validation succeeds.
Explicit unvalidated handoff is separate: completion of downstream display or
readback does not validate the codec result, and derived results must be discarded
if codec validation fails. Accounted leases retain resources through submitted
work, cancellation, and retained output. Raw `wgpu` handle clones do not retain
those accounting guarantees; custom submissions also obey the backend's submission
guard contract. See [backend ownership](crates/jxl_wgpu/README.md#render-plan-execution)
and [same-queue submission](crates/jxl_wgpu/README.md#same-queue-display).

## Formats and display

Sample semantics, numeric representation, packing, color encoding, and subsampling
are separate. Portable pitch-linear layouts are in scope; CUDA-specific block-linear
surfaces are not. Source bit depth alone does not promise lossless precision after
filtering, composition, or lossy reconstruction. See
[format coverage](docs/VPI_FORMAT_COVERAGE.md),
[format APIs](crates/jxl_gpu_formats/README.md), and the decoder's
[integer precision contract](crates/jxl_wgpu_decode/README.md#integer-source-samples).

Numeric buffers are not implicitly color images. The backend's explicit numeric
display APIs require a `NumericDisplayContract`. Color display produces linear
BT.709 textures; wide-gamut/HDR and F32 input use `Rgba16Float` to preserve extended
values. Decoder/output tone and gamut mapping are explicit requests, not automatic
monitor or surface negotiation. See the
[display contract](crates/jxl_wgpu/README.md#same-queue-display),
[tone mapping](docs/TONE_MAPPING.md), and [gamut mapping](docs/GAMUT_MAPPING.md).

GPU-resident output, same-queue display, and explicit readback are different paths.
A queued display conversion is not an end-to-end presentation measurement. Use the
[benchmark methodology](docs/GPU_BENCHMARKS.md) when interpreting performance results.

## Crates

| Component | Responsibility |
|---|---|
| [jxl_gpu_bitstream](crates/jxl_gpu_bitstream) | Bounded transport/header parsing, metadata, bit I/O, and container assembly. |
| [jxl_gpu_protocol](crates/jxl_gpu_protocol) | Backend-neutral plans, packets, and backend/session contracts. |
| [jxl_gpu_formats](crates/jxl_gpu_formats) | Checked image layouts and reference format conversion. |
| [jxl_wgpu](crates/jxl_wgpu) | `WgpuBackend`, WGSL kernels, resource accounting, readback, and display. |
| [jxl_wgpu_decode](crates/jxl_wgpu_decode) | GPU-required decoding and animation sessions. |
| [jxl_wgpu_encode](crates/jxl_wgpu_encode) | GPU-required encoding and animation assembly. |
| [jxl_gpu_harness](tools/jxl_gpu_harness) | Correctness, capture/replay, codec workloads, and measured evidence. |
| [jxl_test_support](tools/jxl_test_support) | Unpublished fixture, GPU-test, and offline-oracle support. |

## Documentation

Use the [topic index](docs/README.md) for API and implementation references,
[internal development notes](docs/DEVELOPMENT.md) for validation, and
[AGENTS.md](AGENTS.md) for repository-specific agent boundaries. Detailed capability
status and corpus results belong in their owning documents, not in this overview.

## License

[BSD-3-Clause](LICENSE). See [THIRD_PARTY.md](THIRD_PARTY.md) for third-party notices.
