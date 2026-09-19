# jxl-wgpu

Portable, GPU-required JPEG XL encode/decode building blocks for Rust.

**Work in progress: this is not a complete JPEG XL implementation.** Production
codec execution requires a compatible `wgpu` adapter. Unsupported features and
device limits produce typed errors; there is no CPU image-codec fallback.

## Build and validate

Use **Rust 1.98 or later**, as declared in [Cargo.toml](Cargo.toml), and a
compatible GPU for codec execution. From a checkout of this repository:

```console
cargo run --locked -p jxl_gpu_harness -- adapters
cargo run --locked -p jxl_gpu_harness -- codec fixtures/gpu_gray8_lossless.jxl \
  --format u8 --output-target cpu-readback
```

The second command decodes a checked-in fixture on the GPU and explicitly reads
back its output. `cpu-readback` is a transport choice, not CPU decoding.
Adapter enumeration alone does not establish codec compatibility.

For library integration, start with the [decoder](crates/jxl_wgpu_decode/README.md)
or [encoder](crates/jxl_wgpu_encode/README.md) API examples. The
[harness guide](tools/jxl_gpu_harness/README.md) covers other commands and their
measurement contracts. See [CONTRIBUTING.md](CONTRIBUTING.md) for focused checks,
reference-only validation, and capability-change gates. GitHub Actions is
[intentionally disabled](.github/workflows/README.md).

## Implemented codec slice

The [full JPEG XL roadmap](docs/FULL_JPEG_XL_ROADMAP.md) is authoritative for
supported variants, remaining work, and the evidence required to mark a feature
complete. This table is an overview, not a conformance claim.

| Area | Available building blocks | Important limits |
|---|---|---|
| Decode | A common GPU frontend for Modular and VarDCT; stills, animation/composition, embedded previews, and validated progressive output in supported paths. | Both coding modes and their feature combinations remain partially implemented or incompletely covered. |
| Encode | Lossless Modular Gray/RGB/RGBA at 1–16-bit integer depth, including supported animation; experimental GPU VarDCT with all 27 transform strategies. | VarDCT does not provide general perceptual-quality guarantees, adaptive rate control, or progressive encoding. |
| Transport and metadata | Bounded raw/`jxlc`/`jxlp` scanning, fragmented input, explicit opaque metadata retention and writing. | Full container policy, frame indexes, and JPEG bitstream reconstruction remain incomplete. |
| Color and rendering | GPU restoration, resampling, composition, enumerated SDR/HDR and supported ICC connections; explicit tone/gamut mapping and still gain-map reconstruction. | Profile, rendering, gain-map, and cross-feature conformance are not complete. |
| Output and scheduling | GPU-resident pitch-linear buffers, explicit readback, display textures, runtime-neutral async APIs, and budgeted resource leases. | Output support depends on the selected codec path and format. Host-thread concurrency is not coalesced codec GPU batching. |

## Execution contract

Image-domain prediction, transforms, coefficient/residual processing, filtering,
color conversion, and supported entropy jobs execute on the GPU. Bounded host
parsing, scheduling, validation, deterministic bit writing, and container assembly
are allowed. Published CPU codecs and native tools are development-only oracles,
not production dependencies or fallback paths.

Output becomes authoritative only after validation. Explicit unvalidated GPU
handoff is a separate contract, and downstream results must be discarded if
validation fails. Memory leases keep accounted resources alive through submitted
work, cancellation, and retained output. See [GPU architecture](docs/GPU_ARCHITECTURE.md)
and the [upstream boundary](docs/UPSTREAM_BOUNDARY.md) when changing these contracts.

## Formats and display

The format model separates sample semantics, numeric representation, plane packing,
color encoding, and subsampling. Portable pitch-linear layouts are in scope;
CUDA-specific block-linear surfaces are not. Numeric output is not implicitly a
color image. See [format coverage](docs/VPI_FORMAT_COVERAGE.md) and
[format APIs](crates/jxl_gpu_formats/README.md) for path-specific support and
precision policies.

Same-queue display conversion and explicit readback are different output paths;
neither implies end-to-end presentation timing. See the
[benchmark guide](docs/GPU_BENCHMARKS.md) before interpreting performance results.

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

Use the [documentation index](docs/README.md) to select a topic,
[CONTRIBUTING.md](CONTRIBUTING.md) for development and validation, and
[AGENTS.md](AGENTS.md) for repository-specific agent guidance.

The previous [detailed README](README.legacy.md) is preserved as a historical
snapshot. Its incremental status statements and testing instructions are not the
current capability or contribution policy; use the roadmap and contributor guide.

## License

[BSD-3-Clause](LICENSE). See [THIRD_PARTY.md](THIRD_PARTY.md) for third-party notices.
