# jxl-wgpu

Portable, GPU-required JPEG XL encode/decode building blocks for Rust.

This is an independent Cargo workspace. Production codec execution requires a compatible GPU.
Published `jxl` and the reference `cjxl`/`djxl` tools are used only by development oracles and the
comparison harness.

## Crates

- `jxl_gpu_bitstream`: bounded raw/container parsing, bit IO, and deterministic `jxlc`/`jxlp`
  assembly shared by encode and decode.
- `jxl_gpu_protocol`: backend-neutral render plans, decoded-group packets, and the canonical
  `RenderBackend`/`FrameSession` contracts.
- `jxl_gpu_formats`: checked pitch-linear image layouts and CPU reference conversion, including
  RGB/BGR, luma, planar and semi-planar YCbCr, packed 4:2:2, high-bit-depth video, and the portable
  NVIDIA VPI 4.1 predefined format set.
- `jxl_wgpu`: the `WgpuBackend` implementation, WGSL kernels, bounded scheduling and reuse,
  GPU-resident output, explicit readback, and display textures.
- `jxl_wgpu_decode`: GPU-required codestream and animation sessions with synchronous and
  runtime-neutral async APIs.
- `jxl_wgpu_encode`: GPU-required encode jobs, group packet assembly, and runtime-neutral
  animation-session contracts.
- `jxl_gpu_harness`: correctness, capture/replay, latency, throughput, concurrency, animation, and
  CPU-readback measurements.

## Execution contract

Creating an encoder or decoder requires a compatible `wgpu` backend. Unsupported codestream
features or device limits return typed errors before a partial output becomes authoritative.

Host code still validates containers and headers, builds command buffers, orders group packets, and
assembles the final codestream. Pixel prediction, transform/quantization, coefficient or residual
processing, and supported entropy work belong to GPU jobs. The exact initially supported profile
is capability-negotiated; broader JPEG XL features remain typed rejections until their kernels and
conformance tests exist.

## Implemented codec slice

The checked-in interoperable slice is intentionally narrow and is not presented as a complete
JPEG XL implementation:

| Direction | Stock `wgpu` implementation | Current limits |
|---|---|---|
| Encode | Standard lossless Modular Gray8 codestream/container | one 2..=256 pixel-wide/high still, one group/pass, fixed Gradient predictor |
| Decode | GPU entropy/LZ77/residual/Gradient reconstruction from the actual `jxlc` payload | the same Gray8 profile in a container carrying a SHA-bound and prefix-verified `jwgp` acceleration index |
| Output | GPU-resident gray/luma and supported 8-bit planar/semi-planar YCbCr, including NV12 | unsupported color/bit-depth combinations return typed errors |
| Presentation | Same-queue buffer-to-RGBA display pipeline | no host synchronization before dependent queue work |
| CPU transport | Explicit mapped readback after GPU completion | transport only; it never selects a host codec |

The encoder's output is independently accepted and reproduced exactly by the published Rust `jxl`
decoder and by `djxl` when it is available in the test environment. `jwgp` is an auxiliary private
box; conforming decoders ignore it and decode the standard `jxlc`. The stock decoder currently
requires that validated index instead of duplicating the generic JPEG XL prefix-tree parser on the
host. Raw or unrelated JPEG XL inputs therefore reject rather than silently taking another path.

The public session traits already model blocking and runtime-neutral asynchronous animation,
bounded frame leases, timing, loop metadata, and reference slots. The stock codec backend still
rejects animation, RGB encode, multi-group/progressive Modular, VarDCT, ICC, patches, splines, and
other profiles until each has a GPU implementation and conformance coverage.

## Formats and display

The format model separates channel semantics, numeric representation, plane packing, subsampling,
chroma siting, color matrix, and range. CUDA-specific block-linear memory is out of scope because
it is not portable through WebGPU; pitch-linear formats are supported.

GPU outputs may be read back explicitly or passed directly to later work on the same `wgpu::Queue`.
`DisplayPipeline` converts supported buffers into an RGBA texture that can be sampled, rendered, or
copied to a surface without a host wait. Pipelines and buffers are reused across frames and decoder
instances within configured memory bounds.

Animation encode/decode exposes frame timing and loop metadata through both blocking and
runtime-neutral `Future`/poll APIs. GPU callbacks wake the future without depending on Tokio,
async-std, or a particular reactor. Bounded in-flight permits provide backpressure.

## Build and validate

```console
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo run -p jxl_gpu_harness -- verify --backend reference
cargo run -p jxl_gpu_harness -- codec --corpus tools/jxl_gpu_harness/codec-corpus.toml
```
