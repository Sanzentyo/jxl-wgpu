# jxl-wgpu

Portable, GPU-required JPEG XL encode/decode building blocks for Rust.

This is an independent Cargo workspace. Production codec execution requires a compatible GPU.
Published `jxl` and the reference `djxl` tool are development-only interoperability oracles and are
not production dependencies or fallback paths.

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
- `jxl_gpu_harness`: correctness, capture/replay, sequential/concurrent timing, output-path, and
  CPU-readback evidence with explicit submission, wait, logical-byte, and staging-byte counters.
  Host-thread fan-out is labelled separately from unimplemented coalesced GPU batching; the stock
  codec's animation boundary remains a typed unsupported result.

## Execution contract

Creating an encoder or decoder requires a compatible `wgpu` backend. Unsupported codestream
features or device limits return typed errors before a partial output becomes authoritative.

Host code still validates containers and headers, builds command buffers, orders group packets, and
assembles the final codestream. Pixel prediction, transform/quantization, coefficient or residual
processing, and supported entropy work belong to GPU jobs. The exact initially supported profile
is capability-negotiated; broader JPEG XL features remain typed rejections until their kernels and
conformance tests exist.

Concurrent encode, decode, and explicit readback work uses byte-weighted, non-blocking memory
admission. The same completion values work with native blocking calls or any async executor.
Decoder output buffers carry cloneable memory leases, so dropping a session cannot free its budget
while a tracked lease is still retained. GPU frame/output containers are intentionally not
cloneable; raw wgpu handles cloned through the explicit interop borrow are outside that accounting.

## Implemented codec slice

The checked-in interoperable slice is intentionally narrow and is not presented as a complete
JPEG XL implementation:

| Direction | Stock `wgpu` implementation | Current limits |
|---|---|---|
| Encode | Standard lossless Modular Gray8 codestream/container | one 2..=256 pixel-wide/high still, one group/pass, fixed Gradient predictor |
| Decode | GPU entropy/LZ77/residual/Gradient reconstruction from the actual `jxlc` payload | the same Gray8 profile in a container carrying a SHA-bound and prefix-verified `jwgp` acceleration index |
| Output | GPU-resident Gray8 written into all 30 portable VPI pitch-linear formats: 20 color layouts and 10 explicitly mapped numeric layouts | numeric normalization is mandatory; F64 requires an explicit native-or-compatibility precision policy |
| Presentation | Same-queue buffer-to-linear-BT.709 RGBA display pipeline | explicit unvalidated handoff can enqueue display/readback/custom GPU work before the 16-byte validation map completes; derived results are discarded if validation later fails |
| CPU transport | Explicit mapped readback after GPU completion | transport only; it never selects a host codec |

The encoder's output is independently accepted and reproduced exactly by the published Rust `jxl`
decoder and by `djxl` when it is available in the test environment. `jwgp` is an auxiliary private
box; conforming decoders ignore it and decode the standard `jxlc`. The stock decoder currently
requires that validated index instead of duplicating the generic JPEG XL prefix-tree parser on the
host. Raw or unrelated JPEG XL inputs therefore reject rather than silently taking another path.

The public decode session traits separate queue submission from completion, prefetch an ordered
bounded frame window, and expose native blocking plus runtime-neutral asynchronous completion.
Frame leases, timing, timecodes, loop metadata, and reference slots remain explicit. The stock
codec backend still rejects animation, RGB encode, multi-group/progressive Modular, VarDCT, ICC,
patches, splines, and other profiles until each has a GPU implementation and conformance coverage.

## Formats and display

The format model separates channel semantics, numeric representation, plane packing, subsampling,
chroma siting, color matrix, and range. CUDA-specific block-linear memory is out of scope because
it is not portable through WebGPU; pitch-linear formats are supported.

GPU outputs may be read back explicitly or passed directly to later work on the same `wgpu::Queue`.
The stock pending frame can expose a distinct `UnvalidatedGpuImageFrame`; its permit-bearing buffer
leases can be consumed immediately while frame metadata and changed regions remain withheld until
validation. `DisplayPipeline` converts supported buffers into an explicit linear-light BT.709 RGBA
texture that can be sampled, rendered, or copied to a surface without an additional host wait.
Unsupported primaries and HDR transfer functions are rejected instead of being mislabeled.

The ten non-color numeric VPI layouts remain GPU-buffer/readback outputs rather than implicitly
colorized display images. They carry no color meaning, so `DisplayPipeline` returns a typed error
instead of inventing a range, component selection, or transfer function. Applications can enqueue
an explicit visualization shader on the same queue through `GpuBufferLease::as_wgpu_buffer()` and
the checked `ImageLayout`.

The animation session contracts expose frame timing and loop metadata through both blocking and
runtime-neutral `Future`/poll APIs. Prefetch submits multiple frames without a host wait; the
ordered pending queue then completes its front through a native wait or a task waker, without
depending on Tokio, async-std, or a particular reactor. The stock codec slice is still-only and
returns a typed unsupported error for animation codestreams.

## Build and validate

```console
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo run -p jxl_gpu_harness -- verify --backend reference
cargo run -p jxl_gpu_harness -- codec fixtures/gpu_gray8_lossless.jxl \
  --format u8 --output-target cpu-readback
```
