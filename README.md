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
  animation-session contracts for lossless Modular plus an experimental GPU VarDCT profile.
- `jxl_gpu_harness`: correctness, capture/replay, sequential/concurrent timing, output-path, and
  CPU-readback evidence with explicit submission, wait, logical-byte, and staging-byte counters.
  Host-thread fan-out is labelled separately from coalesced GPU batching.

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

The checked-in paths are interoperable but are not yet a complete JPEG XL implementation:

| Direction | Stock `wgpu` implementation | Current limits |
|---|---|---|
| Encode | Standard lossless Modular Gray/RGB/RGBA at every integer depth from 1 through 16, multi-group stills, crops/references/blending animation, plus an experimental all-27-strategy VarDCT RGB8 still profile. VarDCT accepts validated exact-binary16 LF dequantization and LF/HF chroma-correlation metadata; tiled DCT8 emits multiple LF and AC groups and accepts checked axes through 16K. | Modular uses one pass and the implemented predictor/entropy set; VarDCT remains a fixed distance-25 LF-only profile that quantizes every AC coefficient to zero and does not yet select mixed strategies over an image. |
| Decode | One public `GpuDecoder::wgpu` reads the frame coding mode and routes one-pass lossless Modular stills or the authoritative bounded VarDCT path without caller mode knowledge. Modular uses GPU Prefix/ANS entropy, LZ77, MA prediction, residual reconstruction, reversible color transform, and requested output conversion. A channel-fixed Gradient group may resume through multiple caller-, budget-, and device-bounded overlapping stream uploads while its cursor, ANS, LZ77, decoded-count, and failure state remain GPU-resident. VarDCT decodes mixed maps containing all 27 transform strategies, single-pass nonzero AC, all 13 natural/custom coefficient-order families, stream-defined HF block contexts, non-default LF dequantization and LF/HF chroma correlation, scanline or entropy-permuted center-first pass groups, every normative default strategy matrix with all 3-bit X/B scale values, multiple LF groups, resident Gaborish, and one-to-three-iteration EPF restoration. Ordinary multi-LF-group `cjxl` local MA trees run through the authoritative runtime-neutral frame engine as two GPU submissions and two aggregate status maps: the host uses LF end cursors only to pack HF metadata and never decodes image entropy. Generated custom-LF and 2056×256 local-tree codestreams are checked on an actual adapter through blocking and runtime-neutral async completion against Rust `jxl` and optional `djxl` within one RGB8 code. | The selector and both engines share one backend byte budget, but generic MA and VarDCT entropy consumers do not yet resume inside one group stream, and Modular still lacks Palette/Squeeze/multiple passes. VarDCT remains limited to one spectral pass, default strategy matrices, RGB8 output, and no composition features. |
| Output | GPU-resident native integer Gray/RGB/RGBA plus all 30 portable VPI pitch-linear formats: 20 color layouts and 10 explicitly mapped numeric layouts | numeric normalization is explicit; F64 requires a native-or-exact-widening precision policy |
| Presentation | Same-queue buffer-to-linear-BT.709 RGBA display pipeline | explicit unvalidated handoff can enqueue display/readback/custom GPU work before final validation; a staged local-tree frame exposes it only after the host-dependent HF submission is queued, and derived results are discarded if validation later fails |
| CPU transport | Explicit mapped readback after GPU completion | transport only; it never selects a host codec |

Lossless encoder output is independently accepted and reproduced exactly by the published Rust
`jxl` decoder and by `djxl` when it is available in the test environment. `jwgp` is an optional
single-group acceleration box; conforming decoders, including this workspace's generic standard
path, ignore it and decode the standard `jxlc`. The VarDCT encoder is likewise checked with both
oracles, including explicit LF metadata and horizontal and vertical two-LF-group images; its output
also round-trips through the stock GPU decoder. Actual GPU tests cover 16K×1 and 1×16K tiled
panoramas.

The public decode session traits separate queue submission from completion, prefetch an ordered
bounded frame window, and expose native blocking plus runtime-neutral asynchronous completion.
Frame leases, timing, timecodes, loop metadata, and reference slots remain explicit. The encoder
implements standard Modular animation. Decoder animation, progressive Modular, non-alpha extra
channels, complete VarDCT reconstruction, arbitrary ICC transforms, patches, splines, and noise
remain typed rejections until their GPU paths and conformance coverage land.

[`docs/FULL_JPEG_XL_ROADMAP.md`](docs/FULL_JPEG_XL_ROADMAP.md) is the canonical capability table,
full-format implementation backlog, dependency order, and acceptance contract. Capability-changing
commits must update it together with this summary and the affected crate documentation.

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
