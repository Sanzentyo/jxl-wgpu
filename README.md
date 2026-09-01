# jxl-wgpu

Portable, GPU-required JPEG XL encode/decode building blocks for Rust.

This is an independent Cargo workspace. Production codec execution requires a compatible GPU.
Published `jxl` and the reference `djxl` tool are development-only interoperability oracles and are
not production dependencies or fallback paths.

## Crates

- `jxl_gpu_bitstream`: bounded raw/container parsing, non-accumulating incremental transport
  events, bit IO, and deterministic `jxlc`/`jxlp` assembly shared by encode and decode.
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

The incremental transport scanner accepts arbitrary shared chunks for raw, `jxlc`, ordered v0 and
out-of-order v1 `jxlp` delivery. Apart from the inline reconstructed two-byte codestream signature,
ordered codestream and auxiliary-box payloads remain zero-copy `Arc` slices; only future fragments
waiting on a v1 gap use payload-only storage bounded by explicitly reported logical bytes. Its
terminal event validates transport end-of-input. A second bounded scanner incrementally parses the
image header and each frame header/TOC, emits frame inventories before their physical sections, and
routes ordered section ranges without retaining the whole codestream. `GpuDecoder::stream` consumes
those borrowed transport events, builds one checked logical span table without joining it, and
hands the same inventory/source pair to the stock coding-mode selector used by `open`. Its shared,
non-blocking incremental-input budget admits a `CodestreamChunk` before scanner state changes, so a
rejected event is retryable across concurrent streams. `GpuDecoder::container_stream_limits`
provides matching hard limits for the caller's transport scanner. The one growable ownership
permit follows
the source into the selected engine: Modular releases it after submission, staged local-tree
VarDCT retains it through cursor-dependent HF submission, and cancellation releases it immediately.
Both engines copy bounded GPU upload ranges across physical chunk boundaries; VarDCT also
initializes its temporary whole-codestream GPU buffer directly from those spans without a second
host-sized `Vec`. All Modular and VarDCT scalar metadata bit parsing is span-native, including
VarDCT block-context maps, custom coefficient-order permutations, MA descriptors, and
cursor-dependent local-HF headers. Inventory also resolves each `USE_LF_FRAME` read to its exact
earlier progressive-DC producer across the four normative LF slots. Missing producers are rejected
before submission. The stock decoder executes recursive progressive-DC chains without pixel
readback. It converts the Modular root's signed `[Y, X, B-Y]` planes to dequantized XYB, packs them
into each dependent VarDCT LF atlas, decodes a single-entry intermediate frame's HF metadata on
GPU, maps only its validated HF-global cursor, then submits its general HF-global/AC and the next
dependency on the same queue. Parametric custom dequantization matrices are expanded as bounded
scalar metadata and installed directly in the resident resource table. Raw mode-7 matrices now use
the common GPU Modular entropy and inverse-transform pipelines, validate one mapped status, and
overlay the three decoded channels into every aliased resident strategy-matrix target before AC.
The checked-in cjpeg-to-cjxl stream executes this primitive on an actual adapter; local-tree raw
matrix conformance and non-XYB JPEG reconstruction remain gaps. Side-image entropy binds only its
four-byte-aligned HF-global packet window rather than the whole codestream.

Concurrent encode, decode, and explicit readback work uses byte-weighted, non-blocking memory
admission. The same completion values work with native blocking calls or any async executor.
Decoder output buffers carry cloneable memory leases, so dropping a session cannot free its budget
while a tracked lease is still retained. GPU frame/output containers are intentionally not
cloneable; raw wgpu handles cloned through the explicit interop borrow are outside that accounting.

## Implemented codec slice

The checked-in paths are interoperable but are not yet a complete JPEG XL implementation:

| Direction | Stock `wgpu` implementation | Current limits |
|---|---|---|
| Encode | Standard lossless Modular Gray/RGB/RGBA at every integer depth from 1 through 16, multi-group stills with caller-selected shared-global or complete local-per-group MA/entropy descriptors, crops/references/blending animation, plus an experimental all-27-strategy VarDCT RGB8 still profile. VarDCT accepts validated exact-binary16 LF dequantization and LF/HF chroma-correlation metadata; the bounded DCT8 path transforms, quantizes, and serializes real AC coefficients on the GPU, while tiled DCT8 emits multiple LF and AC groups and accepts checked axes through 16K. | Modular uses one pass and the implemented predictor/entropy set; local mode currently repeats the frame-trained configuration in each pass group rather than training independent trees. VarDCT remains fixed at distance 25. Its bounded DCT8 AC policy uses natural order, one prefix cluster for all 495 coefficient contexts, no LZ77, and one pass. Scalable/tiled and non-DCT8 encoding remain zero-AC, with no mixed strategy selection or rate control. |
| Decode | One public `GpuDecoder::wgpu` routes standard lossless Modular or bounded VarDCT without caller mode knowledge. Modular keeps Prefix/ANS entropy, LZ77, every accepted MA predictor, RCT/Palette/Squeeze inversion, requested output conversion, and bounded resume on GPU. It supports 128/256/512/1024-pixel groups and one through three passes. Channels with both transformed shifts at least three execute through LF-group streams; the header's downsampling brackets assign every remaining channel to exactly one pass, empty sections are zero-validated without dispatch, and nonempty streams execute in pass/group order before one frame-wide inverse/finalizer. Packed `Pod` descriptors, reusable lanes, a frame-resident arena, and one aggregate status map share the backend byte budget. Actual-GPU coverage includes a byte-exact 2051×259 two-pass `cjxl` Squeeze stream with two LF groups, plus Palette, local transforms/MA trees, NV12, exact-widened F64, and 16K dispatch. VarDCT covers all 27 strategies, single-pass nonzero AC, all 13 coefficient-order families, stream-defined contexts, default and parametric custom matrices, sectioned raw mode-7 matrix execution after global- or local-tree packet staging, LF/HF correlation and dequantization, multiple LF groups, Gaborish, one-to-three-iteration EPF, and recursive GPU-resident progressive-DC dependencies. | Modular Global/LF/HF image streams, intermediate progressive presentation, lossy/XYB Modular, broader sample metadata, and broader libjxl pixel fixtures remain. VarDCT raw side images still need local-tree conformance fixtures and public non-XYB reconstruction coverage; VarDCT otherwise remains limited to one spectral pass, RGB8 output, and no composition features. |
| Output | GPU-resident native integer Gray/RGB/RGBA plus all 30 portable VPI pitch-linear formats: 20 color layouts and 10 explicitly mapped numeric layouts. Generic color output performs D65 BT.709/BT.2020/Display-P3 primary conversion, Linear/sRGB/BT.709/PQ/HLG/BT.2020 transfer conversion, and BT.601/709/2020 NCL/2020 CL YCbCr packing. | numeric normalization is explicit; F64 requires a native-or-exact-widening precision policy; custom ICC/white-point adaptation and tone/gamut mapping remain |
| Presentation | Same-queue buffer-to-linear-BT.709 RGBA8 SDR and RGBA16F wide-gamut/HDR display pipeline, including BT.2020/Display-P3, PQ/HLG, and BT.2020 constant-luminance input | no tone/gamut mapping or direct surface-format negotiation yet; explicit unvalidated handoff can enqueue display/readback/custom GPU work before final validation, and derived results are discarded if validation later fails |
| CPU transport | Explicit mapped readback after GPU completion | transport only; it never selects a host codec |

Lossless encoder output is independently accepted and reproduced exactly by the published Rust
`jxl` decoder and by `djxl` when it is available in the test environment. `jwgp` is an optional
single-group acceleration box; conforming decoders, including this workspace's generic standard
path, ignore it and decode the standard `jxlc`. The VarDCT encoder is likewise checked with both
oracles, including bounded DCT8 nonzero AC, explicit LF metadata, and horizontal and vertical
two-LF-group images; its output also round-trips through the stock GPU decoder. Actual GPU tests
cover 16K×1 and 1×16K tiled panoramas.

The public decode session traits separate queue submission from completion, prefetch an ordered
bounded frame window, and expose native blocking plus runtime-neutral asynchronous completion.
Frame leases, timing, timecodes, loop metadata, and reference slots remain explicit. The encoder
implements standard Modular animation. Decoder animation, intermediate progressive presentation,
non-alpha extra channels, complete VarDCT reconstruction,
arbitrary ICC transforms, patches, splines, and noise
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
validation. Generic pitch-linear output converts D65 BT.709, BT.2020, and Display-P3 signals plus
PQ/HLG on the GPU without an intermediate readback. `DisplayPipeline` converts those buffers into
an explicit linear-light BT.709 RGBA texture that can be sampled, rendered, or copied without an
additional host wait. SDR BT.709 may use `Rgba8Unorm`; wide-gamut/HDR requires `Rgba16Float` so
out-of-range linear values are preserved instead of silently clipped. Tone/gamut mapping and direct
surface-format negotiation remain explicit future work.

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
