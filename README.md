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

The Modular decoder also reconstructs lossy XYB and original-sRGB color on the GPU. A shared
color-output module serves both coding modes; Modular joins Gaborish, EPF, resampling, alpha,
spot colors and frame composition through the same accounted planar boundary. The reproducible
19-stream corpus covers all EPF iteration counts, all orientations, thin and multi-group images,
floating samples, nine extra channels and four nine-layer animations. Full JPEG XL conformance
and the remaining color/render/container/encoder work are tracked in
[`docs/FULL_JPEG_XL_ROADMAP.md`](docs/FULL_JPEG_XL_ROADMAP.md).

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
Checked-in cjpeg-to-cjxl streams now execute complete non-XYB 4:4:4, 4:2:2, 4:4:0, and 4:2:0
reconstruction on an actual adapter: component-sized LF/AC planes remain resident, and the output
kernel applies JPEG XL's quarter/three-quarter edge-replicating upsampling and encoded BT.601 YCbCr
conversion. When restoration is signaled, only shifted components expand into budgeted resident
planes before the shared Gaborish/EPF ping-pong sequence. Public `GpuDecoder` RGB8 output without
subsampled restoration matches Rust `jxl` and `djxl` within one code; a valid subsampled-restoration
codestream fixture is still required for end-to-end conformance. Local-tree raw-matrix conformance
also remains a gap. Side-image entropy binds only its four-byte-aligned HF-global packet window
rather than the whole codestream.

VarDCT frame resampling now executes 2×/4×/8× filters with standard or custom image-header weights
after restoration and before output conversion. Encoded and presented dimensions are distinct;
odd edges, single-sample axes, multiple LF groups, and spectral AC plus resampling have GPU oracle
coverage. Single-entry packets use cursor-based metadata staging for arbitrary transform maps;
image dimensions no longer select or constrain their transform strategy.

VarDCT spectral and quantized refinement passes now retain independent entropy tables, coefficient
orders, and shifts while accumulating into one resident coefficient set. Final output is validated
across every pass. Checked-in three-pass spectral, two-pass refinement, and recursive DC-plus-AC
streams match Rust `jxl` and `djxl` within one RGB8 code on Apple M5, including bounded uploads and
fragmented input. Intermediate pass presentation is still pending.

VarDCT grayscale and RGB images now normalize all eight image orientations in the final GPU output
pass. Grayscale XYB reconstructs linear luminance before the sRGB transfer function, including
resampled and recursive progressive-DC images. Oriented grayscale and 4:2:0 JPEG-transcode fixtures
also verify padded edge blocks and preservation of GPU-decoded raw quantization matrices across
HF-global metadata continuations. These paths return packed RGB8 and match both reference decoders
within one code value under whole and bounded asynchronous input.

XYB VarDCT accepts every integer source depth from 1 through 31.
Twenty synthetic fixtures cover every depth, plus high-depth grayscale, orientation, resampling,
multiple LF groups, and recursive DC. Both reference decoders agree within one RGB8 code; the
non-XYB YCbCr profile remains limited to 8-bit integer input. XYB VarDCT also accepts legal JPEG XL
floating source precision without rescaling its reconstructed XYB samples.

VarDCT output shares the render backend's GPU color conversion and packing. A single fused dispatch
returns all 20 color VPI pitch-linear layouts, planar YUV, NV21/NV42, P010/P012/P016, and other
classified color layouts with explicit SDR transfer, D65 primaries, range, and siting. No intermediate
RGB image or host pixel conversion is added. Tests cover 39 layout/transfer choices, grayscale,
JPEG upsampling, recursive DC, and Display-P3/BT.2020 conversion. Non-color numeric output and the
luminance mapping needed for PQ/HLG output remain unsupported in this decoder path.

Both coding modes now return planar/interleaved F32 RGB/BGR/RGBA/BGRA through
`PixelFormat::rgb_f32`. Modular normalizes 1–31-bit Gray/RGB/RGBA samples, preserving alpha;
VarDCT keeps unclipped reconstructed color. Float outputs retain negative and greater-than-one
values, use the existing GPU leases, and can feed an `Rgba16Float` display texture. Modular float
conversion currently supports BT.709 primaries with sRGB, Linear, BT.709 or BT.2020 transfer;
VarDCT uses the shared D65 primary conversion. `GpuOutputRequest::with_orientation_policy`
selects default `OrientationPolicy::Apply` or `Keep` for codestream coordinates, including mixed
animation and recursive DC. The frame executor uses this unrounded boundary for GPU crop/blend
composition and four post-transform reference slots. It handles negative/oversized/off-canvas
rectangles, Replace/Add/Blend/Mul/MulAdd, straight or associated alpha with separate background
sources, hidden layers, reference-only frames, mixed coding modes, and recursive DC. Packing and
orientation follow composition. The private frame boundary retains planar RGB plus every extra
channel in one accounted GPU allocation. Each plane follows its own blend mode, reference slot,
alpha selector and clamp flag; color output uses the first declared alpha only at presentation.
Seven nine-layer fixtures cover nine independently typed/depth-coded extras, two alpha planes,
Gray/RGB, both coding modes, distributed groups and shifted resampling. Pre-transform patch
references and broader original color domains remain pending.

Integer decoding covers all 1–31-bit declarations. `native_modular_pixel_format` constructs
Gray/RGB/RGBA layouts with 8-, 16-, or 32-bit storage and zero high padding. Unfiltered Modular
integer planes preserve exact codes; independently declared alpha rescales with exact GPU integer
arithmetic. Filtering/composition uses F32, and final integer rounding evaluates the exact F32
value against the requested maximum without losing additional low bits. Source precision does
not imply lossless precision after filtering or lossy VarDCT reconstruction. The new corpus
contains 42 precision/predictor/alpha fixtures and 40 rendering cases, including every 25–31-bit
XYB metadata declaration, wide extras, RCT/Squeeze, resampling and layered composition.

JPEG XL floating sources support all 154 legal combinations of 2–8 exponent bits and 2–23 mantissa
bits, including binary16 and binary32. `DecodeProfile` retains `SampleBitDepth`, including the
exponent width. `NumericSampleMapping::NativeFloat` returns scalar F32 from a Modular gray source
or a selected floating extra channel in either coding mode. Unfiltered, uncomposed delivery widens
the representation bit-for-bit, preserving signed zeros, subnormals, infinities and NaN payloads.
Inverse transforms precede conversion; resampling, alpha, spots and composition consume decoded
F32 values. Integer and floating extras can coexist. RGB8 output quantizes after those operations.
The checked-in corpus covers every floating precision and 27 rendering/animation cases against
libjxl, with byte-identical whole and bounded fragmented GPU output.

Modular reconstructs integer extra channels with independent 1–31-bit precision.
`DecodeProfile` reports color and extra-channel counts separately from native output formats, and
`AnimationMetadata::extra_channels` retains each declaration. `GpuOutputRequest::with_extra_channel`
selects one plane for native unsigned or normalized scalar F32 output. Color output uses the first
alpha declaration, expands gray when needed, and rescales alpha independently. Spot planes remain
available as data; `SpotColorPolicy::Preserve` explicitly requests base color. Default Render mixes
all declared spots on the GPU after reference storage, before target color/alpha conversion and
packing. Both coding modes share this presentation stage, including composed/resampled frames.
Ten additional libjxl fixtures cover five ordered inks with zero, negative and extended solidity,
independent depths, associated alpha, Gray/RGB, thin axes and distributed transforms.
Six original libjxl fixtures cover multiple alpha, depth, selection mask,
spot color, CFA, thermal, black, and optional planes, including transformed multi-group input.

VarDCT now reconstructs global Modular extra-channel streams before parsing the following LF
header. Only a validated GPU ending cursor advances the color decoder. Its first declared
alpha plane remains resident, and the fused output kernel normalizes its independent precision.
Seven libjxl fixtures cover Gray/RGB at 8/12/16-bit source depths, multiple alpha declarations,
Palette metadata wider than a pass group, single-entry and progressive multi-entry TOCs,
orientation Apply/Keep, and fragmented asynchronous input. Global entropy reuses an input buffer
as small as 40 bytes, preserves GPU ANS/LZ77/predictor state across windows, and stops at its exact
ending cursor. Only consumed windows are planned; total budget capacity can reduce the upload size.
Every global extra plane can also be selected through `with_extra_channel` for native unsigned
or normalized scalar F32 output. That path validates the complete LF/HF/AC stream but skips color
inverse transforms, restoration and color image buffers. Native output requires representable
codes; F32 preserves signed normalization without clipping. Integer extras also execute across
global, LF and AC groups.

Both decoders reconstruct extras with effective 2×/4×/8× upsampling, including image-header
`dimension_shift`; Modular color planes also support all three factors. Selected integer planes
normalize and interpolate on the GPU using the shared standard/custom 5×5 filter before orientation
and packing. F32 keeps fractional samples; native output rounds once at the declared output depth.
Twenty libjxl fixtures cover odd and one-sample axes, independent color/alpha rates, dimensions
across LF/group boundaries and progressive Squeeze. Whole and bounded fragmented outputs agree,
and transient render buffers participate in admission, cancellation and shared memory accounting.
The profile variant is now `DecodeProfile::Modular`, since resampled output is not necessarily
lossless. Integer extra-channel composition uses the same normalized/resampled planes and retains
extended values until presentation. Selected native extras clamp and round once at their declared
depth after composition; scalar F32 preserves the normalized result. Floating sources use the same
filter and composition pipeline after representation conversion, without integer normalization.

Both modes accept associated integer alpha, including independent depths and resampling.
`GpuOutputRequest::with_alpha_output_policy` selects `Unassociated` (default), `Preserve`, or
`Associated`. Conversion follows the requested color transfer and precedes packing, including
RGB output that omits alpha; numeric output preserves sample values. Frame references keep their
original association through crop/blend composition, and only presentation output applies the
policy. Fourteen new still fixtures and three nine-layer sequences cover first-alpha selection,
invisible colors, original and linear RGB, native/F32 output, and bounded asynchronous execution.

The low-level AC executor can now validate an entropy stream and return its exact unaligned
cursor for a following Modular substream. Prefix/ANS GPU tests cover continuation, bounded
resume and malformed endings. Modular group ownership is shared between the coding modes;
five distributed VarDCT fixtures now deliver public color/alpha and all 13 independent extra
planes, including empty globals, Palette, Squeeze, multiple LF groups and progressive passes.
Local inverses finish before GPU row copies assemble a frame arena; its global inverse precedes
output. Whole and bounded fragmented input match exact integer source codes and two F32 oracles.
Cancellation, initial memory retry and malformed extra entropy retain checked ownership.

Modular also normalizes orientations 1–8 on the GPU, including exact native RGB/RGBA and 12/16-bit
samples, all 30 Gray8 VPI color/numeric outputs, Palette/Squeeze, and one-pixel axes. Its frontend
now admits parsed header semantics instead of requiring one fixed wire representation. Unsupported
color/alpha/restoration contracts remain checked; unknown image/frame/restoration extensions fail
with their scope and selector before output. The 23-fixture corpus matches source/Rust jxl samples
and djxl color samples; color conversion differs by at most one code, and whole versus
fragmented async output is byte-identical.

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
| Decode | One public `GpuDecoder::wgpu` routes standard Modular or bounded VarDCT without caller mode knowledge. Modular keeps Prefix/ANS entropy, LZ77, every accepted MA predictor, RCT/Palette/Squeeze inversion, requested output conversion, and bounded resume on GPU. It supports 128/256/512/1024-pixel groups and one through three passes. Channels with both transformed shifts at least three execute through LF-group streams; the header's downsampling brackets assign every remaining channel to exactly one pass, empty sections are zero-validated without dispatch, and nonempty streams execute in pass/group order before one frame-wide inverse/finalizer. Integer extra channels retain independent depths and declarations, with native or scalar F32 selection and standard/custom 2×/4×/8× GPU resampling after complete inverse reconstruction. Packed `Pod` descriptors, reusable lanes, a frame-resident arena, and one aggregate status map share the backend byte budget. Actual-GPU coverage includes a byte-exact 2051×259 two-pass `cjxl` Squeeze stream with two LF groups, plus Palette, local transforms/MA trees, NV12, exact-widened F64, and 16K dispatch. VarDCT covers all 27 strategies, spectral/refinement AC accumulation with independent per-pass tables, all 13 coefficient-order families, stream-defined contexts, default and parametric custom matrices, sectioned raw mode-7 matrix execution after global- or local-tree packet staging, LF/HF correlation and dequantization, multiple LF groups, Gaborish, one-to-three-iteration EPF, recursive GPU-resident progressive-DC dependencies, and public non-XYB 4:4:4/4:2:2/4:4:0/4:2:0 JPEG-reconstruction paths with resident component upsampling/YCbCr conversion. | Modular Global/LF/HF image streams, intermediate progressive presentation, broader original color metadata and full conformance remain. VarDCT raw side images still need local-tree conformance fixtures; subsampled adaptive LF, end-to-end subsampled-restoration conformance, uncommon asymmetric JPEG component layouts, numeric color-channel output and HDR luminance mapping remain typed or unproven gaps. Intermediate pass presentation and pre-transform patch composition remain unsupported; post-transform crop/blend/reference execution retains every supported integer extra channel with independent blend/alpha/reference selection, including Gray+alpha and resampled planes. |
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
implements standard Modular animation. The decoder now executes full-canvas Replace frame sequences
through one mode-neutral frame plan, including mixed JPEG-VarDCT/Modular presentations, exact timing
and names, overwritten zero-duration layers, and recursive progressive-DC dependencies. Nine libjxl
fixtures match Rust `jxl` and `djxl` exactly for Modular and within one RGB8 code for VarDCT under
whole and bounded fragmented async input. The common executor also composes crops and all five
blend modes against four resident post-transform reference slots. Intermediate progressive
presentation, arbitrary ICC
transforms, patches, splines, and noise still require production integration and conformance.

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
depending on Tokio, async-std, or a particular reactor. The stock mode-neutral engine supports
independent full-canvas Replace animations and layered stills. It prepares only the next presentation,
retains source spans until the last source-dependent submission, and preserves pending/output leases
through cancellation. Cropped or blended canvases and reference-only frames remain typed errors.

## Build and validate

```console
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo run -p jxl_gpu_harness -- verify --backend reference
cargo run -p jxl_gpu_harness -- codec fixtures/gpu_gray8_lossless.jxl \
  --format u8 --output-target cpu-readback
```
