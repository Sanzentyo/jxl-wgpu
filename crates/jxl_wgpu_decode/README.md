# jxl_wgpu_decode

GPU-required JPEG XL decode orchestration. Production execution uses the stock WGSL engines and has
no dependency on the published `jxl` decoder. The complete-format backlog and acceptance gates are
tracked in [`FULL_JPEG_XL_ROADMAP.md`](../../docs/FULL_JPEG_XL_ROADMAP.md).

## Executable profile

The stock `WgpuSubmissionEngine` implements a standards-only lossless Modular profile:

- a raw codestream, ordinary `jxlc` container, or reconstructed `jxlp` container with no private
  metadata requirement;
- one final still frame with 1-16-bit integer Gray, RGB, or RGBA lossless Modular samples over any
  bounded 128/256/512/1024-pixel pass-group grid and one through three passes;
- bounded DC-global, LF-group-local, or pass-group-local MA trees with all JPEG XL Modular predictors, including
  weighted self-correcting prediction, leaf offsets/multipliers/context selection, Prefix or ANS
  entropy, hybrid integers, context maps, and the standard LZ77 distance alphabet;
- shared DC-global RCT/Palette/Squeeze and per-LF/pass-subimage RCT/Palette/Squeeze stacks, including
  nonempty DC-global sample channels, group-edge geometry, and one full-resolution unassociated
  alpha channel for RGBA; no restoration filters, other extra channels, or references.

Before submission the decoder inventories the standard image header, frame header, and TOC with
explicit limits. It parses only the bounded DC-global and selected LF-group/pass-group-local MA trees,
histogram descriptors, hybrid integer configuration, context maps, ordered Modular transform
metadata, and pass-group ranges needed to build typed GPU metadata. The progressive schedule maps
each non-LF transformed channel to exactly one pass through the normative
`min(hshift, vshift)`/downsampling brackets; declared empty passes are retained in the public
profile and their physical sections must contain only zero bits. RCT, Palette, and explicit or
default Squeeze are meta-applied
to an exact entropy-visible channel topology without decoding pixels. This accounts odd average/
residual dimensions, channel insertion order, shifts, delta-palette storage, and the meta-channel
prefix under bounded transform/channel/squeeze limits. A portable 32-byte `Pod` descriptor then
proves every packed channel offset fits WGSL `u32` addressing before backend allocation.
For every generalized group, a second 32-byte `Pod` record is appended to the immutable entropy metadata
for each transformed channel. It carries arena offset/stride, dimensions, cumulative decoded sample
range, and an absolute range into a flattened reference-channel list. That list contains only prior
channels whose dimensions and horizontal/vertical shifts match, newest first, as properties 16+
require. A geometry-keyed map bounds construction by channel count times the at-most-60 references
addressable by the 8-bit MA property space. Each self-contained MA/entropy descriptor has its
internal config/tree/table offsets rebased into one immutable GPU word buffer; equal local
descriptors are deduplicated. The 244-byte per-group parameter record exposes independent MA and
channel-descriptor bases. Identical transform plans share one descriptor table even when their MA
or weighted-predictor configuration differs; edge groups retain distinct checked geometry. The
resident arena stride is aligned for dynamic storage offsets.
Inverse planning walks the stack and each Squeeze parameter in reverse while retaining only the
current and immediately restored topology, rather than materializing a channel table for every
transform. The parser also charges the cumulative topology work, so a bounded channel count cannot
be combined with an adversarially quadratic transform sequence.
A standalone `ModularSqueezePipeline` executes one horizontal or vertical inverse parameter without
leaving GPU storage. Average, residual, and destination are checked non-overlapping views of one
read-write arena, which removes storage-binding alias ambiguity and permits later lifetime reuse.
Its portable WGSL emulates the required signed 64-bit smooth-tendency intermediate with two `u32`
words, then applies the specified wrapping `i32` reconstruction. Actual-adapter tests compare odd,
even, one-dimensional, and extreme-value cases to an independent scalar oracle. The stock decoder
feeds transformed entropy output directly into this arena.
A standalone `ModularRctPipeline` applies every one of the 42 normative operation/permutation
combinations in place to three equal-size, non-overlapping views of that same arena. Each invocation
loads all three signed words before writing any permutation, while explicit unsigned add/sub helpers
preserve wrapping `i32` behavior and an explicit bit-pattern shift fixes negative rounding. Its
64-byte, 16-byte-aligned `Pod` uniform and linear Scalar/32/64/128/256 policy variants are validated
before recording. An actual-adapter differential covers all types with odd dimensions, padded
strides, nonzero offsets, and signed extremes.
The reverse planner emits RCT and Squeeze jobs in one exact inverse-order stream. RCT jobs reuse the
three current plane views in place, while a best-fit interval allocator reserves each Squeeze
destination before its dispatch, retires that channel's average and residual immediately afterward,
and merges adjacent free spans. Thus an arbitrary valid RCT/Squeeze composition keeps only live
planes, not every transform generation.
A nested horizontal/vertical actual-adapter test records three ordered dispatches in one encoder and
maps only the final plane; the checked 45-word entropy topology uses a 90-word peak arena and reuses
its first retired range for the restored output. The real progressive-DC LF2 fixture lowers 13
parameters to 37 jobs and three final full-resolution planes within twice its entropy sample count.
An RCT/Squeeze/RCT test emits five ordered jobs and executes them in one command encoder, copying all
three noncontiguous final planes into one staging map. Production scheduling applies a concrete
group-local inverse plan and a 144-byte region-aware finalizer as soon as each group's final entropy
segment finishes when no cross-group transform is present. For DC-global Palette/Squeeze, channels
with both transformed shifts at least three are decoded by LF-group subimages first; channels with
either shift below three are decoded by pass-group subimages. Each subimage finishes its local
inverse plan, then its final planes are copied row-wise into one frame-resident transformed arena
before the scratch lane is reused. One frame-wide inverse plan and one finalizer run after the last
pass group.
Palette continuation state remains GPU-resident.
Every token range and canvas origin comes directly from standard frame sections. It does not
decode a pass-group entropy token, residual, predictor, color transform, or pixel on the CPU.
The Modular metadata reader operates on a checked shared-span bit input rather than indexing one
host slice. The same span table copies exact entropy ranges into the reusable GPU window, including
ranges that cross a chunk or entropy-word boundary, without first joining the codestream. The
contiguous `GpuDecoder::open` creates one span, while `GpuDecoder::stream` accepts borrowed
`ContainerStreamEvent` values and supplies the same engine boundary with an arbitrary checked span
table plus the incrementally produced inventory.

The compute shader reads Prefix or ANS symbols and hybrid integers from bounded codestream windows,
validates every bit and output bound, applies LZ77, walks the MA tree, reconstructs all predictors,
and reverses YCoCg for RGB(A). Up to 512 budget- and device-resolved scratch lanes decode independent
groups in one `dispatch_workgroups` wave; 64 logical group invocations are packed into each portable
compute workgroup and their canvas rectangles do not overlap. Large frames use ordered batches backed
by one reusable stream window instead of binding the full codestream. If any accepted stock Modular
group exceeds the resolved window, it resumes over ordered segments with 16-byte backward and
forward overlap. The common logical cursor, Prefix/ANS state, LZ77 copy state, decoded count, last
value, and first error remain in a 16-byte-aligned 32-byte record at the end of the same scratch
lane. Generic MA adds the Property-8 gradient history for a 48-byte record; a
Weighted/SelfCorrecting consumer also retains four true errors and twelve subprediction-error
accumulators for a 112-byte record. A resume exactly at a channel boundary resets channel-local
predictor state. State and LZ allocation use the maximum requirement across all selected group
descriptors, and mixed Prefix/ANS frames are reported explicitly. Multi-group streams also execute
the DC-global Prefix/ANS stream through the same descriptor kernel and exact termination contract,
whether it reconstructs zero or nonzero samples. One aggregate status staging buffer is mapped once
after the last batch, and its four-word record plus every LF/pass-subimage record is checked before the
frame is reported. No reconstructed sample is produced on the CPU.
Output pipeline selection is also per frame. When every plane offset and row stride is four-byte
aligned and every actual internal group edge ends on a distinct storage word, the shader uses
ordinary word RMW/store. Layouts with an odd stride, offset, or group edge use the atomic byte-safe
pipeline. The proof is performed on the host from the validated output layout and typed group
rectangles; there is no caller hint that can force the non-atomic path.
The validated MA-tree IR receives a second independent specialization proof. If every group's
resolved local/global tree has the same fixed specialization, and every decision is
channel-only and channels 0 through 3 all terminate at Gradient leaves with zero offset and unit
multiplier, the shader lowers the four cluster ids into the parameter record and runs a nested-row
Gradient loop without per-sample MA traversal, coordinate division, unused-neighbor loads, or
predictor dispatch. Any unsupported property, malformed/cyclic route, non-Gradient leaf, or
self-correcting requirement selects the complete generic MA-tree kernel instead.
For an exact U8 `NormalizedGray8` request, the same proof also permits the Gradient loop to emit
each sample directly and omit the separate group-finalization traversal. If descriptor analysis
proves that LZ history is invocation-private, the reconstruction lane retains only the current and
previous rows (512 physical words versus 65,536 logical sample words for a full 256x256 group).
Wider LZ histories, native Modular output, RGB(A), and every other VPI mapping keep
the complete logical reconstruction workspace and finalize through the generic output contract.
`DecodeProfile::ModularLossless` reports the declared pass count and this specialization as
`ModularPredictionProfile::MetaAdaptive` with exact node/decision/leaf counts, maximum depth, and
self-correcting usage; custom synthetic engines use the distinct `Fixed` variant.

For the lossless-Modular `WgpuSubmissionEngine`, the complete transform wire grammar and resulting
channel topology are parsed, local MA trees select per-stream metadata bases, and resident
RCT/Palette/Squeeze stacks execute for single- and multi-group streams. Multi-group DC-global
Palette/Squeeze sample data is reconstructed in a frame arena. LF-group streams own exactly the
channels whose horizontal and vertical transformed shifts are both at least three; pass groups own
the remaining channels, including asymmetric shifts. LF streams execute before nonempty pass
streams in pass/group order, all use the same bounded-window executor and aggregate status map, and
one global inverse/finalizer runs after assembly. One through three passes produce an exact final
image; intermediate pass presentation is not yet exposed. Global/LF/HF image streams, lossy/XYB Modular, non-alpha extra channels,
patches, splines, noise, and
reference-frame animation remain typed unsupported profiles. The public `GpuDecoder::wgpu` constructs `WgpuDecodeEngine`, inventories
the standard frame once, and selects this engine or the bounded VarDCT engine from
`FrameEncoding`. Callers do not choose or probe a coding mode. Both child engines retain their
mode-specific bindings and pipeline caches while sharing the backend byte budget.

### Bounded standard VarDCT engine

The coding-mode-neutral `GpuDecoder::wgpu` selects the VarDCT production engine for two bounded
standard packet topologies. A one-entry zero-AC TOC retains the original single regular 8x8,
16x16, 32x32, 16x8, 8x16, 32x8, 8x32, 32x16, or 16x32 transform. A sectioned TOC covers one or
more independently bounded LF groups with GPU-decoded mixed maps of any of JPEG XL's 27 regular
and special strategies across one or more 256-pixel pass groups. Those pass groups may carry real single-pass HF coefficients
using any of the 13 natural or entropy-coded custom coefficient-order families. Scanline and
entropy-coded center-first TOC order are both accepted: inventory retains physical section ranges
and the frontend normalizes them to logical group order before assigning pixel rectangles and
per-group scratch. The explicit section topology removes the ambiguity with a one-entry
single-transform packet. The sectioned form supports odd and asymmetric pixel extents across LF
group boundaries while keeping edge padding internal to GPU storage; 2056x256 is the checked
two-LF-group boundary case.
Both forms accept exactly one final 8-bit XYB still frame and one pass. The packet contract
accepts either adaptive LF smoothing or its standard skip flag, every 3-bit X/B frame
quant-matrix scale, every normative default or parametric custom dequantization matrix encoding,
disabled/default/custom Gaborish, disabled/default/custom
one-to-three-iteration EPF, arbitrary valid HF block-context maps, and one spectral pass.
`global_scale`, `quant_lf`, LF extra precision, the quant field, per-block `hf_mul`, sharpness,
per-frequency-cell HF chroma correlation, MA
properties 0 through 15, and weighted self-correcting prediction are read from the stream. The
combined single-entry and shared-global-tree packet forms resume across the shared bounded-window
planner without an intermediate map. Their 64/128-byte `Pod` state records the active LF/HF phase,
both decoded counts, first-block count, extra precision, ANS/LZ state, and predictor state. The
packet frontend also represents an absent LF-global tree, packs each LF-local tree independently,
executes LF image entropy on GPU, maps the aggregate end cursors, then parses and packs the following
HF-local trees without decoding host image symbols. Oversized LF-local ranges use the shared
16-byte-overlap planner, one reusable upload, and ordered queue submissions. The ABI defines
16-byte-aligned 64-byte generic and 128-byte SelfCorrecting `Pod` records that preserve ANS/LZ,
consumer, and predictor state. Current local-tree planning reserves the conservative 128-byte
capacity per group because the following HF-local tree is discovered only after the LF map; only the final segment performs entropy termination and contributes to the
single aggregate LF status map. Separate `decode_vardct_lf` and `decode_vardct_hf` entry points
preserve resident LF reconstruction across that boundary. The stock runtime-neutral pending-frame
state machine owns every LF submission, both aggregate status maps, and the initial plus dynamically
admitted metadata reservations. It is actual-GPU tested with ordinary multi-LF-group `cjxl` output
through blocking and async completion. The image header
must declare the standard sRGB/D65
presentation encoding, no ICC profile or extra channel, orientation 1, and no crop, blend,
reference, preview, animation, subsampling, upsampling, spectral progressive pass, or other frame
feature. A separate stock path accepts recursive progressive-DC dependencies. It keeps three F32
XYB planes resident, uses 96-byte conversion and 48-byte LF-pack `Pod` uniforms, validates every
hidden and visible status, and publishes only the final frame. A single-entry intermediate frame
first executes HF metadata on GPU, maps its bounded HF-global cursor, host-parses only scalar
HF-global tables, and resumes general AC plus downstream reconstruction on the same queue. Fixed
scratch/status capacity is admitted before submission; descriptor/order/window bytes discovered at
the cursor are admitted through the same shared budget. `cjxl --progressive_dc=1` and
`--progressive_dc=2` actual-GPU outputs are checked through blocking and runtime-neutral async
completion against Rust `jxl` within one RGB8 code. Parametric matrix modes 0 through 6 populate the
resident resource table. For raw mode 7, the sectioned global-tree path now decodes the complete
three-channel Modular side image with the common GPU entropy executor, runs its resident
Palette/RCT/Squeeze inverse schedule, validates one 16-byte mapped status, and overlays positive
finite weights into each aliased strategy-matrix target before AC/render. A real cjpeg-to-cjxl
JPEG-transcode stream covers that primitive on an actual adapter. Local-tree raw matrices, bounded
side-image stream windows, and the fixture's non-XYB JPEG-reconstruction presentation path remain
typed gaps.
A valid UTF-8 frame name is preserved in authoritative `FrameMetadata`; invalid bytes return a
typed error. Container/codestream parsing is capped at 16 MiB and 32 boxes before any fragmented
payload can be reassembled; this is an engine limit, not a late profile check after the generic
1-GiB parser ceiling.

The retained VarDCT source is a checked logically contiguous table of shared spans. Entropy-window
planning depends only on its logical length; LF, combined LF/HF, staged HF, and AC uploads copy
their exact ranges across arbitrary physical span boundaries. Whole-range kernels still require a
full GPU codestream buffer, but it is initialized while mapped directly from the span table and
zero-padded to four bytes without constructing a second full-size host `Vec`. Initial LF/HF packet
metadata, MA descriptors, block-context maps, custom coefficient-order permutations, and
cursor-dependent local-HF metadata all read the same span table without joining it. Slice-based
low-level APIs remain zero-copy through the same reader contract. The public decoder remains
runtime-neutral: `GpuDecodeStream` feeds the public transport events synchronously, then returns the
ordinary sync/async `GpuDecodeSession` after authoritative end-of-input. A separate cloneable input
budget covers the compressed spans retained by concurrent streams without conflating host storage
with the GPU allocation budget. Its permit moves into the codec source and is released at the last
submission that still needs those bytes, or immediately when the stream/session is cancelled.

The host inventories bounded scalar headers, packs the shared or local MA-tree and coefficient entropy descriptors,
and expands only the small HF coefficient-order metadata permutation. It does not decode an LF/HF
image entropy symbol or coefficient value. One GPU submission decodes and validates LF/HF metadata,
dequantizes and smooths LF, lowers every non-overlapping first block into typed HF tasks, decodes
each pass group through the common Prefix/ANS/hybrid-integer/LZ77 executor and all-order coefficient
sink. The HF metadata channels retain their logical dimensions while addressing capacity-strided
storage, so the `hf_mul` row cannot alias the packed strategy row when the actual first-block count
is below the allocation capacity. Block-context selection reads the resident quantized LF planes
and each task's `hf_mul`; its variable tables share the entropy bundle, while per-group LZ history
occupies a disjoint tail slice of its LF group's reconstruction buffer so the pass remains within eight
portable storage bindings. Each LF group owns independent reconstruction, raw-metadata,
coefficient, packet-status, artifact, occupancy, and HF-status buffers. LF values use the stream's
default or explicit channel dequantization multipliers and LF chroma-from-luma slopes before
scattering into the full-image resident atlas. HF correlation uses the same explicit base values
and colour-factor reciprocal while lowering each frequency cell. Adaptive LF smoothing runs once over
the complete block grid, so the 2048-pixel LF boundary is not treated as an image edge. Each
artifact carries global output/LF/correlation origins and creates 27 compact strategy buckets and indirect dispatch
records. The submission executes every populated bucket through the resident regular or special
VarDCT renderer using the normative default matrix for that strategy and an explicit regular/wide/
special coefficient layout, optionally applies the signaled Gaborish weights, constructs the signaled
per-block EPF inverse-sigma field, runs EPF0/EPF1/EPF2 as selected by the one-to-three iteration
contract through a shared resident ping-pong plane set, applies inverse opsin plus sRGB transfer,
and writes tightly packed RGB8 without an intermediate image readback. Combined/global packet and
AC pass-group ranges that fit the resolved entropy cap retain the one-submission path. Either
oversized consumer instead uses the consumer-neutral 16-byte overlap plan, one reusable stream/
parameter pair, and ordered queue submissions. The final combined/global packet command shares the
first downstream submission. A 464-byte aligned `Pod` tail per pass group preserves bit/ANS/LZ
state, nested block/channel/order progress, coefficient-sink error, and the 96-word nonzero context
grid; only the final window validates exact ANS/padding termination. Local-tree frames first run one
or more bounded LF submissions and map one aggregate LF cursor record, host-pack the HF descriptors,
then run one or more bounded HF submissions through the same upload before the already-recorded
downstream work and one final aggregate validation map. The final HF window shares the first
downstream queue submission instead of adding an avoidable boundary. Every LF group's packet and artifact status plus one 32-byte record per pass group
share the final map; cleared downstream buffers and zeroed indirect
dispatch records make a rejected packet non-authoritative rather than an unchecked render. There
is no CPU pixel, coefficient, transform, quantization, residual, entropy, or color fallback.

The only output descriptor is `vardct_rgb8_format()`: interleaved RGB8 with explicit BT.709/sRGB
primaries, IEC sRGB transfer, full range, and no YCbCr encoding. It is accepted directly by
`DisplayPipeline::submit_image`, which produces a GPU-resident linear-BT.709 texture without an
intermediate CPU readback. `VarDctDecodeMemoryStats` accounts every upload, metadata, status,
uniform, artifact, coefficient, XYB, optional three-plane restoration scratch, EPF sigma and
per-pass uniforms, transform-scratch, and
output byte. By default those bytes use
the backend budget shared by decode, encode, and readback; `VarDctSubmissionEngine::with_memory_budget`
can instead select an explicit sharing group. Transient reservations survive until the final
aggregate status map completes. For local-tree frames the initial reservation contains all LF
metadata; if the cursor-dependent HF metadata peak is larger, the exact difference is admitted
from the same shared byte budget before the second submission. The packed output reservation
survives through the final tracked
`GpuBufferLease` clone, including an early `UnvalidatedGpuImageFrame`; only the validated frame
carries authoritative metadata and changed regions. Native blocking and runtime-neutral
poll/future completion use the common decoder session API, and the engine compiles for browser
WebGPU without a Tokio or async-std dependency.

`VarDctDecodeMemoryStats` separately reports the shared packet stream peak, initial packet batch
count, reusable AC stream peak and batch count, reusable parameter bytes, LZ scratch, and
execution-state total. For combined/global packets the initial count covers the complete packet;
for local trees it covers LF. HF packet descriptors are host-discovered after the LF map, so
`hf_packet_stream_batch_count()` and `submissions_per_frame()` become exact when that dynamic plan is
installed; performance harnesses sample them after frame completion. Applying
`WgpuDecodeEngine::with_stream_window_limit` configures both coding-mode engines and governs
combined/global packets, local-tree LF/HF packets, and AC pass groups. It is a caller upper bound:
device limits and deterministic planning against the shared budget's total capacity may select a
smaller four-byte-aligned value, exposed as
`VarDctDecodeMemoryStats::resolved_stream_window_limit_bytes`. Planning does not sample live
headroom, so concurrent opens are reproducible; live jobs still receive typed non-blocking memory
backpressure at submission. If even the 40-byte overlap/sentinel layout exceeds the budget,
`MemoryBudgetTooSmall` reports both exact planned bytes and the configured limit before GPU work.
Cursor-dependent local-HF metadata remains the one dynamic addition and admits only its exact
positive difference from the same budget. Actual-adapter runs force a 40-byte 32x32 combined packet,
256-byte shared-global/local-tree/nonzero-AC paths, and an intermediate budget-resolved cap through
blocking or runtime-neutral async decode, typed corruption/backpressure, and cancellation-driven
reservation release.

The actual-adapter matrix covers all nine accepted single regular transform extents plus sectioned,
odd/asymmetric multi-task and multi-pass-group frames. Lower-level GPU kernel oracles cover all 27
inverse transforms, including AFV and 64/128/256-scale strategies. It GPU-encodes each
standard packet, executes the complete resident decode, reads the result back explicitly, and,
when `djxl` is installed, compares it with that independent decoder (at most one RGB8 code of
rounding difference for the covered solid-image cases). The Dct8 case also exercises the
runtime-neutral async completion and GPU display conversion. A deterministic mutation of the
Modular header into a malformed local-tree descriptor is rejected by the bounded host metadata
parser before submission. The matrix also
verifies that readback releases its shared reservation while the last decode-output buffer clone
continues to own the exact output bytes. A separate 438x589 libjxl fixture exercises six nonempty
pass groups, 4,070 DCT8 tasks, a custom three-channel coefficient order, nonzero AC coefficients,
and a self-correcting MA tree; GPU RGB8 output differs from both Rust `jxl` and `djxl` by at most
one code.
A checked-in 257x257 libjxl effort-5 fixture exercises a mixed strategy map, LF extra precision,
three HF block clusters, custom orders 0 and 1, and a first-block count smaller than the
capacity-strided metadata allocation. The actual-adapter test executes its single-pass AC and every
populated regular/special bucket through the stock decoder, then permits at most one RGB8 code of
difference from Rust `jxl` and optional `djxl`.
A separate deterministic 438x589 fixture stores the same bounded one-pass DCT8 topology in libjxl's
center-first entropy-coded TOC order. Its physical pass-group order differs from row-major order;
the frontend test proves each logical group retains the matching physical bit range, and the
actual-adapter test matches both development-only pixel oracles within one RGB8 code.
A second deterministic 438x589 libjxl fixture enables standard Gaborish weights while disabling
EPF. The same actual-adapter test executes inverse VarDCT, fused three-plane Gaborish, and RGB8
packing in one command buffer, and differs from both development-only pixel oracles by at most one
code. It also checks the exact extra reservation of three padded F32 planes and one 80-byte
uniform.
A pair of deterministic 257x17 libjxl fixtures covers the standard EPF2 bundle and EPF3 custom
iteration count on an odd, 8-pixel-unaligned extent. The actual-adapter test executes Gaborish plus
EPF1/EPF2 or EPF0/EPF1/EPF2 without readback between stages, checks the exact shared scratch,
sigma, and uniform reservations, and permits at most one RGB8 code of difference from both Rust
`jxl` and `djxl`. A separate actual-GPU malformed-metadata test feeds sharpness 8 through the same
WGSL validation function used by the packet decoder and requires the typed `Sharpness` error.
Two deterministic 2056x256 standard fixtures contain a 2048x256 LF group followed by an 8x256
tail group and nine pass groups. One enables whole-image adaptive LF smoothing and the other uses
the standard skip flag. Their actual-adapter tests execute both groups, default Gaborish, EPF1, and
RGB8 packing in one queue submission, validate every packet/artifact/pass-group record from one
map, and match Rust `jxl` plus optional `djxl` within one RGB8 code.
An actual-GPU block-context differential test covers negative and positive LF thresholds, exact
threshold boundaries, multiple quant-field segments, all channel positions, and distinct order
IDs against the normative scalar index formula. Naga semantic validation runs even without an
adapter.
An additional actual-adapter test generates a deterministic 2056x256 RGB source and invokes
`cjxl` with distance 2, effort 7, and raw-codestream output. Its ordinary per-LF-group local trees,
including non-default X=5/B=5 quant-matrix scales, complete through the stock frame engine in two
submissions. Blocking and runtime-neutral async results differ from Rust `jxl` and optional `djxl`
by at most one RGB8 code. The test also requires a typed refusal of early unvalidated output before
the HF submission, and verifies that abandoning the LF stage releases its shared memory reservation
after GPU/map completion.
The serialized pass-1/pass-2 zero-flush values remain present in the frame inventory. libjxl 0.12
and the Rust `jxl` implementation accept those parameters but their EPF weight function does not
apply them; the GPU formula follows those executed references rather than inventing a threshold
operation.

This is not full VarDCT coverage. Multiple spectral/refinement passes, explicitly published
progressive intermediates, local-tree/windowed raw strategy matrices and other Modular side images,
alternate RGB/gray/YUV/NV12/VPI outputs, ICC/HDR and other bit depths, crop/blend,
extra channels, other progressive passes, animation, and reference frames return typed unsupported
errors. They are not substituted with dummy coefficients or a CPU implementation.

### Measured lossless Modular checkpoint

On an Apple M5, the 36,643,474-byte 7680x4320 Gray8 conformance codestream decoded to the exact
33,177,600-byte source hash in one codec submission. Warm sequential decode plus staged UMA
readback (`warmup=1`, `iterations=7`) selected 64 invocations per workgroup. Before the aligned
output and private distance-one history paths, 32/64/128/256 produced median latencies of
280.151/280.387/292.455/303.263 ms respectively, while 64 had the best mean and p95
(280.878/282.067 ms). Aligned output plus private distance-one history first reduced that to
209.949 ms median. The subsequent host-proven channel-fixed Gradient kernel measured 132.569 ms
median, 132.630 ms mean, 129.145 ms minimum, and 134.377 ms p95: 36.86% below the 209.949 ms
checkpoint and 11.0x faster than the earlier 64-lane, 8-MiB-window,
one-invocation-workgroup checkpoint (1.457 s). Direct normalized-Gray8 output plus the proven
two-row workspace then measured 110.366 ms median, 110.507 ms mean, 108.914 ms minimum, and
111.574 ms p95 with the same exact hash and one codec submission: 16.75% below the 132.569 ms
checkpoint. These are one-device engineering measurements, not cross-adapter performance
guarantees; `memory_stats` exposes the resolved window, lanes, workgroups, output and
reconstruction paths, logical and physical sample workspace, LZ storage, and submissions for each
device and request.

The same direct/two-row pipeline decoded the exact 15360x8640 Gray8 hash from a 146,573,715-byte
codestream in four bounded submissions: warm median 426.034 ms, mean 427.862 ms, and minimum
423.385 ms (`warmup=1`, `iterations=3`), 35.20% below the preceding 657.463 ms/six-submission
checkpoint and 13.76x faster than the earlier 5.861 s/32-submission implementation. The
132,710,400-byte output remains persistent while the stream window and proof-sized parallel
scratch are reused across waves.

## GPU output formats

`GpuOutputRequest` always carries a concrete `jxl_gpu_formats::PixelFormat`; the request is never
ignored. Construction is intentionally breaking and explicit: `GpuOutputRequest::color` accepts
only classified color formats, while `GpuOutputRequest::numeric` requires a
`NumericSampleMapping`. Canonical Gray/RGB/RGBA descriptors shared with
`jxl_wgpu_encode::LosslessModularFormat` produce exact native unsigned output: 1-8 valid bits use
one byte per component and 9-16 valid bits use a little-endian 16-bit component with the valid code
in its low bits. Gray selects `NumericSampleMapping::NativeUnsigned`; RGB/RGBA use the color
constructor. The 8-bit Gray conversion path additionally supports:

- all 10 numeric VPI 4.1 pitch-linear formats: U8, S8, U16, U32, S32, S16, 2S16, F32, F64, and
  2F32;
- all 20 color-bearing `VpiPitchLinearFormat` descriptors: limited/full Y8 and Y16; limited/full
  NV12 and NV24; limited/full UYVY and YUYV; and interleaved or planar RGB8, BGR8, RGBA8, and
  BGRA8;
- equivalent classified 8-bit planar or semiplanar YCbCr layouts at representable 4:4:4, 4:2:2,
  or 4:2:0 sampling, including NV21/NV16/NV61/NV42 and I420/I422/I444 descriptors.

Every color request requires a defined transfer and range. Linear, sRGB/sYCC, and BT.709/BT.2020
transfer conversion runs in WGSL. Gray is replicated into RGB/BGR, alpha is opaque, and YCbCr
chroma is the exact neutral code. Full/limited luma quantization, native 16-bit luma words,
four-byte YUYV/UYVY pairs, odd-width tail duplication, and all plane writes happen directly in the
GPU output buffer. PQ and other unimplemented transfers return `UnsupportedOutputFormat`.

Generic outputs use `classify_pixel_format`; the exact native Modular descriptor is recognized
separately because sub-8/sub-16 valid-bit padding is part of its contract. `NormalizedGray8` maps
code 0..255 across the full unsigned range, the nonnegative half of a signed type (0..MAX), or
float 0..1, and replicates the result for 2S16/2F32. Integer endpoints and rounding are computed
without overflow; F32 bits are formed deterministically with integer arithmetic rather than
backend-relaxed division.

F64 precision is never inferred silently. `NormalizedGray8F64` requires an `F64OutputPolicy`:
`NativeRequired` rejects devices without enabled `SHADER_F64`,
`NativeOrExactF32Widening` explicitly permits a compatibility fallback, and
`ExactF32Widening` always constructs binary64 as the exact widening of the correctly-rounded F32
normalization. The compatibility result is valid IEEE-754 F64 storage, but is not a precise F64
division. When `WgpuBackend::native_f64_enabled()` is true, the native path is lazily compiled and
evaluates `f64(gray) / 255.0` in WGSL. `WgpuDecodeSession::f64_output_path` reports the resolved
path. The returned `GpuImageFrame` owns pitch-linear GPU buffers. CPU readback occurs only when the
application explicitly requests one. On an eligible native UMA backend, the stock decoder marks
its caller-visible output `MAP_READ`, allowing `ImageReadbackPipeline::submit` to map the sole
output in place; portable and aggregate requests retain the explicit staging-copy path.

## Public flow and bounds

1. Construct `GpuDecoder::wgpu` around an application's existing `WgpuBackend`; construction is
   fallible because every mode-specific kernel policy is validated up front.
2. Call `open` with encoded bytes and a `GpuOutputRequest`.
3. Fill the ordered GPU queue with `prefetch`, `poll_prefetch`, or the runtime-neutral
   `prefetch_async` future. Prefetch submits work and never waits for frame completion.
4. Optionally borrow `pending_frames`/`front_pending_frame` and call the stock
   `WgpuDecodePendingFrame::unvalidated_gpu_frame` to enqueue same-device, same-queue display, readback,
   or custom GPU work before mapped-status validation completes. For a local-tree VarDCT frame,
   this returns typed `UnvalidatedOutputNotSubmitted` while only the LF stage is queued; call it
   after polling/waiting has validated the LF cursors and queued the dependent HF submission.
5. Consume the oldest pending frame with `next_frame` synchronously or
   `next_frame_async`/`poll_next_frame` through `std::future::Future`.
6. Retain each `GpuFrameLease` only while its GPU resource is needed. The lease holds an
   `InFlightPermit`; dropping it wakes a pending submission.

The early handoff returns `UnvalidatedGpuImageFrame`, not `GpuFrameLease<GpuImageFrame>`. It
contains only the queue token, requested layout, and permit-bearing buffer leases; authoritative
frame metadata and changed regions remain unavailable until `next_frame` succeeds. Queue ordering
removes the host synchronization point, but not validation: consumer commands already submitted
before a later validation failure cannot be rolled back, and their textures or bytes must be
discarded. Keep `GpuBufferLease` clones alive for accounted ownership; cloning a raw
`wgpu::Buffer` through `as_wgpu_buffer` is outside budget tracking.

The engine boundary is likewise split: `GpuSubmissionSession::submit_next` returns an owned
`GpuPendingFrame`, whose native `wait` or runtime-neutral `poll_complete` performs only completion
and mapped-status validation. `GpuDecodeSession` keeps these values in a `VecDeque` and therefore
returns presentation frames in submission order even when later GPU work completes first.
`PrefetchProgress` reports cumulative submitted frames, current queue depth, explicit stream end,
and typed frame-slot/memory/poller backpressure. A requested depth larger than `max_frame_slots` is
rejected instead of creating an async wait that can never complete.

`AnimationMetadata` carries the stream timebase, loop count, and timecode-presence flag.
`FrameMetadata` carries exact duration ticks, cumulative presentation-start ticks, and the
bitstream `timecode` when declared. The session rejects timebase, accumulated presentation tick,
or timecode-presence mismatches as typed errors. A cancelled async wait can be resumed through the
same session synchronously or by a later future.

The CPU/WGSL per-group Modular parameter ABI is a checked 244-byte `repr(C)` POD. Its first 12 bytes are the
shared `EntropyStreamParams`: token start/end bounds and the descriptor-derived LZ ring mask. The
same typed prefix starts the 240-byte, 16-byte-aligned VarDCT packet entropy record. Each consumer supplies its own
storage access and LZ scratch-base functions; geometry, prediction, output, and coefficient state
remain consumer-specific. Consumers whose entropy owns the complete token range also call one
shared terminator for the ANS final-state and at most seven zero-padding bits; VarDCT packet streams
followed by fixed metadata finalize ANS first and validate the enclosing section after that tail.
The VarDCT AC parameter record is a separate 144-byte aligned `Pod`. Its window suffix carries
logical/upload starts, available/full ends, the yield boundary, first/final flags, a 464-byte
execution-state offset, and the canonical status index. The VarDCT packet record has its own seven
window fields, including a bounded-mode bit independent of FIRST/FINAL and a stream-base bit offset.
Combined/global-tree packets use those fields with 64-byte generic or 128-byte SelfCorrecting state;
five explicit words retain phase, LF/HF counts, first blocks, and extra precision. Staged local-tree
LF and HF packets use the same state sizes and reuse one allocation across their two sequential
stages.
The Modular suffix begins with six window fields: logical segment start, physical upload start,
full stream end, yield boundary, first/final flags, and the aligned entropy-state offset. It then
carries four plane offset/stride pairs, exact output
channel/order/depth/range/transfer codes, the resolved numeric mapping, global status index, MA
stream index, the proven fixed-leaf predictor/offset/multiplier and four channel cluster ids, the
output traversal mode, weighted-predictor header, and shader-visible logical size. Records are a
tightly packed read-only storage array; a separate 16-byte uniform selects the global group range and local
scratch-lane stride for each wave.
Codestream segments are rounded to four bytes and include a zero sentinel word for bounded
cross-word peeks. A caller may set an additional cap with
`WgpuSubmissionEngine::with_stream_window_limit`; device limits and the shared per-slot byte budget
can only reduce it. Whole-group token offsets are rebased to their upload. Split groups instead keep
one group-relative logical cursor and map it into each upload through the segment fields. A token
may finish past a yield boundary inside the 16-byte forward overlap; the following segment includes
the same bytes as backward overlap and resumes only after the completed output value. Final ANS and
zero-padding validation runs only on the final segment. The peak window grows only when a larger
batch fits the actual per-slot shared byte budget and device storage-binding limit, allowing a large
still image to coalesce submissions without compromising concurrent small-frame admission.

Entropy metadata, bounded lane scratch including the selected 32/48/112-byte resume record, aggregate
status/readback, parameters, dispatch control,
output, and peak stream-window sizes are overflow-checked against storage, uniform, and device
buffer limits. Lane count is the minimum of the 512-lane watchdog cap, group count, device workgroup/storage
limits, and the scratch plus actual peak stream space affordable per requested frame slot. The LZ
ring itself is the next power of two above the largest reachable back-reference derived from the
distance histogram, hybrid-integer configuration, and group width; it is never sized by decoding
residuals on the host. A proven one-word ring uses invocation-private last-value state and consumes
no reconstruction storage; wider histories retain the descriptor-sized storage ring. If the requested slot count is
not affordable but one complete frame is, the prepared backend narrows it and propagates the
resolved bound into the actual session limiter and prefetch validation. `WgpuDecodeSession::memory_stats`
reports complete per-frame, output-lease, transient, peak-window, resolved-slot, logical/physical
LZ sizes, logical/physical reconstruction-sample workspace, per-lane execution-state bytes, parsed
Prefix/ANS representation, selected output-write/output-traversal
and reconstruction-specialization paths, lane/workgroup counts, stream batches, and actual
submission counts. Concurrent jobs opened through an engine or its
clones use the `WgpuBackend`'s shared
transient memory budget by default, so decode, encode, and generic readback apply one aggregate
admission bound. `WgpuSubmissionEngine::with_memory_budget` instead accepts an explicit cloneable
`MemoryBudget` for applications that intentionally define another sharing group. Admission is
non-blocking and memory pressure returns a typed, retryable error.

Transient stream-window/entropy-metadata/reconstruction/status bytes remain reserved until the
status map has completed. The shared host codestream `Arc` is not counted as GPU memory. Output
bytes are carried by `GpuBufferLease`, so explicitly cloning that lease extends
the same reservation and dropping the decode session cannot release it prematurely. GPU frame and
output containers are intentionally not cloneable. A raw `wgpu::Buffer` cloned through
`GpuBufferLease::as_wgpu_buffer()` remains valid wgpu ownership but is outside the byte budget; the
reservation returns after the final tracked lease is dropped. The current stock profile has
exactly one visible frame; the same ownership contract applies to future animation frames.

Repeated small and sequential decodes reuse a decoder-local, bounded cache for entropy metadata,
reconstruction, status, status-staging, and POD parameter buffers (plus the native-F64 dummy when
needed). A cache hit requires the exact allocation size, usage flags, and ABI alignment. The raw
JPEG XL codestream and caller-owned output are never admitted to this pool. Codestream upload reads
only each planned bounded range from a checked table of shared input spans, even when the range
crosses physical chunks, while metadata and packed 244-byte `ShaderParams` records (including the
12-byte shared entropy prefix) use `Queue::write_buffer`; no second full-codestream host `Vec` is
created.

Idle retention defaults to 32 MiB, 256 buffers total, and 32 buffers per exact key.
`WgpuSubmissionEngine::{buffer_pool_limits,set_buffer_pool_limits,clear_buffer_pool,buffer_pool_stats}`
expose limits, generation invalidation, hits/misses, idle/leased bytes and objects, and eviction
counters. An explicit clear invalidates outstanding generations without disrupting their GPU
work; those leases are destroyed after completion instead of re-entering the cache. A dropped
session or Future is also safe: the map callback retains every transient lease, then unmaps status
staging before returning allocations. The shared `MemoryBudget` still charges the complete active
logical job exactly once. Idle physical cache bytes are reported and bounded separately, rather
than being double-counted as active work.

`GpuPendingFrame::poll_complete` registers the latest supplied `Waker` and returns quickly while
status readback is pending. Native builds drive `Device::poll` on the backend's bounded completion
worker; poll admission is reserved before source consumption or queue submission, so saturation
is exposed in `PrefetchProgress` and can be retried without losing the source. Browser WebGPU uses
the polling/future API. There is no Tokio or async-std dependency.

The WGSL storage/uniform ABI and F64 output words are explicitly little-endian. Because the
`repr(C)` + `bytemuck::Pod` host structs use native endian, non-little-endian targets are rejected
at compile time rather than silently corrupting transport values.
