# GPU architecture

The canonical current capability table, full JPEG XL backlog, dependencies, and acceptance gates
are maintained in [`FULL_JPEG_XL_ROADMAP.md`](FULL_JPEG_XL_ROADMAP.md). This document describes the
backend architecture and invariants, not a broader support claim.

## Repository boundary

This repository is a standalone Cargo workspace. Production codec crates do not vendor, patch, or
call an upstream CPU codec. Published `jxl` and reference `cjxl`/`djxl` binaries are dev-only
correctness oracles.
The backend-neutral protocol, portable image formats, `wgpu` execution, decode facade, and
measurement harness can therefore be released independently.

The dependency graph is intentionally one-way:

```text
jxl_gpu_bitstream   jxl_gpu_protocol   jxl_gpu_formats
        |                   |                  |
        +-------------------+------------------+
                            v
                         jxl_wgpu
                         /      \
                        v        v
              jxl_wgpu_decode  jxl_wgpu_encode
                        \        /
                         harness
```

`jxl_gpu_protocol`, `jxl_gpu_bitstream`, and `jxl_gpu_formats` contain no `wgpu` or upstream-decoder
types. `jxl_wgpu` has no production dependency on `jxl`; reference codec crates and tools are used
only by tests and the harness.

## GPU-required codec path

Container/header parsing and packet ordering remain host orchestration. Pixel and coefficient work
for an accepted profile is submitted to the GPU:

```text
JPEG XL bytes -> bounded container/header parse -> GPU group decode
                                                -> GPU restoration/color/output
                                                |-> optional CPU readback
                                                `-> same-queue display/consumer

GPU image source -> GPU prediction/transform/quantization/tokenization
                 -> deterministic group packet assembly -> JPEG XL codestream/container
```

Before codestream inventory, `jxl_gpu_bitstream::ContainerStreamScanner` can now consume arbitrary
owned chunks without joining the complete transport. Apart from the inline reconstructed two-byte
signature, raw, `jxlc`, and in-order `jxlp` payloads are emitted as ranges over the caller's
`Arc<[u8]>`; version-1 fragments ahead of a gap coalesce input chunks into one retained payload
buffer per fragment under a separate logical-byte limit, then release it when its logical turn is
emitted. Auxiliary box start/chunk/end events retain the original compact, extended, or to-end
header bytes. No output is authoritative until the end-of-input event validates transport and
fragment completeness.

`CodestreamStreamScanner` observes that event stream without taking ownership of auxiliary boxes.
It copies only a separately limited image or current-frame metadata probe, parses the image header
and frame header/TOC at geometrically increasing probe sizes, emits `Arc<FrameInventory>` before
section bytes, and splits subsequent `StreamSlice` ranges at exact physical TOC boundaries. A
probe may contain a bounded section prefix; that tail keeps the probe allocation rather than being
copied again, while later section ranges retain caller/transport backing. Complete image/frame/TOC
state no longer requires a contiguous codestream. Both decoders now retain a checked table of
logically contiguous shared spans, and exact bounded GPU upload ranges may cross physical chunks.
Modular and VarDCT metadata bit reads are span-native. VarDCT uses the common generic entropy IR for
block-context maps and custom coefficient-order permutations, plans from logical length, and
initializes its still-required whole-codestream GPU buffer directly from spans without a second
host-sized `Vec`. Both mode engines and the selector accept the same public multi-span source.

`GpuDecoder::stream` is the ownership adapter between the two scanners and that source boundary.
It borrows `ContainerStreamEvent` values, retains only their codestream slices, collects emitted
image/frame inventories, and opens a session only after authoritative `End`. One growable permit
from a cloneable incremental-input budget accounts exact logical retained bytes across concurrent
streams. Admission precedes scanner mutation, so exhaustion is retryable with the same event.
Host-source admission is separate from GPU allocation admission because the source and its upload
are physically live at the same time. The permit moves with the source: Modular releases it after
queue submission, while a staged local-tree VarDCT path retains it across the LF status map until
the cursor-dependent HF submission. Dropping the stream, prepared session, or pending local-HF work
releases its ownership without waiting for unrelated GPU buffer callbacks.

`GpuDecoder::wgpu` is the sole stock high-level decode constructor. Contiguous `open` and event-fed
`stream` each inventory a codestream once before `WgpuDecodeEngine` reads `FrameEncoding` and moves
the validated codestream plus that inventory into either the Modular or VarDCT submission session.
The selector does not merge their
incompatible storage-binding layouts: pipeline caches and mode-specific state remain independent,
while device, queue, completion worker, and the aggregate byte budget are shared. The pending-frame
enum performs only lifetime operations (`submit_next`, blocking wait, and runtime-neutral poll), so
bounded VarDCT profile limitations do not become a common entropy or frame abstraction.

The entropy executor has a narrower shared boundary than either submission pipeline. Rust and WGSL
both define a 12-byte, four-byte-aligned `EntropyStreamParams` prefix for token start/end bounds and
the LZ77 ring mask. It is the first field of the 236-byte Modular parameter record and the 208-byte
VarDCT packet record. The shared entropy fragment consumes that prefix plus consumer-provided
storage access and LZ scratch-base functions. Geometry, prediction/output, and VarDCT
metadata/coefficient state retain separate suffixes and bindings. This preserves one checked
executor ABI while allowing the VarDCT pass-group consumer to decode nonzero coefficients without
turning coefficient placement into a common stream type. The fragment also owns exact-range
termination: it validates the ANS signature, rejects more than seven trailing bits or any nonzero
padding, and advances to the declared token end. Consumers with fixed fields after entropy retain
the narrower ANS finalizer and validate the complete enclosing section after reading their tail.
The HF pass-group consumer also applies the stream's quant-field and signed X/Y/B LF thresholds on
GPU. Its context-map and threshold words follow the entropy tables in one storage bundle. Raw
quantized LF planes and each pass group's LZ ring use non-overlapping slices of the reconstruction
buffer, preserving the portable eight-storage-buffer stage limit without a readback.

Every accepted stock Modular reconstruction specialization supports intra-group streaming.
A consumer-neutral entropy-window planner owns byte-range validation, four-byte packing, sentinel
space, group batching, logical-to-upload rebasing, and the first/final flags; codec consumers retain
their own resume records and dispatch ordering. The Modular engine, VarDCT AC pass-group consumer,
and combined/global-tree plus staged local-tree VarDCT packet consumers use it in production.
The common inventory already resolves progressive-DC reads to exact earlier LF producer frames in
the four normative slots for both contiguous and incremental input; rendering those producers into
resident LF slots and consuming them without readback remains `ENT-D02` work. The planner resolves one upload cap from the
caller policy, storage binding limit, and shared per-frame byte budget. A group that exceeds it is
divided into ordered core ranges with 16-byte
backward/forward overlap; a dispatch finishes the current output token before yielding, so no
partial Prefix/ANS/hybrid/LZ token becomes host-visible state. The 236-byte record maps the one
group-relative cursor into the current physical upload and identifies first/final segments. A
16-byte-aligned tail in each reconstruction lane retains the bit cursor, ANS state, LZ77 copy and
last-value state, consumer progress, and a sticky error while the descriptor-sized history ring
remains resident in the same lane. Channel-fixed Gradient uses 32 bytes; generic MA uses 48 bytes;
Weighted/SelfCorrecting MA uses 112 bytes for its gradient history, four true errors, and twelve
subprediction-error accumulators. Channel-local predictor state resets at a channel boundary.
Every segment reuses one stream buffer and lane through queue ordering. Only the final segment
performs exact entropy termination and only the last frame batch maps the aggregate group-status
buffer. VarDCT AC retains an independent 464-byte aligned record containing the common state,
nested coefficient progress, sink error, and 96-word nonzero-neighbour grid. Its stream and
parameter buffers are rewritten only between ordered queue submissions; the final resident render
and aggregate status map follow the last window. A staged local-tree LF packet uses a separate
240-byte aligned parameter record and a 64-byte generic or 128-byte SelfCorrecting state ABI within
its LF reconstruction allocation. Current local-tree planning reserves 128 bytes because the HF
tree is not known until after the LF status map. The record retains the common entropy/LZ state, decoded-sample
progress, sticky failure, previous-gradient state, and every weighted-predictor accumulator. Only
the final LF segment checks exact termination and one aggregate map exposes the LF end cursors.
After the aggregate LF cursor map, the host packs each HF descriptor and builds exact HF ranges
against the same retained codestream and shared packet upload. HF reuses the packet state
sequentially, persists progress across its two correlation maps, strategy/quantizer plane, and
sharpness plane, and performs fixed-tail validation only on its final segment. The last HF command
shares the first downstream queue submission. Combined single-entry and shared-global-tree packets
need no cursor map: their known range starts at `section_bits.x`, including extra precision and the
LF local header. Five words in the same state retain LF/HF phase, both decoded counts, first-block
count, and extra precision across the three LF and four HF channels. Only the final segment validates
the packet tail, and its command shares the first downstream queue submission.

For VarDCT, the caller/device cap is first evaluated against exact packet, AC, render, validation,
and output bytes. If that frame would exceed the shared budget's total capacity, the planner
evaluates the 40-byte minimum and searches four-byte-aligned caps for the largest tested layout that
fits. The resolved cap is recorded in `VarDctDecodeMemoryStats`; an impossible minimum layout is a
typed pre-submission error. Planning deliberately does not use momentary live headroom: concurrent
jobs race only at non-blocking `MemoryBudget` admission and report retryable backpressure. A staged
local tree reuses its already planned packet upload for host-discovered HF entropy; only a larger
HF metadata allocation requests its exact difference after the LF cursor map.

For the bounded stock VarDCT profile, restoration remains resident in the downstream command
buffer. Global-tree streams execute that buffer in the packet submission. Streams with local
per-substream MA trees first submit one or more bounded LF entropy windows, map only aggregate end cursors, host-pack the
cursor-dependent HF descriptors, and then submit one or more bounded HF windows plus the downstream
buffer. The inverse
transform dispatches each independently bounded LF-group artifact into shared padded resident XYB
planes. Group-local LF and correlation data first scatter into full-image atlases; adaptive LF
smoothing runs once over that atlas, or the standard skip flag copies it resident-to-resident.
An optional fused Gaborish dispatch reads only the actual image extent with mirrored image-edge
sampling. When EPF is enabled, one linear dispatch per LF group lowers per-transform `hf_mul` and
per-block sharpness into disjoint rectangles of the full-image inverse-sigma plane, then the
signaled EPF0/EPF1/EPF2 sequence and Gaborish advance one shared restoration cursor between the
resident image and one three-plane scratch set. The output packer consumes whichever set is current
after the final pass. Global-tree frames copy every LF group's packet and artifact status plus all
pass-group records into one mapped staging buffer. Local-tree frames map that buffer once for LF
cursors and once for final validation. No coefficient or pixel crosses the host. The
three F32 scratch planes, inverse-sigma plane, and exact 80-byte restoration uniforms are included
in the same backend byte admission as entropy, transform, output, and aggregate validation storage.

An unsupported profile rejects during capability negotiation or header validation. A CPU oracle
may decode the result in tests, but oracle code cannot be reached from a production encode/decode
session. The initial interoperable target is a deliberately narrow single-group lossless Modular
profile; capabilities expand only with valid-bitstream round trips and conformance evidence.

Before that capability gate, the Modular frontend now lowers every RCT, Palette, and Squeeze wire
record into a bounded transform plan. Meta-application computes the entropy-visible channel list in
codestream order, including the default Squeeze expansion, odd average/residual extents, accumulated
shifts, Palette/delta storage at the meta prefix, and transforms that target existing meta channels.
The plan contains no host pixel data. A checked packed-channel layout proves its complete sample
address space fits portable WGSL `u32`. The first resident inverse primitive reconstructs one
horizontal or vertical Squeeze parameter from pairwise non-overlapping average, residual, and output
views of one read-write storage arena. One invocation owns a complete row or column because the
previous reconstructed odd sample is a normative dependency; linear 1/32/64/128/256-lane variants
parallelize independent lines. Portable two-word signed arithmetic preserves the scalar `i64`
smooth-tendency calculation over the complete `i32` input domain. The stock executor still rejects
non-direct topologies until generalized entropy output targets the inverse arena and Palette/RCT
passes exist.
Inverse lowering does not retain one full channel table per transform. It starts from the final
entropy topology, reverses each Squeeze parameter and transform, invokes the lowering callback with
only the current/restored pair, and verifies that the source topology is recovered exactly. A
cumulative topology-work budget bounds repeated vector insertion/removal independently of transform
and final-channel count limits.
For Squeeze-only stacks, lowering now continues into an executable lifetime schedule. Entropy-visible
planes begin as tightly packed ranges in one storage arena. Before each channel dispatch a best-fit
allocator reserves a non-overlapping restored range; after that ordered dispatch its average and
residual ranges become reusable and adjacent holes are coalesced. In-place and tail-appended residual
orders use the same logical-plane table. A composed two-parameter case is executed as three GPU jobs
inside one command encoder with only the final readback, while the 13-parameter progressive-DC root
is preflighted as 37 jobs. The planner returns a typed address-space, topology-state, free-list, or
unsupported-transform error before backend allocation.

## Protocol execution

For capture/replay tests and codec frontends, `jxl_gpu_protocol` describes planes, resources,
decoded group payloads, VarDCT packets, operations, output descriptors, changed regions, and
transactional frame sessions. `RenderBackend` creates a `FrameSession`; `WgpuBackend` validates and
lowers the protocol into a bounded batch:

```text
RenderPlan
    -> validation and capability negotiation
    -> lifetime-based resident-buffer allocation
    -> kernel selection and safe fusion
    -> one command encoder / ordered queue submission
    -> GPU-resident output or explicit mapped readback
```

Unsupported nodes reject before output becomes authoritative, and the backend never returns a
partially valid frame as success.

## Portable output formats

`jxl_gpu_formats` separates the independent properties of an image:

- color model and channel semantics;
- component/sample representation and bit depth;
- interleaved, planar, semi-planar, or packed plane organization;
- chroma subsampling and siting;
- matrix coefficients, primaries, transfer, and full/limited range;
- checked per-plane offset, extent, row stride, and logical byte size.

This models all relevant pitch-linear NVIDIA VPI 4.1 predefined formats as well as common planar
and higher-bit-depth video formats. CUDA-specific block-linear storage is deliberately out of
scope: WebGPU does not expose that physical layout, and this project does not pretend that a
portable buffer is CUDA block-linear memory.

Every layout calculation is checked for overflow, overlapping planes, undersized strides, and
subsampling constraints. Odd extents use explicit ceil division where the format permits them;
formats such as packed 4:2:2 can require an even width.

F64 output from the stock Gray8 decoder is an explicit precision contract, not an inferred format
upgrade. `F64OutputPolicy` selects native shader arithmetic, native-or-exact-F32-widening, or the
portable exact-widening path. `NativeRequired` rejects a logical device without enabled
`SHADER_F64`; the compatibility path produces valid binary64 storage but does not claim that its
normalization was evaluated with F64 arithmetic.

## GPU output and display

There are three distinct terminal paths:

1. CPU output: map a readback buffer after a submission token completes.
2. GPU buffer output: return reference-counted pitch-linear plane buffers immediately after queue
   submission.
3. Display texture output: encode conversion/copy commands after decode work on the same queue and
   return an RGBA texture suitable for sampling, rendering, or copying to a surface texture.

Queue ordering is the synchronization primitive for path 2 and 3. No host wait is required before
encoding a dependent command. Wgpu command ownership keeps referenced allocations physically
alive. The crate's byte accounting is narrower: it is retained by `GpuBufferLease` and
engine-owned release guards, not by raw `wgpu::Buffer` clones made by callers. Explicit completion
is needed for CPU access and for application-defined reuse across unrelated queues/devices.

The stock decoder exposes this early path through a deliberately separate
`UnvalidatedGpuImageFrame`. Its queue token, requested layout, and permit-bearing buffers can feed
`DisplayPipeline`, `ImageReadbackPipeline`, or a custom same-queue command, but authoritative frame
metadata and changed regions are withheld until status validation succeeds. Transport completion
does not imply codec validation. If validation later fails, queued consumers cannot be rolled back
and every derived texture or byte buffer must be discarded.

For already-produced authoritative frames, `ImageReadbackPipeline::submit_frames` combines all
outputs into one bounded staging allocation, command buffer, queue submission, map callback, and
runtime-neutral completion object. This aggregates only transport; it does not merge the codec
submissions that produced those frames or claim true codec batching.

Generic color packing has an explicit two-sided source contract: `OutputDesc::color_encoding`
declares the render-plan signal and `ImageOutputRequest::source_encoding` must match it. The
portable shader converts BT.709 primaries between Linear, sRGB, and BT.709 transfer functions via
linear light before applying RGB or YCbCr output packing. Undefined metadata, unsupported transfer
functions, and wide/HDR primaries fail before the output dispatch instead of relabelling samples.

`DisplayPipeline` caches pipelines and bind-group layouts by source format and color conversion.
Direct RGBA copies are used when storage and texture layouts agree. Planar, semi-planar, and packed
YUV use a shader conversion into an RGBA display texture. WebGPU has no portable native NV12
multi-plane texture, so the public contract remains explicit plane buffers rather than claiming a
native multi-plane texture object.

## Animation and concurrency

Sync and async animation frontends drive one GPU codec state machine. Stream metadata carries the
timebase, loop count, and whether frame timecodes exist. A frame carries its index, exact duration,
cumulative presentation-start ticks, optional bitstream timecode, and composed output.
Reference-frame and blend dependencies are explicit session state. The session's count slot is
occupied from submission through its ordered pending value and then by the returned
`GpuFrameLease`; dropping that lease releases the slot. This slot count is an admission bound, not
proof that arbitrary downstream wgpu commands have completed. Engine-owned release guards and
tracked buffer leases separately retain byte reservations.

Decode submission and completion are separate contracts. `GpuSubmissionSession::submit_next`
records queue work and returns an owned `GpuPendingFrame`; native `wait` and runtime-neutral
`poll_complete` validate that exact submission later. `GpuDecodeSession` retains pending frames in
submission order and can prefetch several frames before waiting for the front. Its progress value
distinguishes target depth, explicit stream end, frame-slot pressure, shared memory pressure, and
bounded poll-worker pressure. Completion order cannot reorder presentation metadata or outputs.

The synchronous API advances and, when requested, waits for one GPU frame at a time. The
runtime-neutral async API is expressed with `Future`, `Poll`, `Context`, and `Waker`; it does not
depend on Tokio, async-std, or a particular reactor. Completion callbacks wake the task.

GPU animation output uses a bounded frame-slot count plus a separate byte-weighted memory budget.
A queued submission or returned `GpuFrameLease` occupies one slot. Engine work holds its own
release guards, and output allocations hold tracked `GpuBufferLease` permits; only the latter two
participate in byte accounting. Raw wgpu handle clones cannot be observed by that budget.

Clones of one stock decoder engine share its compiled pipelines. Modular and VarDCT sessions
selected by one `WgpuDecodeEngine` also share the backend-wide byte budget. Decoders built from one backend
share the device, queue, bounded poll worker, and byte-weighted transient memory admission while
retaining separate frame state. Its bounded exact-match pool reuses lookup, reconstruction,
status/readback, and POD-parameter buffers only after the completion lifetime is safe; raw
codestream and caller-owned output buffers are never pooled. Encoder contexts created with
`WgpuContext::from_backend` share the same bounded poll worker and memory budget, while standalone
contexts own one bounded worker. The harness reports measured still-image latency and throughput
for isolated, sequential, burst, and persistent-worker workloads; stock-codec animation remains a
typed unsupported result.

## Safety invariants

- All sizes, offsets, strides, dispatch counts, and allocation sums use checked arithmetic.
- A plane descriptor cannot address outside its allocation or overlap another writable plane.
- A protocol resource revision increases monotonically and a final submission requires complete
  latest group revisions.
- GPU-resident public output is never returned from an internal recyclable buffer.
- CPU mapping waits for exactly the associated submission; GPU consumers use ordered queue work.
- Device/resource-limit errors remain typed and never downgrade precision silently.
- Production crates deny unsafe Rust.
