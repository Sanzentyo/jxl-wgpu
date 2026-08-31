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

An unsupported profile rejects during capability negotiation or header validation. A CPU oracle
may decode the result in tests, but oracle code cannot be reached from a production encode/decode
session. The initial interoperable target is a deliberately narrow single-group lossless Modular
profile; capabilities expand only with valid-bitstream round trips and conformance evidence.

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

Clones of one stock decoder engine share its compiled pipelines. Decoders built from one backend
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
