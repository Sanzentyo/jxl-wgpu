# jxl_wgpu_decode

GPU-required JPEG XL decode orchestration. Production execution uses the stock WGSL submission
engine and has no dependency on the published `jxl` decoder.

## Executable profile

The stock `WgpuSubmissionEngine` implements one deliberately narrow end-to-end profile:

- a standard, reference-decodable JPEG XL container with one `jwgp` acceleration-index box;
- one final still frame, 8-bit grayscale lossless Modular, one group (at most 256x256), one pass,
  fixed Gradient predictor, prefix entropy, and the profile's distance-one zero-run coding;
- no transforms, restoration filters, extra channels, or animation references.

`jwgp` contains no pixels, residuals, or entropy events. It contains the extent, actual `jxlc`
token-bit range, and canonical prefix tables, all bound to the complete contiguous codestream by
length and SHA-256. Before submission the decoder validates the fixed standard JPEG XL image/frame
envelope and TOC, regenerates the DC-global/context-map/four-prefix-tree group prefix, compares it
bit-for-bit with `jxlc`, and proves that its exact end is the indexed token offset. The private box
therefore accelerates bounded parsing but cannot redefine the standard codestream.

The compute shader reads prefix/hybrid tokens from the actual codestream buffer, validates every
bit and output bound, expands only the profile's constrained zero runs, unpacks signed residuals,
and performs Gradient reconstruction. A mapped four-word status buffer is checked before a frame
is reported as successful. No reconstructed sample is produced on the CPU.

Containers without a valid `jwgp` index and raw/generic JPEG XL codestreams return typed
`UnsupportedProfile`/`AccelerationIndex` errors. VarDCT, adaptive predictors, multiple
groups/passes, extra channels, patches, splines, noise, and reference-frame animation remain typed
unsupported profiles.

## GPU output formats

`GpuOutputRequest` always carries a concrete `jxl_gpu_formats::PixelFormat`; the request is never
ignored. Construction is intentionally breaking and explicit: `GpuOutputRequest::color` accepts
only classified color formats, while `GpuOutputRequest::numeric` requires a
`NumericSampleMapping`. The current Gray8 kernel supports:

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

`classify_pixel_format` is the sole storage classifier. `NormalizedGray8` maps code 0..255 across
the full unsigned range, the nonnegative half of a signed type (0..MAX), or float 0..1, and
replicates the result for 2S16/2F32. Integer endpoints and rounding are computed without overflow;
F32 bits are formed deterministically with integer arithmetic rather than backend-relaxed
division.

F64 precision is never inferred silently. `NormalizedGray8F64` requires an `F64OutputPolicy`:
`NativeRequired` rejects devices without enabled `SHADER_F64`,
`NativeOrExactF32Widening` explicitly permits a compatibility fallback, and
`ExactF32Widening` always constructs binary64 as the exact widening of the correctly-rounded F32
normalization. The compatibility result is valid IEEE-754 F64 storage, but is not a precise F64
division. When `WgpuBackend::native_f64_enabled()` is true, the native path is lazily compiled and
evaluates `f64(gray) / 255.0` in WGSL. `WgpuDecodeSession::f64_output_path` reports the resolved
path. The returned `GpuImageFrame` owns pitch-linear GPU buffers. CPU readback occurs only when the
application explicitly stages one.

## Public flow and bounds

1. Construct `GpuDecoder::wgpu` around an application's existing `WgpuBackend`.
2. Call `open` with encoded bytes and a `GpuOutputRequest`.
3. Fill the ordered GPU queue with `prefetch`, `poll_prefetch`, or the runtime-neutral
   `prefetch_async` future. Prefetch submits work and never waits for frame completion.
4. Optionally borrow `pending_frames`/`front_pending_frame` and call the stock
   `WgpuPendingFrame::unvalidated_gpu_frame` to enqueue same-device, same-queue display, readback,
   or custom GPU work before mapped-status validation completes.
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

The CPU/WGSL parameter ABI is a checked 96-byte `repr(C)` POD. It carries four plane
offset/stride pairs, exact channel/order/depth/range/transfer codes, the resolved numeric mapping,
and the shader-visible logical size. Codestream uploads are rounded to
four bytes and include an additional zero sentinel word for the shader's bounded cross-word peek.
Prefix lookup (128 KiB), reconstruction, output, status/readback, codestream, and parameter sizes
are checked with overflow detection against both storage-binding and device-buffer limits. The
requested `max_frame_slots` multiplied by the complete per-frame allocation estimate must remain
within a 64 MiB session exposure. `WgpuDecodeSession::memory_stats` reports the complete per-frame
bytes, the output-lease and transient portions, and the maximum exposure implied by the requested
window. Concurrent jobs opened through an engine or its clones use the `WgpuBackend`'s shared
transient memory budget by default, so decode, encode, and generic readback apply one aggregate
admission bound. `WgpuSubmissionEngine::with_memory_budget` instead accepts an explicit cloneable
`MemoryBudget` for applications that intentionally define another sharing group. Admission is
non-blocking and memory pressure returns a typed, retryable error.

Transient codestream/lookup/reconstruction/status bytes remain reserved until the status map has
completed. Output bytes are carried by `GpuBufferLease`, so explicitly cloning that lease extends
the same reservation and dropping the decode session cannot release it prematurely. GPU frame and
output containers are intentionally not cloneable. A raw `wgpu::Buffer` cloned through
`GpuBufferLease::as_wgpu_buffer()` remains valid wgpu ownership but is outside the byte budget; the
reservation returns after the final tracked lease is dropped. The current stock profile has
exactly one visible frame; the same ownership contract applies to future animation frames.

Repeated small and sequential decodes reuse a decoder-local, bounded cache for prefix lookup,
reconstruction, status, status-staging, and POD parameter buffers (plus the native-F64 dummy when
needed). A cache hit requires the exact allocation size, usage flags, and ABI alignment. The raw
JPEG XL codestream and caller-owned output are never admitted to this pool. Codestream upload reads
aligned spans directly from the shared input storage, while lookup and the 96-byte `ShaderParams`
POD use `Queue::write_buffer`; no second full-codestream host `Vec` is created.

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
