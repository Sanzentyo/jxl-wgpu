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
ignored. The current Gray8 kernel supports:

- exact unsigned 8-bit, one-channel `NonColor/X` output;
- 8-bit luma output;
- native 8-bit planar or semiplanar YCbCr at the representable 4:4:4, 4:2:2, or 4:2:0 sampling
  ratios, including NV12/NV21/NV16/NV61/NV24/NV42 and I420/I422/I444 descriptors.

YCbCr requests require a defined color specification. Linear, sRGB/sYCC, and BT.709/BT.2020
transfer conversion plus full/limited range coding run in WGSL; grayscale chroma is written
directly as neutral code values. Packed RGB(A), packed 4:2:2, PQ, undefined transfer semantics,
and greater-than-8-bit output are explicitly rejected by `UnsupportedOutputFormat` in this first
profile. The returned `GpuImageFrame` owns pitch-linear GPU buffers; CPU readback is never
performed unless the application explicitly stages one.

## Public flow and bounds

1. Construct `GpuDecoder::wgpu` around an application's existing `WgpuBackend`.
2. Call `open` with encoded bytes and a `GpuOutputRequest`.
3. Consume `GpuDecodeSession::next_frame` synchronously or use
   `next_frame_async`/`poll_next_frame` through `std::future::Future`.
4. Retain each `GpuFrameLease` only while its GPU resource is needed. The lease holds an
   `InFlightPermit`; dropping it wakes a pending submission.

The CPU/WGSL parameter ABI is a checked 64-byte `repr(C)` POD. Codestream uploads are rounded to
four bytes and include an additional zero sentinel word for the shader's bounded cross-word peek.
Prefix lookup (128 KiB), reconstruction, output, status/readback, codestream, and parameter sizes
are checked with overflow detection against both storage-binding and device-buffer limits. The
requested `max_in_flight` multiplied by the complete per-frame allocation estimate must remain
within a 64 MiB session reservation. `WgpuDecodeSession::memory_stats` reports the per-frame,
in-flight, and reserved byte counts. Concurrent sessions opened through an engine or its clones
also share a 256 MiB aggregate reservation by default; `WgpuSubmissionEngine::with_memory_budget`
sets an explicit bound, and dropping a session releases its reservation. The current stock profile
has exactly one visible frame; the same accounting contract is retained for future animation
support.

An engine's `poll_next_frame` registers the supplied `Waker` and returns quickly while status
readback is pending. Native builds drive `Device::poll` on a completion thread; browser WebGPU must
use the polling/future API. There is no Tokio or async-std dependency.
