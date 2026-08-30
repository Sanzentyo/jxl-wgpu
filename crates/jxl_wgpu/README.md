# jxl_wgpu

`jxl_wgpu` is the portable, GPU-required render backend for this standalone JPEG XL workspace. It
implements `jxl_gpu_protocol::RenderBackend` with `wgpu` 30 and exposes its concrete
`WgpuBackend`/`WgpuFrameSession` APIs for GPU-resident image output, same-queue presentation, and
explicit mapped readback.

Host code validates plans and packets, records command buffers, and resolves completion. Supported
pixel, coefficient, restoration, color, packing, and display work executes in WGSL. Unsupported
operations, layouts, precision contracts, and device limits return typed errors before an output
is authoritative.

## Backend creation

Request a device owned by the backend:

```rust,no_run
use jxl_wgpu::{WgpuBackend, WgpuBackendConfig};

# async fn create() -> Result<(), Box<dyn std::error::Error>> {
let backend = WgpuBackend::request_default(WgpuBackendConfig::default()).await?;
let device = backend.device();
let queue = backend.queue();
# let _ = (device, queue);
# Ok(())
# }
```

An application that already owns a renderer can construct `WgpuBackend::from_device` with its
`wgpu::Device`, `wgpu::Queue`, `wgpu::AdapterInfo`, and `WgpuBackendConfig`. Conversion, display,
and later renderer work can then share one queue and use queue ordering after the producer exposes
its GPU buffer handle. The stock decoder can expose an explicitly `UnvalidatedGpuImageFrame` from
its ordered pending queue before its small validation mapping completes; validated metadata is
still returned only after that mapping succeeds.

`WgpuBackendConfig::shader_f64_policy` defaults to `ShaderF64Policy::Auto`: backend-owned device
creation requests `wgpu::Features::SHADER_F64` exactly when the adapter advertises it. `Disabled`
never selects native double arithmetic, while `Require` rejects an unavailable feature instead of
downgrading. For an application-supplied device, the feature must already have been requested;
`WgpuBackend::native_f64_enabled` reports the enabled logical-device capability after applying the
policy. In wgpu 30 this is a native Vulkan capability. The decoder's native WGSL is validated under
Naga's `FLOAT64` capability and compiled lazily only for an F64 request that resolves to the native
path.

On native targets, `WgpuBackend` implements the canonical
`jxl_gpu_protocol::RenderBackend` factory and returns `Box<dyn FrameSession>`. The concrete
`WgpuBackend::create_session` method returns `WgpuFrameSession` and is also available when callers
need GPU-buffer, generic-format, display, or concrete instrumentation APIs.

## Render-plan execution

A frontend supplies a validated `RenderPlan` and `FrameSessionDesc`, then streams revisioned
resources and decoded group packets into a frame session:

```rust,ignore
use std::sync::Arc;

use jxl_gpu_protocol::{FrameSession, RenderBackend, RenderIntent};

let mut session = backend.create_frame_session(&frame_desc, Arc::new(plan))?;
session.update_resource(resource)?;
session.enqueue(group)?;
let token = session.submit(RenderIntent::Final)?;
let frame = session.wait(token)?;
```

`FrameSession::submit` records and submits GPU work without completing a mapped output on the
calling thread. `FrameSession::wait` resolves that submission into protocol-owned bytes. The
concrete session adds GPU-resident terminal paths:

```rust,ignore
let frame = session.submit_gpu(RenderIntent::Final)?;
let output = &frame.outputs[0];

// Once submit_gpu has returned the handle, a dependent command on the same queue can use it.
consumer.copy_buffer_to_buffer(
    output.buffer.as_wgpu_buffer(),
    0,
    &destination,
    0,
    output.logical_size,
);
backend.queue().submit([consumer.finish()]);
```

Each `GpuOutputBuffer` carries its output ID, extent, sample type, channel count, layout, meaningful
byte length, and a cloneable `GpuBufferLease` around the `wgpu::Buffer`. The lease remains valid
after the frame session is dropped and retains the shared memory reservation. `wait_gpu` is only
needed when native host code explicitly requires completion; dependent commands on the same queue
do not need it. `GpuOutputBuffer`, `GpuFrame`, `GpuImageOutput`, and `GpuImageFrame` are
intentionally not cloneable; clone the specific buffer lease whose accounted lifetime must be
extended.

`GpuBufferLease::as_wgpu_buffer()` is the explicit raw interoperability boundary. A
`wgpu::Buffer` handle cloned from that borrow is valid wgpu ownership, but it is outside this
crate's byte accounting and does not retain the lease's `MemoryPermit`. Safe Rust cannot attach a
permit to a raw handle cloned by external code, so retain a lease clone for as long as budget
tracking is required.

## Generic pitch-linear formats

`WgpuFrameSession::submit_gpu_image` converts the final F32 RGB planes directly into a checked
`jxl_gpu_formats::PixelFormat`. The plan's `OutputDesc::color_encoding` and the request's
`source_encoding` must agree. BT.709 primaries with Linear, sRGB, or BT.709 transfer functions are
converted as source EOTF -> linear light -> target OETF; undefined metadata, wide-gamut primaries,
and HDR transfer functions return `Error::Unsupported`. `submit_image` plus `wait_image` provides
the corresponding mapped transport. Supported layout families include:

- unsigned luma at 8 or 16 storage bits;
- planar and semiplanar YCbCr 4:4:4, 4:2:2, and 4:2:0 at 8/10/12/16 bits, including NV12, NV21,
  NV16, NV61, NV24, NV42, P010, P012, and P016;
- packed YUYV and UYVY 4:2:2; and
- planar or interleaved RGB, BGR, RGBA, and BGRA8.

```rust,ignore
use jxl_gpu_protocol::RenderIntent;
use jxl_wgpu::{
    ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, ImageOutputRequest, PixelFormat,
};

let color = ColorSpecification::Defined(ColorSpec::bt709(
    ColorRange::Limited,
    ChromaLocation2d::CENTER,
));
let request = ImageOutputRequest::new(
    jxl_wgpu::RgbColorEncoding::BT709,
    PixelFormat::nv12(color),
);
let nv12 = session.submit_gpu_image(RenderIntent::Final, request)?;
```

Every output has an `ImageLayout` with explicit plane offsets, row strides, extents, sampling, and
logical byte size. Odd chroma extents use checked ceil division. Centered siting averages the valid
subsampling footprint; cosited output uses the top-left luma position. Numeric representation,
matrix, range, and siting are never inferred from an ambiguous descriptor.

These contracts describe portable pitch-linear buffers. Vendor block-linear memory and native
multi-plane texture handles are outside WebGPU's portable buffer model.

Non-color numeric formats deliberately require an application-defined numeric-to-color mapping.
`DisplayPipeline::encode_numeric_image`/`submit_numeric_image` accepts all ten numeric VPI
pitch-linear formats only with a complete `NumericDisplayContract`: stored signed/unsigned/float
kind, affine `x * scale + bias`, NaN/infinity policy, unit clamp, one/two-channel visualization,
and Linear or sRGB transfer are explicit. F64 reports whether it used native shader f64 or the
portable round-to-f32 path in `DisplayTexture::numeric_precision`; native f64 is selected whenever
the logical device has `SHADER_F64` enabled and the supplied F64 plane offset/row pitch are
naturally eight-byte aligned.

## Same-queue display

`DisplayPipeline` turns a `GpuOutputBuffer` or `GpuImageOutput` into an `Rgba8Unorm` storage texture
with texture-binding, render-attachment, and copy usages. The convenience submission uses the
backend's queue and returns without a host wait:

```rust,ignore
use jxl_wgpu::{DisplayPipeline, DisplayTextureDescriptor};

let display = DisplayPipeline::new(&backend);
let submitted = display.submit_image(
    &nv12.outputs[0],
    DisplayTextureDescriptor::default(),
)?;
renderer.sample(submitted.texture.view());
```

The source buffer and returned texture are retained by their GPU handles while commands are in
flight. The pipeline cache is keyed by the complete source-format and color-conversion contract.
The compute path accepts arbitrary valid pitch-linear row strides. The direct RGBA8 copy path
enforces WebGPU's 256-byte multi-row pitch requirement.

Callers that already own a command encoder can use `encode_rgb`, `encode_image`,
`encode_numeric_image`, or `encode_rgba8_copy`, then submit all work as one application-defined
batch.

`encode_unvalidated_image` and `submit_unvalidated_image` accept only the visibly distinct
`UnvalidatedGpuImageOutput` type. They preserve same-queue ordering without a host wait. If the
codec subsequently rejects its mapped status, already-submitted display work is irreversible and
the application must discard the texture. Numeric outputs use the equally explicit
`encode_unvalidated_numeric_image`/`submit_unvalidated_numeric_image` methods and still require a
`NumericDisplayContract`.

## Aggregate CPU readback

`ImageReadbackPipeline::submit_frames` copies every output across multiple already-produced
`GpuImageFrame` values into one bounded staging buffer. All buffer copies are recorded in one
command buffer, followed by one queue submission, one mapping callback, and one completion future
or native wait. Frame/output order, tokens, changed regions, layouts, and logical byte boundaries
are preserved. Each source region and staging offset is independently padded to WebGPU's four-byte
buffer-copy alignment; returned vectors contain only each layout's logical bytes. This batches the
explicit transport only—the codec submissions that produced the GPU frames remain distinct.

```rust,ignore
use jxl_wgpu::ImageReadbackPipeline;

let frames = [first_gpu_frame, second_gpu_frame];
let pending = ImageReadbackPipeline::new(&backend).submit_frames(&frames)?;
let planned = pending.stats();

// Runtime-neutral asynchronous completion:
let result = pending.await?;
# let _ = (planned, result);
```

`ImageReadbackBatchSubmission` implements `std::future::Future` directly and has no
executor-specific dependency. Native callers may use `pending.wait()` instead. Browser WebGPU uses
the future path. `submit` remains the explicit single-frame convenience and returns an
`ImageReadbackSubmission`/`ImageReadbackResult` instead of a one-element vector.
`submit_unvalidated` is the corresponding explicit early-handoff path; its distinct result omits
changed regions and remains non-authoritative even after transport completes, until codec
validation separately succeeds.

`ImageReadbackStats` reports frame count, output count, logical bytes, exact padded staging bytes,
and padding bytes. `ImageReadbackLimits` sets both the per-submission staging limit and the
aggregate bytes admitted across concurrently live submissions. Admission reserves the entire
single staging allocation atomically, is non-blocking, and returns a typed `MemoryBackpressure`
error. Pipeline clones share one byte budget. An in-flight submission and its mapping callback
retain the memory permit and every source `GpuBufferLease`; abandoning the future is safe and
releases those resources only after GPU completion.

For protocol `submit`/`wait`, `DirectReadbackPolicy` selects automatic direct mapping on eligible
unified-memory adapters, an explicit storage-to-staging copy, or required direct mapping. Requiring
direct mapping returns a typed error when `MAPPABLE_PRIMARY_BUFFERS` is unavailable.

## Memory policy and instrumentation

`WgpuBackendConfig::memory` contains five independent checked limits:

- `max_resident_bytes` for physical plane slots;
- `max_scratch_bytes` for simultaneously live intermediate planes;
- `max_transient_bytes` for one submission's uniforms, uploads, packed outputs, and staging; and
- `max_in_flight_transient_bytes` for byte-weighted admission across live frame-session,
  readback, decoder, and encoder jobs sharing the backend; and
- `max_cached_buffer_bytes` for idle reusable buffers.

The planner computes plane lifetimes, reuses physical slots only across disjoint live ranges, and
validates both device buffer and storage-binding limits. The shared buffer pool leases by exact size
and usage, never returns one allocation to two live submissions, and exposes hits, misses,
evictions, rejected leases, and idle bytes through `buffer_pool_stats()`.

`WgpuSubmissionStats` reports planned, actual, and fused dispatch counts plus resident and transient
bytes. `WgpuFrameSession::pending_transient_bytes()` reports the checked sum associated with tokens
that have not been waited. Public GPU outputs remain caller-owned and are not recycled through the
internal pool. `MemoryBudget`/`MemoryPermit` are runtime-independent: sync and async entry points
use the same atomic, non-blocking admission path, and clones release one reservation only after the
last owner is dropped. All `WgpuFrameSession` submission modes preflight and reserve their complete
explicit transient allocation before creating submission-owned buffers or entering the queue.
GPU-only output buffers carry clones of that same reservation—sibling outputs and buffer-lease
clones do not reserve again, and their `reserved_bytes()` values must not be summed.
Conservatively, the full submission reservation remains charged until GPU completion, its pending
token is consumed or dropped, and the final caller-visible buffer lease is dropped. Raw
`wgpu::Buffer` clones are not observable by this accounting.

All copy sizes are four-byte aligned, all offsets/strides/allocation sums use checked arithmetic,
and values consumed as WGSL indices must fit `u32`. Rust uniform/storage records use `#[repr(C)]`
plus `bytemuck::Pod`; tests pin their field order, size, alignment, and structured-array stride.
The optimized DCT8 kernel uses 1,536 bytes of workgroup memory and the special-transform kernel
uses 2,304 bytes. Large rectangular DCTs use two globally accounted scratch buffers instead of
transform-sized workgroup arrays. The complete table is in
[`docs/WGSL_MEMORY.md`](../../docs/WGSL_MEMORY.md).

## Implemented render operations

The current planner and scheduler execute these protocol stages:

- copy and integer-Modular-to-F32 conversion;
- horizontal/vertical chroma reconstruction and fused 2-D chroma reconstruction;
- per-plane and fused three-plane Gaborish;
- EPF passes 0, 1, and 2 with constant or supplied sigma;
- 2x, 4x, and 8x image upsampling with validated weights;
- all 27 JPEG XL VarDCT strategies: square and rectangular DCTs through 256x256, Hornuss,
  hierarchical DCT2, DCT4 variants, and all four AFV orientations, with GPU dequantization, color
  correlation, LF-grid reinterpretation, and inverse transform;
- stream-defined inverse XYB-to-linear-RGB and Linear, sRGB, BT.709, Gamma, PQ, and HLG transfer
  functions;
- all JPEG XL frame and patch blend modes for straight or associated alpha, including alpha-channel
  composition and the no-alpha fallbacks;
- exact partial-frame extension/cropping to the animation canvas with an optional reference slot;
- YCbCr-to-RGB components, alpha premultiplication, sample conversion, orientation, Save packing,
  and generic image packing; and
- direct generic-image and RGBA display conversion.

The backend validates precision contracts, node arity, plane types/extents/strides, resource
revisions, transform coverage, dispatch dimensions, shader-visible addresses, and device limits
before submission. Unsupported render operations, explicit streaming execution, invalid color
semantics, and unavailable device features return
`Error::Unsupported` or another specific resource/payload error.

## Platform notes

Native builds expose both trait-object and concrete frame-session paths. Browser WebGPU resources
are single-threaded, so callers use `WgpuBackend::create_session` and the concrete session API.
GPU-resident output and same-queue display remain non-blocking there; host mapping uses
`ImageReadbackSubmission` or `ImageReadbackBatchSubmission` as a future.

For architecture, performance methodology, and ABI details, see
[`GPU_ARCHITECTURE.md`](../../docs/GPU_ARCHITECTURE.md),
[`GPU_BENCHMARKS.md`](../../docs/GPU_BENCHMARKS.md), and
[`WGSL_MEMORY.md`](../../docs/WGSL_MEMORY.md).
