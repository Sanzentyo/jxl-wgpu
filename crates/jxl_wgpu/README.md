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
`wgpu::Device`, `wgpu::Queue`, `wgpu::AdapterInfo`, and `WgpuBackendConfig`. Decode, conversion,
display, and later renderer work can then share one queue and use queue ordering instead of a host
wait.

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

// A dependent command on backend.queue() can use output.buffer immediately.
consumer.copy_buffer_to_buffer(
    &output.buffer,
    0,
    &destination,
    0,
    output.logical_size,
);
backend.queue().submit([consumer.finish()]);
```

Each `GpuOutputBuffer` carries its output ID, extent, sample type, channel count, layout, meaningful
byte length, and an `Arc<wgpu::Buffer>`. The handle remains valid after the frame session is
dropped. `wait_gpu` is only needed when native host code explicitly requires completion; dependent
commands on the same queue do not need it.

## Generic pitch-linear formats

`WgpuFrameSession::submit_gpu_image` converts the final nonlinear R'G'B' planes directly into a
checked `jxl_gpu_formats::PixelFormat`. `submit_image` plus `wait_image` provides the corresponding
mapped transport. Supported layout families include:

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
let request = ImageOutputRequest::new(PixelFormat::nv12(color));
let nv12 = session.submit_gpu_image(RenderIntent::Final, request)?;
```

Every output has an `ImageLayout` with explicit plane offsets, row strides, extents, sampling, and
logical byte size. Odd chroma extents use checked ceil division. Centered siting averages the valid
subsampling footprint; cosited output uses the top-left luma position. Numeric representation,
matrix, range, and siting are never inferred from an ambiguous descriptor.

These contracts describe portable pitch-linear buffers. Vendor block-linear memory and native
multi-plane texture handles are outside WebGPU's portable buffer model.

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

Callers that already own a command encoder can use `encode_rgb`, `encode_image`, or
`encode_rgba8_copy`, then submit all work as one application-defined batch.

## Aggregate CPU readback

`ImageReadbackPipeline` copies every output in one `GpuImageFrame` into one bounded staging buffer.
Each source region and staging offset is independently padded to WebGPU's four-byte buffer-copy
alignment; returned vectors contain only each layout's logical bytes.

```rust,ignore
use jxl_wgpu::ImageReadbackPipeline;

let pending = ImageReadbackPipeline::new(&backend).submit(&nv12)?;
let planned = pending.stats();

// Runtime-neutral asynchronous completion:
let result = pending.await?;
# let _ = (planned, result);
```

`ImageReadbackSubmission` implements `std::future::Future` directly and has no executor-specific
dependency. Native callers may use `pending.wait()` instead. Browser WebGPU uses the future path.
`ImageReadbackStats` reports output count, logical bytes, and the exact padded staging allocation;
`ImageReadbackLimits` sets the per-submission staging limit.

For protocol `submit`/`wait`, `DirectReadbackPolicy` selects automatic direct mapping on eligible
unified-memory adapters, an explicit storage-to-staging copy, or required direct mapping. Requiring
direct mapping returns a typed error when `MAPPABLE_PRIMARY_BUFFERS` is unavailable.

## Memory policy and instrumentation

`WgpuBackendConfig::memory` contains four independent checked limits:

- `max_resident_bytes` for physical plane slots;
- `max_scratch_bytes` for simultaneously live intermediate planes;
- `max_transient_bytes` for one submission's uniforms, uploads, packed outputs, and staging; and
- `max_cached_buffer_bytes` for idle reusable buffers.

The planner computes plane lifetimes, reuses physical slots only across disjoint live ranges, and
validates both device buffer and storage-binding limits. The shared buffer pool leases by exact size
and usage, never returns one allocation to two live submissions, and exposes hits, misses,
evictions, rejected leases, and idle bytes through `buffer_pool_stats()`.

`WgpuSubmissionStats` reports planned, actual, and fused dispatch counts plus resident and transient
bytes. `WgpuFrameSession::pending_transient_bytes()` reports the checked sum associated with tokens
that have not been waited. Public GPU outputs remain caller-owned and are not recycled through the
internal pool.

All copy sizes are four-byte aligned, all offsets/strides/allocation sums use checked arithmetic,
and values consumed as WGSL indices must fit `u32`. Rust uniform/storage records use `#[repr(C)]`
plus `bytemuck::Pod`; tests pin their field order, size, alignment, and structured-array stride.
VarDCT DCT8 additionally checks its 1,536-byte workgroup-memory requirement. The complete table is
in [`docs/WGSL_MEMORY.md`](../../docs/WGSL_MEMORY.md).

## Implemented render operations

The current planner and scheduler execute these protocol stages:

- copy and integer-Modular-to-F32 conversion;
- horizontal/vertical chroma reconstruction and fused 2-D chroma reconstruction;
- per-plane and fused three-plane Gaborish;
- EPF passes 0, 1, and 2 with constant or supplied sigma;
- 2x, 4x, and 8x image upsampling with validated weights;
- DCT8 VarDCT dequantization, color correlation, inverse transform, and LF composition;
- YCbCr-to-RGB components, alpha premultiplication, sample conversion, orientation, Save packing,
  and generic image packing; and
- direct generic-image and RGBA display conversion.

The backend validates precision contracts, node arity, plane types/extents/strides, resource
revisions, transform coverage, dispatch dimensions, shader-visible addresses, and device limits
before submission. Unsupported render operations, unsupported VarDCT transform buckets, explicit
streaming execution, invalid color semantics, and unavailable device features return
`Error::Unsupported` or another specific resource/payload error.

## Platform notes

Native builds expose both trait-object and concrete frame-session paths. Browser WebGPU resources
are single-threaded, so callers use `WgpuBackend::create_session` and the concrete session API.
GPU-resident output and same-queue display remain non-blocking there; host mapping uses
`ImageReadbackSubmission` as a future.

For architecture, performance methodology, and ABI details, see
[`GPU_ARCHITECTURE.md`](../../docs/GPU_ARCHITECTURE.md),
[`GPU_BENCHMARKS.md`](../../docs/GPU_BENCHMARKS.md), and
[`WGSL_MEMORY.md`](../../docs/WGSL_MEMORY.md).
