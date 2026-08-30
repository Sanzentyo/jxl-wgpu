# jxl_wgpu

`jxl_wgpu` is the optional portable GPU backend for this `jxl-rs` fork. It implements
`jxl::accelerator::JxlAccelerator` on top of `wgpu` 30 while keeping `jxl` itself free of graphics
dependencies.

The forked decoder needs a narrow backend-neutral hook because the stock public API exposes only
the already-rendered CPU image; attaching at that point cannot accelerate IDCT, restoration, or
upsampling. `wgpu` resources and native YUV/NV12 conversion remain entirely in this crate. See the
architecture document for the exact minimal boundary and fallback contract.

```rust,no_run
use std::sync::Arc;

use jxl::api::{JxlDecoder, JxlDecoderOptions, states};
use jxl_wgpu::{WgpuAccelerator, WgpuAcceleratorConfig};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let gpu = Arc::new(
    WgpuAccelerator::request_default(WgpuAcceleratorConfig {
        minimum_frame_pixels: Some(0),
        ..WgpuAcceleratorConfig::default()
    })
    .await?,
);
let decoder = JxlDecoder::<states::Initialized>::builder(JxlDecoderOptions::default())
    .accelerator(gpu)
    .build();
# let _ = decoder;
# Ok(())
# }
```

Applications with an existing renderer can use `WgpuAccelerator::from_device` to share a
`wgpu::Device` and `wgpu::Queue`.

Automatic decoder integration is disabled by default (`minimum_frame_pixels: None`) because the
CPU-readback path has not yet demonstrated a crossover on the checked-in Apple M5 baseline. Set it
to `Some(0)` to force GPU planning (useful for validation), or to a measured pixel threshold for the
target adapter and workload. Attaching a default backend therefore preserves the decoder's original
low-memory CPU pipeline rather than silently regressing it.

## Zero-copy output

The concrete `WgpuFrameSession` also exposes a GPU-only path for renderers that do not need the
backend-neutral CPU `RenderedOutput`:

```rust,ignore
use jxl::accelerator::RenderIntent;

let gpu_frame = session.submit_gpu(RenderIntent::Final)?;
let rgba = &gpu_frame.outputs[0];

// `rgba.buffer` is an Arc<wgpu::Buffer> with STORAGE | COPY_SRC usage. It can be bound or copied
// by a command submitted to `gpu.queue()` immediately; queue ordering waits on the save kernel on
// the GPU without blocking the CPU.
consumer_encoder.copy_buffer_to_buffer(
    &rgba.buffer,
    0,
    &renderer_buffer,
    0,
    rgba.logical_size,
);
gpu.queue().submit([consumer_encoder.finish()]);

// Native callers may optionally confirm host-visible completion. This is not needed for a
// dependent command on the same queue.
session.wait_gpu(gpu_frame.token)?;
```

Each `GpuOutputBuffer` carries its `OutputId`, extent, sample type, channel count, layout, and
meaningful byte length. Its `Arc<wgpu::Buffer>` remains valid after the session is dropped. GPU-only
submissions allocate one packed buffer per output and do not create a readback buffer, record a
GPU-to-readback copy, or map memory. The existing `AcceleratedFrameSession::submit`/`wait` behavior
is unchanged.

On native unified-memory adapters that expose `MAPPABLE_PRIMARY_BUFFERS`, CPU submissions write
into a `STORAGE | MAP_READ` allocation and map that allocation directly. This removes the
full-size GPU-to-staging copy and second output allocation. The default
`DirectReadbackPolicy::Auto` enables this only for integrated/CPU adapters; discrete adapters use
staging even if they expose the feature. `DirectReadbackPolicy::Force` is an explicit opt-in for a
measured discrete target and returns a typed error if the feature is unavailable.
`enable_direct_readback: false` remains the compatibility master switch for forcing the portable
path, and `WgpuSubmissionStats::direct_readback` reports which path a submission used.

## Bounded internal buffer reuse

One `WgpuAccelerator` owns a thread-safe, exact-size-and-usage buffer pool shared by all of its
frame sessions. Repeated and concurrent CPU decodes reuse resident arena slots and the packed CPU
output allocation. Portable readback staging remains independently owned by the pending map, and
GPU-only `Arc<wgpu::Buffer>` outputs are never pooled, so they stay valid until the caller drops
them.

Reuse follows queue ownership boundaries: resident slots and portable packed outputs return only
after their command buffer is submitted to the same queue; directly mapped outputs return only
after successful readback and `unmap`. VarDCT slots that may be only partially written use a
known-zero pool class and are cleared at the end of the preceding submission. Full-frame kernels
and zero-filled source uploads reuse dirty allocations without paying that clear bandwidth.

`WgpuMemoryPolicy::max_cached_buffer_bytes` bounds idle allocations (128 MiB by default and zero to
disable). This cache limit is separate from per-submission resident/transient budgets. Use
`WgpuAccelerator::buffer_pool_stats()` to inspect hits, misses, evictions, rejected aliases, and
idle bytes, or `clear_buffer_pool()` to release all idle buffers without affecting in-flight work
or public GPU outputs.

The ignored release microbenchmark
`session::tests::repeated_cpu_decode_pool_release_benchmark` measures 20 repeated 512x512 CPU
readbacks after pipeline warmup. One Apple M5/Metal 4 reference run improved from 21.25 ms with the
pool disabled to 18.55 ms enabled (0.873x elapsed); this is an allocation-churn indicator, not a
portable crossover claim.

## Generic pitch-linear image output

The concrete session can lower its final three normalized, nonlinear R'G'B' F32 planes directly
to a canonical `jxl_gpu_formats::PixelFormat`. Supported GPU output includes Y8/Y16,
planar/semi-planar 4:4:4, 4:2:2, and 4:2:0 at 8/10/12/16 bits (including NV12/NV21,
NV24/NV42, P010/P012/P016), YUYV/UYVY, and planar/interleaved RGB/BGR/RGBA/BGRA8.
No intermediate CPU codec or RGB(A) readback is involved:

```rust,ignore
use jxl::accelerator::RenderIntent;
use jxl_wgpu::{
    ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, ImageOutputRequest, PixelFormat,
};

let color = ColorSpecification::Defined(ColorSpec::bt709(
    ColorRange::Limited,
    ChromaLocation2d::CENTER,
));
let request = ImageOutputRequest::new(PixelFormat::nv12(color));

// CPU-visible bytes and explicit per-plane offsets, strides, and extents:
let token = session.submit_image(RenderIntent::Final, request.clone())?;
let nv12 = session.wait_image(token)?.outputs.remove(0);

// Or a same-queue, zero-copy GPU buffer with the same layout descriptor:
let gpu_nv12 = session.submit_gpu_image(RenderIntent::Final, request)?;
```

Centered chroma means a box average of the valid subsampling footprint, including cropped odd
edges; even/cosited chroma uses the top-left luma location. The matrix is applied directly to the
supplied R'G'B' values and does not add a transfer-function conversion. Unsupported or ambiguous
numeric/color descriptors return `Error::Unsupported` rather than guessing display semantics.

This is currently a `WgpuFrameSession` API. CPU image readback is an output transport and must not
be confused with a CPU codec fallback: decoding and image conversion remain GPU operations.

## Display textures without a host wait

`DisplayPipeline` converts a `GpuOutputBuffer` or generic `GpuImageOutput` into an
`Rgba8Unorm` texture carrying `TEXTURE_BINDING`, `RENDER_ATTACHMENT`, and copy usages. The
conversion and every later render/copy can be submitted to the accelerator's same queue without
waiting on the host:

```rust,ignore
use jxl_wgpu::{DisplayPipeline, DisplayTextureDescriptor};

let display = DisplayPipeline::new(&gpu);
let image_frame = session.submit_gpu_image(RenderIntent::Final, request)?;
let display_submission = display.submit_image(
    &image_frame.outputs[0],
    DisplayTextureDescriptor::default(),
)?;
renderer.sample(display_submission.texture.view());
```

The format/matrix/range/siting combination keys a thread-safe compiled-pipeline cache. Arbitrary
tight pitch-linear rows are accepted by the compute path. The explicit RGBA8 buffer-to-texture
copy path validates WebGPU's 256-byte multi-row pitch requirement and returns
`Error::TextureCopyRowAlignment` when it is not met. Native multi-plane handles and vendor-specific
block-linear memory are outside this portable pitch-linear contract.

`WgpuFrameSession::last_submission_stats()` exposes the most recent `WgpuSubmissionStats` for
instrumentation. `planned_dispatches` is the number of dispatch groups in the `ExecutionPlan`,
`compute_dispatches` counts actual compute-pass `dispatch_workgroups` calls (including multi-pass
Save, YCbCr, and VarDCT work), and `fused_dispatches` counts dispatches emitted by a fused encoder
instead of a single-node kernel.

On WebAssembly, `submit_gpu` remains non-blocking and same-queue consumers work normally.
`wait_gpu` returns `Error::Unsupported`, because browser WebGPU cannot implement a synchronous host
wait; applications should rely on queue ordering or an asynchronous queue-completion callback.
Browser WebGPU handles are not `Send + Sync`, so wasm callers use the concrete
`WgpuAccelerator::create_session` / `WgpuFrameSession` API. The `JxlAccelerator` and
`AcceleratedFrameSession` trait implementations remain native-only because those backend-neutral
traits intentionally require `Send + Sync` / `Send`.

The backend implements typed planning, budget-checked resident execution, lifetime-based slot
reuse, revisioned frame sessions, real Chroma2d/GaborishRgb dispatch fusion, pipeline caching,
batched submission, checked upload and readback, accuracy metrics, and adapter-specific autotuning
profiles. Explicit streaming requests
are rejected until the tiled scheduler is connected, so the backend never silently exceeds the
declared memory model. The decoder automatically falls back to its CPU stages if a plan segment is
unsupported or GPU execution fails.

See [`docs/GPU_ARCHITECTURE.md`](../docs/GPU_ARCHITECTURE.md),
[`docs/GPU_BENCHMARKS.md`](../docs/GPU_BENCHMARKS.md), and the `jxl_gpu_harness` crate for the
protocol, current performance boundary, and validation workflow.
