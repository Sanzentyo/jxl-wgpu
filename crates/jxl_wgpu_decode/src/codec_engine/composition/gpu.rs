use std::borrow::Cow;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::FrameInventory;
use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::{Extent2d, OutputOrientation, RgbColorEncoding};
use jxl_wgpu::{
    GpuBufferLease, GpuImageOutput, IMAGE_ORIENTATION_SHADER, IMAGE_OUTPUT_SHADER,
    ImageOutputParams, ImageOutputSource, MemoryPermit, WgpuBackend,
};
use wgpu::util::DeviceExt;

use crate::{Error, GpuOutputRequest, Result};

/// The composition boundary is tightly packed, unrounded, unrotated, original-encoding RGBA.
/// The fourth component is virtual opaque alpha when the codestream has no alpha channel.
#[derive(Clone, Debug)]
pub(super) struct Surface {
    pub(super) buffer: GpuBufferLease,
    pub(super) extent: Extent2d,
}

impl Surface {
    pub(super) fn from_output(
        output: GpuImageOutput,
        working_format: &jxl_gpu_formats::PixelFormat,
    ) -> Result<Self> {
        let expected = ImageLayout::packed(output.layout.extent, working_format.clone())?;
        if output.layout != expected || output.buffer.size() < expected.logical_size {
            return Err(Error::EngineContract(
                "frame producer returned an invalid F32 composition surface",
            ));
        }
        Ok(Self {
            buffer: output.buffer,
            extent: output.layout.extent,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlendParams {
    canvas: [u32; 4],
    intersection: [u32; 4],
    source: [u32; 4],
    blend: [u32; 4],
    flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NativeParams {
    extent: [u32; 4],
    format: [u32; 4],
    output: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<BlendParams>() == 80);
const _: () = assert!(std::mem::size_of::<NativeParams>() == 48);

#[derive(Debug)]
enum Packing {
    Color(Box<ImageOutputParams>),
    Native(NativeParams),
}

#[derive(Debug)]
pub(super) struct Compositor {
    backend: WgpuBackend,
    canvas: Extent2d,
    has_alpha: bool,
    blend: wgpu::ComputePipeline,
    pack: wgpu::ComputePipeline,
    packing: Packing,
    pub(super) layout: ImageLayout,
    blend_dispatch: [u32; 2],
    output_dispatch: [u32; 2],
}

impl Compositor {
    pub(super) fn new(
        backend: WgpuBackend,
        canvas: Extent2d,
        has_alpha: bool,
        grayscale: bool,
        orientation: OutputOrientation,
        request: &GpuOutputRequest,
    ) -> Result<Self> {
        let orientation = request.orientation_policy().resolve(orientation);
        let layout = ImageLayout::packed(orientation.map_extent(canvas), request.format().clone())?;
        let device = backend.device();
        validate_size(device, surface_bytes(canvas)?)?;
        let output_size = aligned(layout.logical_size)?;
        validate_size(device, output_size)?;
        let blend_dispatch = dispatch(device, u64::from(canvas.width) * u64::from(canvas.height))?;
        let output_dispatch = dispatch(device, output_size / 4)?;
        let (packing, source) = if let Some(native) =
            crate::model::native_modular_format(request.format())
        {
            if native.channels == crate::ModularChannels::Gray && !grayscale {
                return Err(Error::UnsupportedOutputFormat(
                    "numeric grayscale composition requires a grayscale codestream".into(),
                ));
            }
            (
                Packing::Native(NativeParams {
                    extent: [
                        layout.extent.width,
                        layout.extent.height,
                        canvas.width,
                        canvas.height,
                    ],
                    format: [
                        native.channels.count(),
                        u32::from(native.bits_per_sample),
                        u32::from(native.storage_bits) / 8,
                        u32::try_from(layout.planes[0].row_stride).map_err(|_| address_error())?,
                    ],
                    output: [
                        u32::try_from(layout.logical_size).map_err(|_| address_error())?,
                        output_dispatch[0] * 64,
                        orientation.to_exif_value() - 1,
                        0,
                    ],
                }),
                format!(
                    "{IMAGE_ORIENTATION_SHADER}\n{}",
                    include_str!("native.wgsl")
                ),
            )
        } else {
            (
                Packing::Color(Box::new(ImageOutputParams::new(
                    &layout,
                    ImageOutputSource {
                        extent: canvas,
                        orientation,
                        strides: [canvas.width * 4; 3],
                        encoding: RgbColorEncoding::SRGB_BT709,
                    },
                    output_dispatch[0] * 64,
                )?)),
                format!("{IMAGE_OUTPUT_SHADER}\n{}", include_str!("output.wgsl")),
            )
        };
        let blend = pipeline(
            device,
            "JPEG XL frame composition",
            include_str!("blend.wgsl"),
            false,
        );
        let pack = pipeline(
            device,
            "JPEG XL composed frame output",
            &source,
            matches!(packing, Packing::Color(_)),
        );
        Ok(Self {
            backend,
            canvas,
            has_alpha,
            blend,
            pack,
            packing,
            layout,
            blend_dispatch,
            output_dispatch,
        })
    }

    pub(super) fn blend(
        &self,
        foreground: &Surface,
        color: Option<&Surface>,
        alpha: Option<&Surface>,
        frame: &FrameInventory,
    ) -> Result<GpuWork> {
        if foreground.extent != Extent2d::new(frame.width, frame.height)
            || [color, alpha].into_iter().flatten().any(|base| {
                base.extent.width < self.canvas.width || base.extent.height < self.canvas.height
            })
        {
            return Err(Error::EngineContract(
                "composition surface geometry disagrees with the frame plan",
            ));
        }
        let (intersection, origin) = intersection(self.canvas, frame);
        let ec = frame
            .extra_channel_blends
            .first()
            .copied()
            .unwrap_or_default();
        let params = BlendParams {
            canvas: [
                self.canvas.width,
                self.canvas.height,
                foreground.extent.width,
                self.blend_dispatch[0] * 64,
            ],
            intersection,
            source: [
                origin[0],
                origin[1],
                color.map_or(0, |s| s.extent.width),
                alpha.map_or(0, |s| s.extent.width),
            ],
            blend: [
                frame.color_blend.mode as u32,
                ec.mode as u32,
                u32::from(frame.color_blend.clamp),
                u32::from(ec.clamp),
            ],
            flags: [
                u32::from(self.has_alpha),
                u32::from(color.is_some()),
                u32::from(alpha.is_some()),
                0,
            ],
        };
        let size = surface_bytes(self.canvas)?;
        self.submit(
            &self.blend,
            bytemuck::bytes_of(&params),
            &[
                (0, &foreground.buffer),
                (1, &color.unwrap_or(foreground).buffer),
                (2, &alpha.unwrap_or(foreground).buffer),
            ],
            3,
            4,
            size,
            self.blend_dispatch,
        )
    }

    pub(super) fn pack(&self, source: &Surface) -> Result<GpuWork> {
        if source.extent != self.canvas {
            return Err(Error::EngineContract(
                "presentation canvas has the wrong extent",
            ));
        }
        let (params, output_binding, uniform_binding): (&[u8], _, _) = match &self.packing {
            Packing::Color(params) => (bytemuck::bytes_of(params.as_ref()), 3, 4),
            Packing::Native(params) => (bytemuck::bytes_of(params), 1, 2),
        };
        self.submit(
            &self.pack,
            params,
            &[(0, &source.buffer)],
            output_binding,
            uniform_binding,
            aligned(self.layout.logical_size)?,
            self.output_dispatch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn submit(
        &self,
        pipeline: &wgpu::ComputePipeline,
        params: &[u8],
        inputs: &[(u32, &GpuBufferLease)],
        output_binding: u32,
        uniform_binding: u32,
        size: u64,
        dispatch: [u32; 2],
    ) -> Result<GpuWork> {
        let device = self.backend.device();
        let memory = self.backend.transient_memory_budget();
        let poll = self.backend.submission_poller().try_reserve()?;
        let output_permit = memory.try_reserve(size)?;
        let uniform_permit = memory.try_reserve(params.len() as u64 + completion_fence_bytes())?;
        let guards = inputs
            .iter()
            .map(|(_, lease)| lease.try_acquire_gpu_submission())
            .collect::<jxl_wgpu::Result<Vec<_>>>()?;
        let output = GpuBufferLease::from_tracked(
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("JPEG XL composed frame"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            output_permit,
        );
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JPEG XL composition parameters"),
            contents: params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut entries = inputs
            .iter()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: buffer.as_wgpu_buffer().as_entire_binding(),
            })
            .collect::<Vec<_>>();
        entries.push(wgpu::BindGroupEntry {
            binding: output_binding,
            resource: output.as_wgpu_buffer().as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: uniform_binding,
            resource: uniform.as_entire_binding(),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JPEG XL composition bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("JPEG XL frame composition"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("JPEG XL frame composition"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch[0], dispatch[1], 1);
        }
        let completion = Arc::new(Completion::default());
        // wgpu work-done callbacks require Send even on WebGPU. A tiny mapped completion fence
        // uses the browser-local map callback instead, retaining the non-Send WebGPU handles.
        // Its contents are never read; no image samples cross the CPU boundary.
        #[cfg(target_arch = "wasm32")]
        let completion_fence = {
            let fence = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("JPEG XL composition completion fence"),
                size: completion_fence_bytes(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.clear_buffer(&fence, 0, None);
            fence
        };
        let lifetime = Arc::new(WorkLifetime {
            _buffers: inputs
                .iter()
                .map(|(_, b)| (*b).clone())
                .chain([output.clone()])
                .collect(),
            _uniform: uniform,
            _permit: uniform_permit,
            #[cfg(target_arch = "wasm32")]
            completion_fence,
        });
        let done = Arc::clone(&completion);
        let retained = Arc::clone(&lifetime);
        #[cfg(not(target_arch = "wasm32"))]
        encoder.on_submitted_work_done(move || {
            drop(retained);
            done.complete(Ok(()));
        });
        let submission = self.backend.queue().submit([encoder.finish()]);
        drop(guards);
        #[cfg(target_arch = "wasm32")]
        lifetime
            .completion_fence
            .map_async(wgpu::MapMode::Read, .., move |result| {
                if result.is_ok() {
                    retained.completion_fence.unmap();
                }
                drop(retained);
                done.complete(result.map_err(|error| error.to_string()));
            });
        let failed = Arc::clone(&completion);
        poll.register(submission, move |error| {
            #[cfg(not(target_arch = "wasm32"))]
            drop(lifetime);
            failed.complete(Err(error));
        })?;
        Ok(GpuWork {
            output: Some(output),
            completion,
        })
    }
}

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    color: bool,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: if color {
                &[("wg_x", 64.0), ("wg_y", 1.0)]
            } else {
                &[]
            },
            ..Default::default()
        },
        cache: None,
    })
}

fn address_error() -> Error {
    Error::CompositionResourceLimit {
        resource: "byte addressing",
        requested: u64::MAX,
        limit: u64::from(u32::MAX - 3),
    }
}

fn surface_bytes(extent: Extent2d) -> Result<u64> {
    u64::from(extent.width)
        .checked_mul(u64::from(extent.height))
        .and_then(|pixels| pixels.checked_mul(16))
        .ok_or_else(address_error)
}

fn aligned(size: u64) -> Result<u64> {
    size.checked_add(3)
        .map(|n| n & !3)
        .ok_or_else(address_error)
}

fn validate_size(device: &wgpu::Device, size: u64) -> Result<()> {
    let limits = device.limits();
    let limit = u64::from(u32::MAX - 3)
        .min(limits.max_buffer_size)
        .min(limits.max_storage_buffer_binding_size);
    if size == 0 {
        return Err(Error::EngineContract(
            "composition requires a nonempty surface",
        ));
    }
    if size > limit {
        return Err(Error::CompositionResourceLimit {
            resource: "buffer bytes",
            requested: size,
            limit,
        });
    }
    Ok(())
}

fn dispatch(device: &wgpu::Device, elements: u64) -> Result<[u32; 2]> {
    let limit = u64::from(device.limits().max_compute_workgroups_per_dimension);
    if elements == 0 || limit == 0 {
        return Err(Error::EngineContract(
            "composition requires a nonempty dispatch",
        ));
    }
    let x = elements.div_ceil(64).min(limit);
    let y = elements.div_ceil(x * 64);
    if y > limit {
        return Err(Error::CompositionResourceLimit {
            resource: "dispatch rows",
            requested: y,
            limit,
        });
    }
    Ok([x as u32, y as u32])
}

fn intersection(canvas: Extent2d, frame: &FrameInventory) -> ([u32; 4], [u32; 2]) {
    fn axis(limit: u32, origin: i32, size: u32) -> (u32, u32, u32) {
        let start = i64::from(origin).clamp(0, i64::from(limit));
        let end = (i64::from(origin) + i64::from(size)).clamp(start, i64::from(limit));
        let width = (end - start) as u32;
        let source = if width == 0 {
            0
        } else {
            (start - i64::from(origin)) as u32
        };
        (start as u32, width, source)
    }
    let (x, width, sx) = axis(canvas.width, frame.x0, frame.width);
    let (y, height, sy) = axis(canvas.height, frame.y0, frame.height);
    ([x, y, width, height], [sx, sy])
}

struct WorkLifetime {
    _buffers: Vec<GpuBufferLease>,
    _uniform: wgpu::Buffer,
    _permit: MemoryPermit,
    #[cfg(target_arch = "wasm32")]
    completion_fence: wgpu::Buffer,
}

const fn completion_fence_bytes() -> u64 {
    if cfg!(target_arch = "wasm32") { 4 } else { 0 }
}

#[derive(Debug, Default)]
struct Completion {
    state: Mutex<CompletionState>,
    condition: Condvar,
}
#[derive(Debug, Default)]
struct CompletionState {
    result: Option<std::result::Result<(), String>>,
    waker: Option<Waker>,
}

impl Completion {
    fn complete(&self, result: std::result::Result<(), String>) {
        let waker = {
            let mut state = super::lock(&self.state);
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        self.condition.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }
    fn poll(&self, context: &Context<'_>) -> Poll<Result<()>> {
        let mut state = super::lock(&self.state);
        if let Some(result) = state.result.as_ref() {
            return Poll::Ready(result.clone().map_err(Error::backend));
        }
        state.waker = Some(context.waker().clone());
        Poll::Pending
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) -> Result<()> {
        let mut state = super::lock(&self.state);
        while state.result.is_none() {
            state = self
                .condition
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state
            .result
            .as_ref()
            .expect("completion signalled")
            .clone()
            .map_err(Error::backend)
    }
}

#[derive(Debug)]
pub(super) struct GpuWork {
    output: Option<GpuBufferLease>,
    completion: Arc<Completion>,
}
impl GpuWork {
    pub(super) fn poll(&mut self, context: &Context<'_>) -> Poll<Result<GpuBufferLease>> {
        self.completion.poll(context).map(|result| {
            result?;
            self.output.take().ok_or(Error::EngineContract(
                "composition completion consumed twice",
            ))
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn wait(mut self) -> Result<GpuBufferLease> {
        self.completion.wait()?;
        self.output.take().ok_or(Error::EngineContract(
            "composition completion consumed twice",
        ))
    }
    pub(super) fn unvalidated(&self) -> Result<GpuBufferLease> {
        self.output.clone().ok_or(Error::EngineContract(
            "composition completion consumed twice",
        ))
    }
}
