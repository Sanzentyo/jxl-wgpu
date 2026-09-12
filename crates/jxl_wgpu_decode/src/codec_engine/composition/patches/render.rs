use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{GpuBufferLease, WgpuBackend};
use wgpu::util::DeviceExt;

use super::super::gpu::{Surface, dispatch, pipeline};
use super::super::submission::{GpuWork, completion_fence_bytes, submit_recorded, validate_size};
use super::Dictionary;
use crate::frame_surface::{FrameSurfaceEncoding, FrameSurfaceLayout};
use crate::progressive_dc::{
    ProgressiveDcExtras, ProgressiveDcGpuError, ProgressiveDcOutput, ProgressiveDcXybPlanes,
};
use crate::{Error, Result};

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    canvas: [u32; 4],
    jobs: [u32; 4],
    flags: [u32; 4],
    references: [[u32; 4]; 4],
}
const _: () = assert!(std::mem::size_of::<Params>() == 112);

/// The same ordered patch program operates on ordinary frame surfaces and LF prediction planes.
struct Render<'a> {
    backend: &'a WgpuBackend,
    extent: Extent2d,
    plane_words: u32,
    channels: u32,
    references: &'a [Option<Surface>; 4],
    dictionary: &'a Dictionary,
    has_alpha: bool,
    shape: [u32; 2],
    batches: u32,
    size: u64,
}

impl<'a> Render<'a> {
    fn new(
        backend: &'a WgpuBackend,
        extent: Extent2d,
        plane_words: u32,
        references: &'a [Option<Surface>; 4],
        dictionary: &'a Dictionary,
        extra_count: u32,
        has_alpha: bool,
    ) -> Result<Self> {
        let channels = extra_count
            .checked_add(3)
            .ok_or_else(|| Error::backend("patch channels overflow"))?;
        let size = u64::from(plane_words) * u64::from(channels) * 4;
        validate_size(backend.device(), size)?;
        Ok(Self {
            backend,
            extent,
            plane_words,
            channels,
            references,
            dictionary,
            has_alpha,
            shape: dispatch(
                backend.device(),
                u64::from(extent.width) * u64::from(extent.height),
            )?,
            batches: dictionary.count.div_ceil(64),
            size,
        })
    }

    fn scratch_bytes(&self) -> u64 {
        self.size + u64::from(self.batches) * 112 + completion_fence_bytes()
    }

    fn inputs(&self, source: impl IntoIterator<Item = GpuBufferLease>) -> Vec<GpuBufferLease> {
        source
            .into_iter()
            .chain([self.dictionary.commands.clone()])
            .chain(
                self.references
                    .iter()
                    .flatten()
                    .map(|surface| surface.buffer.clone()),
            )
            .collect()
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::Buffer,
        scratch: &wgpu::Buffer,
        fallback: &wgpu::Buffer,
    ) -> Vec<wgpu::Buffer> {
        let device = self.backend.device();
        let pipeline = pipeline(
            device,
            "JPEG XL patch rendering",
            include_str!("render.wgsl"),
            &[],
        );
        (0..self.batches)
            .map(|batch| {
                let params = Params {
                    canvas: [
                        self.extent.width,
                        self.extent.height,
                        self.plane_words,
                        self.channels,
                    ],
                    jobs: [
                        batch * 64,
                        self.dictionary.count.min((batch + 1) * 64),
                        self.dictionary.stride,
                        self.shape[0] * 64,
                    ],
                    flags: [u32::from(self.has_alpha), 0, 0, 0],
                    references: self.references.each_ref().map(|slot| {
                        slot.as_ref().map_or([0; 4], |surface| {
                            [
                                surface.extent.width,
                                surface.extent.height,
                                surface.plane_words,
                                1,
                            ]
                        })
                    }),
                };
                let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("JPEG XL patch batch"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let mut entries: Vec<_> = self
                    .references
                    .iter()
                    .enumerate()
                    .map(|(index, slot)| wgpu::BindGroupEntry {
                        binding: index as u32,
                        resource: slot
                            .as_ref()
                            .map_or(fallback, |surface| surface.buffer.as_wgpu_buffer())
                            .as_entire_binding(),
                    })
                    .collect();
                entries.extend(
                    [
                        (4, self.dictionary.commands.as_wgpu_buffer()),
                        (5, output),
                        (6, scratch),
                        (7, &uniform),
                    ]
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding,
                        resource: buffer.as_entire_binding(),
                    }),
                );
                let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("JPEG XL patch render bindings"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                });
                {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bindings, &[]);
                    pass.dispatch_workgroups(self.shape[0], self.shape[1], 1);
                }
                uniform
            })
            .collect()
    }
}

fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

pub(in super::super) fn render(
    backend: &WgpuBackend,
    source: &Surface,
    references: &[Option<Surface>; 4],
    dictionary: &Dictionary,
    extra_count: u32,
    has_alpha: bool,
) -> Result<GpuWork> {
    if source.encoding != FrameSurfaceEncoding::Encoded {
        return Err(Error::EngineContract(
            "patch input must precede color conversion",
        ));
    }
    let render = Render::new(
        backend,
        source.extent,
        source.plane_words,
        references,
        dictionary,
        extra_count,
        has_alpha,
    )?;
    let device = backend.device();
    let poll = backend.submission_poller().try_reserve()?;
    let output_permit = backend.transient_memory_budget().try_reserve(render.size)?;
    let permit = backend
        .transient_memory_budget()
        .try_reserve(render.scratch_bytes())?;
    let output = GpuBufferLease::from_tracked(
        storage(device, "JPEG XL patched frame", render.size),
        output_permit,
    );
    let scratch = storage(device, "JPEG XL patch channel snapshot", render.size);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(
        source.buffer.as_wgpu_buffer(),
        0,
        output.as_wgpu_buffer(),
        0,
        render.size,
    );
    let uniforms = render.record(
        &mut encoder,
        output.as_wgpu_buffer(),
        &scratch,
        source.buffer.as_wgpu_buffer(),
    );
    submit_recorded(
        backend,
        encoder,
        output,
        render.inputs([source.buffer.clone()]),
        (scratch, uniforms),
        permit,
        poll,
    )
}

/// Patch all LF channels, then retain prediction and presentation in separate allocations.
/// Copies and ordered patch batches share one submission; no samples are mapped to the host.
pub(in super::super) fn render_lf(
    backend: &WgpuBackend,
    source: &ProgressiveDcOutput,
    references: &[Option<Surface>; 4],
    dictionary: &Dictionary,
    extras: &[jxl_gpu_bitstream::ExtraChannelInventory],
    retain_extras: bool,
) -> Result<GpuWork<ProgressiveDcOutput>> {
    let device = backend.device();
    let extent = Extent2d::new(source.xyb.width(), source.xyb.height());
    let layout = FrameSurfaceLayout::with_encoding(
        extent,
        extras.len(),
        FrameSurfaceEncoding::Encoded,
        &device.limits(),
    )?;
    let render = Render::new(
        backend,
        extent,
        (layout.plane_bytes / 4) as u32,
        references,
        dictionary,
        extras.len() as u32,
        extras.iter().any(|extra| {
            matches!(
                extra.channel_type,
                jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { .. }
            )
        }),
    )?;
    let extra_planes = source
        .extras
        .as_ref()
        .map_or(&[][..], |extras| extras.planes.as_slice());
    if extra_planes.len() != extras.len() {
        return Err(Error::EngineContract(
            "LF patches require every decoded extra plane",
        ));
    }
    let inputs: Vec<_> = source
        .xyb
        .planes
        .iter()
        .map(|plane| Plane {
            buffer: &plane.buffer,
            offset: 0,
            stride: plane.stride,
            width: plane.width,
            height: plane.height,
        })
        .chain(extra_planes.iter().map(|plane| Plane {
            buffer: &source.extras.as_ref().expect("extra planes exist").buffer,
            offset: plane.word_offset,
            stride: plane.row_stride_words,
            width: plane.width,
            height: plane.height,
        }))
        .collect();
    for plane in &inputs {
        plane.validate(extent)?;
    }
    let plane_bytes = u64::from(extent.width) * u64::from(extent.height) * 4;
    let extra_bytes = if retain_extras {
        layout.plane_bytes * extras.len() as u64
    } else {
        0
    };
    let poll = backend.submission_poller().try_reserve()?;
    let mut permit = backend
        .transient_memory_budget()
        .try_reserve(render.scratch_bytes() + render.size + plane_bytes * 3 + extra_bytes)?;
    let mut retained = |label, size| -> Result<GpuBufferLease> {
        let part = permit
            .split_off(size)
            .map_err(ProgressiveDcGpuError::from)?;
        Ok(GpuBufferLease::from_tracked(
            storage(device, label, size),
            part,
        ))
    };
    let [x, y, b] = [(); 3].map(|()| retained("JPEG XL patched LF prediction", plane_bytes));
    let output = ProgressiveDcOutput {
        xyb: ProgressiveDcXybPlanes::from_leases(
            [x?, y?, b?],
            extent.width,
            extent.height,
            extent.width,
        )?,
        extras: if extra_bytes != 0 {
            Some(ProgressiveDcExtras {
                buffer: retained("JPEG XL patched LF extras", extra_bytes)?,
                planes: extra_planes
                    .iter()
                    .enumerate()
                    .map(
                        |(index, plane)| crate::modular_transform::GpuModularChannelLayout {
                            word_offset: index as u32 * (layout.plane_bytes / 4) as u32,
                            row_stride_words: extent.width,
                            ..*plane
                        },
                    )
                    .collect(),
            })
        } else {
            None
        },
    };
    let working = storage(device, "JPEG XL LF patch working planes", render.size);
    let scratch = storage(device, "JPEG XL LF patch channel snapshot", render.size);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for (channel, plane) in inputs.iter().enumerate() {
        let destination = channel as u64 * layout.plane_bytes;
        if plane.stride == extent.width {
            encoder.copy_buffer_to_buffer(
                plane.buffer.as_wgpu_buffer(),
                u64::from(plane.offset) * 4,
                &working,
                destination,
                plane_bytes,
            );
        } else {
            for row in 0..extent.height {
                encoder.copy_buffer_to_buffer(
                    plane.buffer.as_wgpu_buffer(),
                    (u64::from(plane.offset) + u64::from(row) * u64::from(plane.stride)) * 4,
                    &working,
                    destination + u64::from(row) * u64::from(extent.width) * 4,
                    u64::from(extent.width) * 4,
                );
            }
        }
    }
    let uniforms = render.record(
        &mut encoder,
        &working,
        &scratch,
        source.xyb.planes[0].buffer.as_wgpu_buffer(),
    );
    for (channel, plane) in output.xyb.planes.iter().enumerate() {
        encoder.copy_buffer_to_buffer(
            &working,
            channel as u64 * layout.plane_bytes,
            plane.buffer.as_wgpu_buffer(),
            0,
            plane_bytes,
        );
    }
    if let Some(extra) = &output.extras {
        for (channel, plane) in extra.planes.iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                &working,
                (3 + channel) as u64 * layout.plane_bytes,
                extra.buffer.as_wgpu_buffer(),
                u64::from(plane.word_offset) * 4,
                plane_bytes,
            );
        }
    }
    submit_recorded(
        backend,
        encoder,
        output,
        render.inputs(inputs.iter().map(|plane| plane.buffer.clone())),
        (working, scratch, uniforms),
        permit,
        poll,
    )
}

struct Plane<'a> {
    buffer: &'a GpuBufferLease,
    offset: u32,
    stride: u32,
    width: u32,
    height: u32,
}

impl Plane<'_> {
    fn validate(&self, extent: Extent2d) -> Result<()> {
        let bytes = (u64::from(self.offset)
            + u64::from(extent.height - 1) * u64::from(self.stride)
            + u64::from(extent.width))
            * 4;
        if self.width != extent.width
            || self.height != extent.height
            || self.stride < extent.width
            || self.buffer.size() < bytes
            || !self
                .buffer
                .as_wgpu_buffer()
                .usage()
                .contains(wgpu::BufferUsages::COPY_SRC)
        {
            return Err(Error::EngineContract("invalid LF patch input plane"));
        }
        Ok(())
    }
}
