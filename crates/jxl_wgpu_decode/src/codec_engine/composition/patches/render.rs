use bytemuck::{Pod, Zeroable};
use jxl_wgpu::{GpuBufferLease, WgpuBackend};
use wgpu::util::DeviceExt;

use super::super::gpu::{Surface, dispatch, pipeline};
use super::super::submission::{GpuWork, completion_fence_bytes, submit_recorded, validate_size};
use super::Dictionary;
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

pub(in super::super) fn render(
    backend: &WgpuBackend,
    source: &Surface,
    references: &[Option<Surface>; 4],
    dictionary: &Dictionary,
    extra_count: u32,
    has_alpha: bool,
) -> Result<GpuWork> {
    let device = backend.device();
    if source.encoding != crate::frame_surface::FrameSurfaceEncoding::Encoded {
        return Err(Error::EngineContract(
            "patch input must precede color conversion",
        ));
    }
    let channels = extra_count
        .checked_add(3)
        .ok_or_else(|| Error::backend("patch channels overflow"))?;
    let size = u64::from(source.plane_words) * u64::from(channels) * 4;
    validate_size(device, size)?;
    let batches = dictionary.count.div_ceil(64);
    let uniform_bytes = u64::from(batches) * 112;
    let poll = backend.submission_poller().try_reserve()?;
    let output_permit = backend.transient_memory_budget().try_reserve(size)?;
    let permit = backend
        .transient_memory_budget()
        .try_reserve(size + uniform_bytes + completion_fence_bytes())?;
    let output = GpuBufferLease::from_tracked(
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("JPEG XL patched frame"),
            size,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        }),
        output_permit,
    );
    let scratch = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("JPEG XL patch channel snapshot"),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let pipeline = pipeline(
        device,
        "JPEG XL patch rendering",
        include_str!("render.wgsl"),
        &[],
    );
    let shape = dispatch(
        device,
        u64::from(source.extent.width) * u64::from(source.extent.height),
    )?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(
        source.buffer.as_wgpu_buffer(),
        0,
        output.as_wgpu_buffer(),
        0,
        size,
    );
    let mut uniforms = Vec::with_capacity(batches as usize);
    for batch in 0..batches {
        let params = Params {
            canvas: [
                source.extent.width,
                source.extent.height,
                source.plane_words,
                channels,
            ],
            jobs: [
                batch * 64,
                dictionary.count.min((batch + 1) * 64),
                dictionary.stride,
                shape[0] * 64,
            ],
            flags: [u32::from(has_alpha), 0, 0, 0],
            references: references.each_ref().map(|slot| {
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
        let mut entries: Vec<_> = references
            .iter()
            .enumerate()
            .map(|(index, slot)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: slot
                    .as_ref()
                    .map_or(&source.buffer, |surface| &surface.buffer)
                    .as_wgpu_buffer()
                    .as_entire_binding(),
            })
            .collect();
        for (binding, buffer) in [
            (4, dictionary.commands.as_wgpu_buffer()),
            (5, output.as_wgpu_buffer()),
            (6, &scratch),
            (7, &uniform),
        ] {
            entries.push(wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            });
        }
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JPEG XL patch render bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(shape[0], shape[1], 1);
        }
        uniforms.push(uniform);
    }
    let inputs = [source.buffer.clone(), dictionary.commands.clone()]
        .into_iter()
        .chain(
            references
                .iter()
                .flatten()
                .map(|surface| surface.buffer.clone()),
        )
        .collect();
    submit_recorded(
        backend,
        encoder,
        output,
        inputs,
        (scratch, uniforms),
        permit,
        poll,
    )
}
