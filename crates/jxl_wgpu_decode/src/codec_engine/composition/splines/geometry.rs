//! Bounded geometry generation with exact admission for the ordered tile cache.

use std::sync::Arc;
use std::task::{Context, Poll};

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{GpuBufferLease, MemoryPermit, SubmissionPollPermit, WgpuBackend};

use super::super::entropy_program::DecodedProgram;
use super::super::submission::{Completion, validate_size};
use crate::{Error, Result, SplineResource};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

const STATE_BYTES: u64 = 256;
const STATUS_BYTES: u64 = 32;
const SPLAT_BYTES: u64 = 48;
const TILE_SIDE: u32 = 32;
const GEOMETRY_STEPS: u32 = 1 << 24;
const READY: u32 = 3;
const EMIT: u32 = 4;
const DONE: u32 = 5;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    image: [u32; 4],
    run: [u32; 4],
    correlation: [f32; 4],
    work: [u32; 4],
}
const PARAM_BYTES: u64 = std::mem::size_of::<Params>() as u64;
const _: () = assert!(PARAM_BYTES == 64);

#[derive(Debug)]
pub(in super::super) struct GeometryPlan {
    program: DecodedProgram,
    params: Params,
}

impl GeometryPlan {
    pub(in super::super) fn new(
        program: DecodedProgram,
        extent: Extent2d,
        correlation: [f32; 2],
        limits: &wgpu::Limits,
    ) -> Result<Self> {
        let tile_columns = extent.width.div_ceil(TILE_SIDE);
        let tile_count = tile_columns
            .checked_mul(extent.height.div_ceil(TILE_SIDE))
            .filter(|&count| count != 0 && count < u32::MAX / 3)
            .ok_or(Error::EngineContract("spline tile geometry exceeds u32"))?;
        if program.stride != 1 || program.count < 4 || !correlation.into_iter().all(f32::is_finite)
        {
            return Err(Error::EngineContract("spline geometry input contract"));
        }
        let storage_limit = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size)
            .min(u64::from(u32::MAX - 3));
        Ok(Self {
            program,
            params: Params {
                image: [extent.width, extent.height, tile_columns, tile_count],
                run: [
                    1,
                    0,
                    (storage_limit / SPLAT_BYTES) as u32,
                    (storage_limit / 4) as u32,
                ],
                correlation: [correlation[0], correlation[1], 0.0, 0.0],
                work: [GEOMETRY_STEPS, super::HEADER_WORDS, TILE_SIDE, 512],
            },
        })
    }

    pub(in super::super) fn submit(self, backend: WgpuBackend) -> Result<PendingGeometry> {
        let device = backend.device();
        let tile_bytes = (3 * u64::from(self.params.image[3]) + 1) * 4;
        validate_size(device, tile_bytes)?;
        let mut permit = backend
            .transient_memory_budget()
            .try_reserve(STATE_BYTES + PARAM_BYTES + STATUS_BYTES + tile_bytes + SPLAT_BYTES + 4)?;
        let poll = backend.submission_poller().try_reserve()?;
        let tiles = retained(device, &mut permit, tile_bytes)?;
        let records = retained(device, &mut permit, SPLAT_BYTES)?;
        let references = retained(device, &mut permit, 4)?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JPEG XL spline geometry"),
            source: wgpu::ShaderSource::Wgsl(include_str!("geometry.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("JPEG XL spline geometry"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let resources = Arc::new(Resources {
            program: self.program.commands,
            state: buffer(
                device,
                STATE_BYTES,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            params: buffer(
                device,
                PARAM_BYTES,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            status: buffer(
                device,
                STATUS_BYTES,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            _permit: permit,
        });
        let mut pending = PendingGeometry {
            backend,
            pipeline,
            resources,
            tiles,
            records,
            references,
            params: self.params,
            expected: None,
            completion: Arc::new(Completion::default()),
            submissions: 0,
        };
        pending.submit_step(poll)?;
        Ok(pending)
    }
}

fn buffer(device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("JPEG XL spline geometry storage"),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn retained(
    device: &wgpu::Device,
    permit: &mut MemoryPermit,
    bytes: u64,
) -> Result<GpuBufferLease> {
    let allocation = permit.split_off(bytes)?;
    Ok(GpuBufferLease::from_tracked(
        buffer(
            device,
            bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        ),
        allocation,
    ))
}

#[derive(Debug)]
struct Resources {
    program: GpuBufferLease,
    state: wgpu::Buffer,
    params: wgpu::Buffer,
    status: wgpu::Buffer,
    _permit: MemoryPermit,
}

#[derive(Debug)]
pub(in super::super) struct Cache {
    pub(super) extent: Extent2d,
    pub(super) tile_columns: u32,
    pub(super) tile_count: u32,
    pub(super) max_population: u32,
    pub(super) records: GpuBufferLease,
    pub(super) references: GpuBufferLease,
    pub(super) tiles: GpuBufferLease,
}

#[derive(Debug)]
pub(in super::super) struct PendingGeometry {
    backend: WgpuBackend,
    pipeline: wgpu::ComputePipeline,
    resources: Arc<Resources>,
    tiles: GpuBufferLease,
    records: GpuBufferLease,
    references: GpuBufferLease,
    params: Params,
    expected: Option<[u32; 3]>,
    completion: Arc<Completion>,
    pub(in super::super) submissions: usize,
}

impl PendingGeometry {
    fn submit_step(&mut self, poll: SubmissionPollPermit) -> Result<()> {
        let guards = [
            &self.resources.program,
            &self.tiles,
            &self.records,
            &self.references,
        ]
        .into_iter()
        .map(GpuBufferLease::try_acquire_gpu_submission)
        .collect::<jxl_wgpu::Result<Vec<_>>>()?;
        self.backend.queue().write_buffer(
            &self.resources.params,
            0,
            bytemuck::bytes_of(&self.params),
        );
        let buffers = [
            self.resources.program.as_wgpu_buffer(),
            &self.resources.state,
            self.tiles.as_wgpu_buffer(),
            self.records.as_wgpu_buffer(),
            self.references.as_wgpu_buffer(),
            &self.resources.params,
        ];
        let entries: Vec<_> = buffers
            .into_iter()
            .enumerate()
            .map(|(index, buffer)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bindings = self
            .backend
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("JPEG XL spline geometry"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
        let mut encoder = self
            .backend
            .device()
            .create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.resources.state,
            0,
            &self.resources.status,
            0,
            STATUS_BYTES,
        );
        let completion = Arc::new(Completion::default());
        let done = Arc::clone(&completion);
        let retained = (
            Arc::clone(&self.resources),
            self.tiles.clone(),
            self.records.clone(),
            self.references.clone(),
        );
        let submission = self.backend.queue().submit([encoder.finish()]);
        drop(guards);
        self.resources
            .status
            .map_async(wgpu::MapMode::Read, .., move |result| {
                drop(retained);
                done.complete(result.map_err(|error| error.to_string()));
            });
        let failed = Arc::clone(&completion);
        poll.register(submission, move |error| failed.complete(Err(error)))?;
        self.completion = completion;
        self.submissions += 1;
        self.params.run[0] = 0;
        Ok(())
    }

    fn advance(&mut self) -> Result<Option<Option<Arc<Cache>>>> {
        let view = self
            .resources
            .status
            .get_mapped_range(..)
            .map_err(Error::backend)?;
        let status: [u32; 8] = bytemuck::pod_read_unaligned(&view);
        drop(view);
        self.resources.status.unmap();
        let resource = match status[0] {
            16 => Some((SplineResource::GeometrySteps, self.params.work[0])),
            17 => Some((SplineResource::DrawRecords, self.params.run[2])),
            18 => Some((SplineResource::TileReferences, self.params.run[3])),
            _ => None,
        };
        if let Some((resource, limit)) = resource {
            return Err(Error::SplineResourceLimit {
                resource,
                limit: u64::from(limit),
            });
        }
        if status[0] != 0 {
            return Err(Error::SplineGeometry { code: status[0] });
        }
        let counts = [status[2], status[3], status[4]];
        match status[1] {
            READY => {
                if self.expected.is_some() {
                    return Err(Error::EngineContract(
                        "spline geometry replay returned to count phase",
                    ));
                }
                if counts[0] == 0 {
                    if counts[1] != 0 || counts[2] != 0 {
                        return Err(Error::EngineContract(
                            "empty spline cache has tile references",
                        ));
                    }
                    return Ok(Some(None));
                }
                let record_bytes = u64::from(counts[0]) * SPLAT_BYTES;
                let reference_bytes = u64::from(counts[1]) * 4;
                for size in [record_bytes, reference_bytes] {
                    validate_size(self.backend.device(), size)?;
                }
                let mut permit = self
                    .backend
                    .transient_memory_budget()
                    .try_reserve(record_bytes + reference_bytes)?;
                self.records = retained(self.backend.device(), &mut permit, record_bytes)?;
                self.references = retained(self.backend.device(), &mut permit, reference_bytes)?;
                self.params.run = [1, EMIT, counts[0], counts[1]];
                self.expected = Some(counts);
            }
            DONE => {
                if self.expected != Some(counts) {
                    return Err(Error::EngineContract(
                        "spline geometry replay changed its allocation counts",
                    ));
                }
                return Ok(Some(Some(Arc::new(Cache {
                    extent: Extent2d::new(self.params.image[0], self.params.image[1]),
                    tile_columns: self.params.image[2],
                    tile_count: self.params.image[3],
                    max_population: counts[2],
                    records: self.records.clone(),
                    references: self.references.clone(),
                    tiles: self.tiles.clone(),
                }))));
            }
            0..=2 | EMIT => {}
            _ => return Err(Error::EngineContract("unknown spline geometry phase")),
        }
        self.submit_step(self.backend.submission_poller().try_reserve()?)?;
        Ok(None)
    }

    pub(in super::super) fn poll(
        &mut self,
        context: &Context<'_>,
    ) -> Poll<Result<Option<Arc<Cache>>>> {
        loop {
            match self.completion.poll(context) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(result) => result?,
            }
            if let Some(cache) = self.advance()? {
                return Poll::Ready(Ok(cache));
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn complete_submission(&mut self) -> Result<Option<Option<Arc<Cache>>>> {
        self.completion.wait()?;
        self.advance()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(in super::super) fn wait(mut self) -> Result<(Option<Arc<Cache>>, usize)> {
        loop {
            if let Some(cache) = self.complete_submission()? {
                return Ok((cache, self.submissions));
            }
        }
    }
}
