//! Budgeted submissions. Mapping callbacks own every allocation until GPU completion.
use super::super::{execution::MapCompletion, jpeg::GpuJpegCoefficients};
use super::{Error, add, check, storage_bytes};
use jxl_wgpu::{GpuBufferLease, MemoryBudget, MemoryPermit, WgpuBackend};
use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::Context,
};

mod assemble;
mod scan;
pub(super) use assemble::Assembly;
pub(super) use scan::{Scan, ScanPhase};
const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

#[derive(Debug)]
pub(crate) struct Pipelines {
    entropy_layout: wgpu::BindGroupLayout,
    entropy: Vec<wgpu::ComputePipeline>,
    assembly_layout: wgpu::BindGroupLayout,
    assembly: Vec<wgpu::ComputePipeline>,
}
impl Pipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        fn make(
            device: &wgpu::Device,
            source: &'static str,
            kinds: &[Option<bool>],
            names: &[&str],
        ) -> (wgpu::BindGroupLayout, Vec<wgpu::ComputePipeline>) {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("JPEG reconstruction"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let entries: Vec<_> = kinds
                .iter()
                .enumerate()
                .map(|(binding, kind)| wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: match kind {
                            None => wgpu::BufferBindingType::Uniform,
                            Some(read_only) => wgpu::BufferBindingType::Storage {
                                read_only: *read_only,
                            },
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("JPEG reconstruction"),
                entries: &entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("JPEG reconstruction"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let phases = names
                .iter()
                .map(|name| {
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(name),
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: Some(name),
                        compilation_options: Default::default(),
                        cache: None,
                    })
                })
                .collect();
            (layout, phases)
        }
        let (entropy_layout, entropy) = make(
            device,
            include_str!("entropy.wgsl"),
            &[
                None,
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                None,
                Some(false),
            ],
            &[
                "count_blocks",
                "prepare_block_sizes",
                "finish_block_offsets",
                "emit_blocks",
                "count_bytes",
                "finish_byte_offsets",
                "pack_bytes",
                "scan_groups",
                "add_scan_carries",
                "classify_blocks",
                "seed_run_heads",
                "attach_run_lengths",
                "max_scan_groups",
                "max_scan_carries",
                "commit_padding_cursor",
            ],
        );
        let (assembly_layout, assembly) = make(
            device,
            include_str!("assembly.wgsl"),
            &[
                None,
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                Some(false),
            ],
            &["copy_bytes", "patch_quantizers", "validate_coefficients"],
        );
        Self {
            entropy_layout,
            entropy,
            assembly_layout,
            assembly,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Resources {
    pub backend: WgpuBackend,
    pub memory: MemoryBudget,
    pub pipelines: Arc<Pipelines>,
    pub dispatch_width: u32,
}
impl Resources {
    fn width(&self) -> u32 {
        self.dispatch_width
    }
    fn dispatch(
        &self,
        commands: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        buffers: &[&GpuBufferLease],
        groups: u32,
    ) -> crate::Result<()> {
        check(
            "dispatch rows",
            u64::from(groups.div_ceil(self.width())),
            u64::from(
                self.backend
                    .device()
                    .limits()
                    .max_compute_workgroups_per_dimension,
            ),
        )?;
        // Even inactive lanes in a rectangular final row must have non-wrapping u32 IDs.
        check(
            "dispatch invocations",
            u64::from(groups.min(self.width())) * u64::from(groups.div_ceil(self.width())) * 64,
            u64::from(u32::MAX),
        )?;
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_wgpu_buffer().as_entire_binding(),
            })
            .collect();
        let bindings = self
            .backend
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("JPEG reconstruction"),
                layout,
                entries: &entries,
            });
        let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("JPEG reconstruction"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(groups.min(self.width()), groups.div_ceil(self.width()), 1);
        Ok(())
    }
    pub(super) fn padding(&self) -> crate::Result<GpuBufferLease> {
        Allocator::new(self, 16)?.buffer(16, STORAGE, &[])
    }
}

struct Allocator<'a> {
    resources: &'a Resources,
    permit: MemoryPermit,
}
impl<'a> Allocator<'a> {
    fn new(resources: &'a Resources, bytes: u64) -> crate::Result<Self> {
        Ok(Self {
            resources,
            permit: resources.memory.try_reserve(bytes)?,
        })
    }
    fn buffer(
        &mut self,
        size: u64,
        usage: wgpu::BufferUsages,
        parts: &[&[u8]],
    ) -> crate::Result<GpuBufferLease> {
        let size = storage_bytes(size)?.max(4);
        let limits = self.resources.backend.device().limits();
        check("GPU buffer bytes", size, limits.max_buffer_size)?;
        if usage.contains(wgpu::BufferUsages::STORAGE) {
            check(
                "GPU storage binding bytes",
                size,
                limits.max_storage_buffer_binding_size,
            )?;
        }
        let permit = self
            .permit
            .split_off(size)
            .map_err(|_| Error::Invalid("allocation plan"))?;
        let buffer = self
            .resources
            .backend
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("JPEG reconstruction"),
                size,
                usage,
                mapped_at_creation: !parts.is_empty(),
            });
        if !parts.is_empty() {
            {
                let mut mapped = buffer
                    .slice(..)
                    .get_mapped_range_mut()
                    .map_err(crate::Error::backend)?;
                let mut offset = 0;
                for part in parts {
                    let end = offset + part.len();
                    if end > mapped.len() {
                        return Err(Error::Invalid("upload plan").into());
                    }
                    mapped.slice(offset..end).copy_from_slice(part);
                    offset = end;
                }
            }
            buffer.unmap();
        }
        Ok(GpuBufferLease::from_tracked(buffer, permit))
    }
    fn uniform(&mut self, words: &[u32]) -> crate::Result<GpuBufferLease> {
        self.buffer(
            words.len() as u64 * 4,
            wgpu::BufferUsages::UNIFORM,
            &[bytemuck::cast_slice(words)],
        )
    }
    fn staging(&mut self) -> crate::Result<GpuBufferLease> {
        self.buffer(
            16,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            &[],
        )
    }
}

struct Job {
    staging: GpuBufferLease,
    _held: Vec<GpuBufferLease>,
    completion: Arc<MapCompletion>,
    mapped: AtomicBool,
}
impl Drop for Job {
    fn drop(&mut self) {
        if self.mapped.load(Ordering::Acquire) {
            self.staging.as_wgpu_buffer().unmap();
        }
    }
}
pub(super) struct Operation {
    job: Arc<Job>,
}
impl fmt::Debug for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JpegOperation").finish_non_exhaustive()
    }
}
impl Operation {
    pub(super) fn poll(&self, context: &Context<'_>) -> Option<std::result::Result<(), String>> {
        self.job.completion.poll(context)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn wait(&self) -> std::result::Result<(), String> {
        self.job.completion.wait()
    }
    pub(super) fn status(
        &self,
        mapped: std::result::Result<(), String>,
        stage: &'static str,
    ) -> crate::Result<[u32; 4]> {
        mapped.map_err(crate::Error::backend)?;
        let view = self
            .job
            .staging
            .as_wgpu_buffer()
            .slice(..)
            .get_mapped_range()
            .map_err(crate::Error::backend)?;
        let status: [u32; 4] =
            bytemuck::try_pod_read_unaligned(&view).map_err(|_| Error::Invalid("status ABI"))?;
        if status[0] != 0 {
            return Err(Error::GpuStatus { stage, status }.into());
        }
        Ok(status)
    }
}
fn submit(
    resources: &Resources,
    mut commands: wgpu::CommandEncoder,
    status: &GpuBufferLease,
    staging: GpuBufferLease,
    held: Vec<GpuBufferLease>,
    coefficients: &GpuJpegCoefficients,
) -> crate::Result<Operation> {
    let poll = resources.backend.submission_poller().try_reserve()?;
    commands.copy_buffer_to_buffer(status.as_wgpu_buffer(), 0, staging.as_wgpu_buffer(), 0, 16);
    let job = Arc::new(Job {
        staging,
        _held: held,
        completion: Arc::default(),
        mapped: AtomicBool::new(false),
    });
    let guard = coefficients.buffer().try_acquire_gpu_submission()?;
    let submission = resources.backend.queue().submit([commands.finish()]);
    drop(guard);
    let callback_job = Arc::clone(&job);
    let completion = Arc::clone(&job.completion);
    job.staging
        .as_wgpu_buffer()
        .map_async(wgpu::MapMode::Read, .., move |result| {
            if result.is_ok() {
                callback_job.mapped.store(true, Ordering::Release);
            }
            drop(callback_job);
            completion.complete(result.map_err(|error| error.to_string()));
        });
    let completion = Arc::clone(&job.completion);
    if let Err(error) = poll.register(submission, move |error| completion.complete(Err(error))) {
        job.completion.complete(Err(error.to_string()));
    }
    Ok(Operation { job })
}
fn word(value: u64) -> crate::Result<u32> {
    u32::try_from(value).map_err(|_| Error::Invalid("GPU address overflow").into())
}
