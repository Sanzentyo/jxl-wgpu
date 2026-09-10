//! Independently admitted images at Modular global/LF and residual-pass boundaries.

use crate::modular_inverse::ModularInverseJob;
use crate::modular_rct::ModularRctParams;
use crate::modular_squeeze::ModularSqueezeParams;
use crate::profile::{ModularPassBoundary, StandardModularProfile};
use crate::{Error, Result};

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use jxl_wgpu::{GpuBufferLease, MemoryPermit, ResidentStorageBinding, WgpuBackend};

use super::execution::{OutputPlan, SubmitPipelines, encode_modular_inverse_jobs};
use super::lifetime::{DecodeJobLifetime, DecodeSource, MapCompletion};
use super::render::{ModularRenderInput, ModularRenderTarget, encode_modular_output};
use super::types::{F64OutputPath, NATIVE_F64_DUMMY_WORD_BYTES};
use crate::buffer_pool::{DecodeBufferLease, DecodeBufferPool};

pub(super) struct IntermediateFrame {
    pub output: GpuBufferLease,
    pub boundary: ModularPassBoundary,
    pub completion: Arc<MapCompletion>,
    pub status_staging: DecodeBufferLease,
    status: DecodeBufferLease,
    arena: Option<DecodeBufferLease>,
    native_f64_dummy: Option<DecodeBufferLease>,
    render: Option<crate::modular_render::ModularRenderBuffers>,
    render_uniforms: Mutex<Vec<wgpu::Buffer>>,
    mapped: AtomicBool,
    _transient: MemoryPermit,
}

impl Drop for IntermediateFrame {
    fn drop(&mut self) {
        if self.mapped.swap(false, Ordering::AcqRel) {
            self.status_staging.buffer().unmap();
        }
    }
}

pub(super) fn allocate_intermediates(
    backend: &WgpuBackend,
    source: &DecodeSource,
    buffers: &Arc<DecodeBufferPool>,
    permit: &mut MemoryPermit,
) -> Result<VecDeque<Arc<IntermediateFrame>>> {
    let memory = source.dispatch_layout.intermediate_memory;
    let device = backend.device();
    source
        .profile
        .intermediate_passes
        .iter()
        .map(|&boundary| {
            let output_permit = permit
                .split_off(memory.output_bytes)
                .map_err(Error::backend)?;
            let transient = permit
                .split_off(memory.transient_bytes)
                .map_err(Error::backend)?;
            let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            let mut usage = storage | wgpu::BufferUsages::COPY_SRC;
            if backend.direct_readback_enabled() {
                usage |= wgpu::BufferUsages::MAP_READ;
            }
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jxl-wgpu immutable Modular intermediate image"),
                size: memory.output_bytes,
                usage,
                mapped_at_creation: false,
            });
            let arena = source.profile.resident_frame_plan.as_ref().map(|frame| {
                buffers.checkout(
                    "jxl-wgpu Modular intermediate inverse arena",
                    frame.inverse_plan.arena_bytes(),
                    storage,
                    4,
                )
            });
            let render = if arena.is_some() {
                source
                    .output
                    .render
                    .as_ref()
                    .map(|plan| plan.allocate(device))
                    .transpose()?
            } else {
                None
            };
            let native_f64_dummy = (arena.is_some()
                && source.output.f64_output_path == Some(F64OutputPath::NativeArithmetic))
            .then(|| {
                buffers.checkout(
                    "jxl-wgpu Modular intermediate F64 dummy",
                    NATIVE_F64_DUMMY_WORD_BYTES,
                    storage,
                    4,
                )
            });
            Ok(Arc::new(IntermediateFrame {
                output: GpuBufferLease::from_tracked(output, output_permit),
                boundary,
                completion: Arc::new(MapCompletion::default()),
                status_staging: buffers.checkout(
                    "jxl-wgpu Modular intermediate status readback",
                    source.dispatch_layout.status_bytes,
                    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    4,
                ),
                status: buffers.checkout(
                    "jxl-wgpu Modular intermediate status",
                    source.dispatch_layout.status_bytes,
                    storage | wgpu::BufferUsages::COPY_SRC,
                    4,
                ),
                arena,
                native_f64_dummy,
                render,
                render_uniforms: Mutex::new(Vec::new()),
                mapped: AtomicBool::new(false),
                _transient: transient,
            }))
        })
        .collect()
}

pub(super) fn encode_intermediate(
    device: &wgpu::Device,
    commands: &mut wgpu::CommandEncoder,
    source: &DecodeSource,
    pipelines: &SubmitPipelines,
    lifetime: &Arc<DecodeJobLifetime>,
    image: &Arc<IntermediateFrame>,
) -> Result<Vec<wgpu::Buffer>> {
    // Finalizer errors belong to this immutable boundary, never to the continued entropy state.
    commands.copy_buffer_to_buffer(
        lifetime._status.buffer(),
        0,
        image.status.buffer(),
        0,
        source.dispatch_layout.status_bytes,
    );
    let mut uniforms = Vec::new();
    match (&source.profile.resident_frame_plan, &image.arena) {
        (Some(frame), Some(arena)) => {
            let original = lifetime._frame_arena.as_ref().ok_or(Error::EngineContract(
                "Modular intermediate lost its assembly arena",
            ))?;
            let inverse = pipelines.inverse.as_deref().ok_or(Error::EngineContract(
                "Modular intermediate lost its inverse pipelines",
            ))?;
            commands.copy_buffer_to_buffer(
                original.buffer(),
                0,
                arena.buffer(),
                0,
                frame.inverse_plan.arena_bytes(),
            );
            commands.clear_buffer(image.output.as_wgpu_buffer(), 0, None);
            let storage = ResidentStorageBinding::entire(arena.buffer())
                .map_err(|_| Error::EngineContract("empty Modular intermediate arena"))?;
            uniforms.extend(encode_modular_inverse_jobs(
                device,
                commands,
                storage,
                &frame.inverse_plan,
                frame.wp_header,
                inverse,
            )?);
            uniforms.extend(encode_modular_output(
                device,
                commands,
                &source.output,
                inverse,
                ModularRenderTarget {
                    output: image.output.as_wgpu_buffer(),
                    status: image.status.buffer(),
                    native_f64_dummy: image
                        .native_f64_dummy
                        .as_ref()
                        .map(DecodeBufferLease::buffer),
                    render: image.render.as_ref(),
                    render_uniforms: &image.render_uniforms,
                },
                ModularRenderInput {
                    arena: storage,
                    planes: &frame.inverse_plan.final_gpu_layouts(),
                    params: source.finalize_params.first().ok_or(Error::EngineContract(
                        "Modular intermediate lost its output parameters",
                    ))?,
                },
            )?);
        }
        (None, None) => commands.copy_buffer_to_buffer(
            lifetime.output.as_wgpu_buffer(),
            0,
            image.output.as_wgpu_buffer(),
            0,
            source.output.storage_bytes()?,
        ),
        _ => {
            return Err(Error::EngineContract(
                "Modular intermediate arena plan changed",
            ));
        }
    }
    commands.copy_buffer_to_buffer(
        image.status.buffer(),
        0,
        image.status_staging.buffer(),
        0,
        source.dispatch_layout.status_bytes,
    );
    let retained_job = Arc::clone(lifetime);
    let retained_image = Arc::clone(image);
    let completion = Arc::clone(&image.completion);
    commands.map_buffer_on_submit(
        image.status_staging.buffer(),
        wgpu::MapMode::Read,
        ..,
        move |result| {
            if result.is_ok() {
                retained_image.mapped.store(true, Ordering::Release);
            }
            drop(retained_job);
            drop(retained_image);
            completion.complete(
                result.map_err(|error| format!("Modular intermediate mapping failed: {error}")),
            );
        },
    );
    Ok(uniforms)
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct IntermediateMemory {
    pub count: usize,
    pub output_bytes: u64,
    pub transient_bytes: u64,
}

impl IntermediateMemory {
    pub(super) fn new(
        profile: &StandardModularProfile,
        output: &OutputPlan,
        status_bytes: u64,
        finalizer_uniform_bytes: u64,
    ) -> Result<Self> {
        if profile.intermediate_passes.is_empty() {
            return Ok(Self::default());
        }
        let mut bytes = vec![status_bytes, status_bytes];
        if let Some(frame) = &profile.resident_frame_plan {
            bytes.extend([
                frame.inverse_plan.arena_bytes(),
                finalizer_uniform_bytes,
                output.render.as_ref().map_or(0, |plan| plan.total_bytes()),
                if output.f64_output_path == Some(F64OutputPath::NativeArithmetic) {
                    NATIVE_F64_DUMMY_WORD_BYTES
                } else {
                    0
                },
            ]);
            bytes.extend(frame.inverse_plan.jobs().iter().map(|job| match job {
                ModularInverseJob::Squeeze { .. } => {
                    std::mem::size_of::<ModularSqueezeParams>() as u64
                }
                ModularInverseJob::Rct { .. } => std::mem::size_of::<ModularRctParams>() as u64,
                ModularInverseJob::Palette { job } => job.uniform_bytes(),
            }));
        }
        Ok(Self {
            count: profile.intermediate_passes.len(),
            output_bytes: output.storage_bytes()?,
            transient_bytes: bytes
                .into_iter()
                .try_fold(0u64, u64::checked_add)
                .ok_or_else(|| Error::backend("Modular intermediate byte count overflow"))?,
        })
    }

    pub(super) fn total_output_bytes(self) -> Result<u64> {
        self.multiply(self.output_bytes)
    }

    pub(super) fn total_transient_bytes(self) -> Result<u64> {
        self.multiply(self.transient_bytes)
    }

    pub(super) fn total_bytes(self) -> Result<u64> {
        self.total_output_bytes()?
            .checked_add(self.total_transient_bytes()?)
            .ok_or_else(|| Error::backend("Modular intermediate total byte count overflow"))
    }

    fn multiply(self, bytes: u64) -> Result<u64> {
        u64::try_from(self.count)
            .ok()
            .and_then(|count| count.checked_mul(bytes))
            .ok_or_else(|| Error::backend("Modular intermediate image count overflow"))
    }
}
