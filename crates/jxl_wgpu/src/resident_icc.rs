//! Ordered ICC processing on resident planar F32 channels.
//!
//! Device values follow the ICC unit-domain/range rules. PCS matrix intermediates remain
//! unclipped. Alpha and extra planes are not included in the supplied color channel views.
//! This is a resident execution primitive; callers admit its explicit memory plan and retain
//! the program and returned dispatch resources through GPU completion, as for other resident
//! codec stages. Dynamic connections require checking a mapped metadata status word. No image
//! samples are read back and this primitive does not submit commands.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::{
    Extent2d,
    icc::{IccCurve, IccCurveKind, IccInverseDirection, IccStage, IccTransform},
};
use wgpu::util::DeviceExt;

use crate::{KernelVariant, ResidentStorageBinding};

const ICC_SHADER: &str = concat!(
    include_str!("../shaders/image_transfer.wgsl"),
    "\n",
    include_str!("../shaders/icc.wgsl"),
);

mod program;
use program::{lower_program, program_size};

/// Scalar offsets relative to a storage binding; row padding is left untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentIccPlane {
    pub offset: u32,
    pub stride: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ResidentIccInputs<'a> {
    pub input: ResidentStorageBinding<'a>,
    pub output: ResidentStorageBinding<'a>,
    pub extent: Extent2d,
    /// One plane per selected device channel. Every source value must be finite.
    pub input_planes: &'a [ResidentIccPlane],
    pub output_planes: &'a [ResidentIccPlane],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentIccMemoryPlan {
    pub program_bytes: u64,
    pub dispatch_bytes: u64,
    /// A mapped status word for a GPU-derived connection; zero for static connections.
    pub validation_bytes: u64,
}

impl ResidentIccMemoryPlan {
    /// Computes and checks physical allocations before upload. Shared channel curves use one
    /// descriptor/table. A program is reusable across arbitrary frame extents and row pitches.
    pub fn new(transform: &IccTransform, limits: &wgpu::Limits) -> Result<Self, ResidentIccError> {
        validate_capabilities(limits)?;
        let max_channels = transform.program().max_channels();
        if max_channels > MAX_CHANNELS {
            return Err(ResidentIccError::Limit {
                resource: "processing channels",
                required: max_channels as u64,
                available: MAX_CHANNELS as u64,
            });
        }
        let limit = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size)
            .min(u64::from(u32::MAX));
        let bytes = program_size(transform, limit)?;
        Ok(Self {
            program_bytes: bytes,
            dispatch_bytes: std::mem::size_of::<DispatchParams>() as u64,
            validation_bytes: if transform
                .program()
                .stages()
                .iter()
                .any(|stage| matches!(stage, IccStage::BlackPointConnection(_)))
            {
                4
            } else {
                0
            },
        })
    }

    #[must_use]
    pub const fn transient_bytes(self) -> u64 {
        self.dispatch_bytes + self.validation_bytes
    }
}

/// Completion-owned parameters and optional validation readback. If `validation_buffer` is
/// present, map it after submission and call `validate_status` before accepting output pixels.
#[must_use]
#[derive(Debug)]
pub struct ResidentIccDispatch {
    parameters: wgpu::Buffer,
    validation: Option<wgpu::Buffer>,
}

impl ResidentIccDispatch {
    #[must_use]
    pub const fn parameters(&self) -> &wgpu::Buffer {
        &self.parameters
    }

    #[must_use]
    pub const fn validation_buffer(&self) -> Option<&wgpu::Buffer> {
        self.validation.as_ref()
    }

    pub fn validate_status(bytes: &[u8]) -> Result<(), ResidentIccError> {
        if bytes != [0; 4] {
            return Err(ResidentIccError::Precision);
        }
        Ok(())
    }
}

/// Immutable uploaded metadata. Callers retain this handle for every recorded dispatch.
#[derive(Debug)]
pub struct ResidentIccProgram {
    buffer: wgpu::Buffer,
    input_channels: usize,
    output_channels: usize,
    memory: ResidentIccMemoryPlan,
}

impl ResidentIccProgram {
    pub fn new(device: &wgpu::Device, transform: &IccTransform) -> Result<Self, ResidentIccError> {
        let memory = ResidentIccMemoryPlan::new(transform, &device.limits())?;
        let bytes = lower_program(transform, memory)?;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu ICC program"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        Ok(Self {
            buffer,
            input_channels: transform.source().channels(),
            output_channels: transform.target().channels(),
            memory,
        })
    }

    #[must_use]
    pub const fn memory_plan(&self) -> ResidentIccMemoryPlan {
        self.memory
    }

    #[must_use]
    pub const fn input_channels(&self) -> usize {
        self.input_channels
    }

    #[must_use]
    pub const fn output_channels(&self) -> usize {
        self.output_channels
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResidentIccError {
    #[error("ICC {resource} requires {required}, available {available}")]
    Limit {
        resource: &'static str,
        required: u64,
        available: u64,
    },
    #[error("ICC processing metadata cannot be represented by finite GPU F32 values")]
    Precision,
    #[error("ICC buffer addressing exceeds checked WGSL u32 arithmetic")]
    Addressing,
    #[error("ICC workgroup {variant:?} exceeds device capabilities")]
    Workgroup { variant: KernelVariant },
    #[error("ICC {role} expects {expected} channels, got {actual}")]
    Channels {
        role: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("ICC {role} plane {plane} has invalid geometry or stride")]
    Plane { role: &'static str, plane: usize },
    #[error("ICC {role} storage binding has invalid usage, alignment, size or range")]
    Binding { role: &'static str },
    #[error("ICC input and output must use distinct buffers")]
    Aliasing,
    #[error("ICC output plane ranges overlap")]
    OutputOverlap,
}

#[derive(Debug)]
pub struct ResidentIccPipeline {
    pipeline: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl ResidentIccPipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, ResidentIccError> {
        Self::with_variant(device, KernelVariant::Tile16x16)
    }

    pub fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, ResidentIccError> {
        validate_capabilities(&device.limits())?;
        variant
            .validate_for("resident_icc", &device.limits(), 0)
            .map_err(|_| ResidentIccError::Workgroup { variant })?;
        let (x, y) = variant.workgroup_size();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu ICC shader"),
            source: wgpu::ShaderSource::Wgsl(ICC_SHADER.into()),
        });
        let bindings = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jxl-wgpu ICC storage layout"),
            entries: &std::array::from_fn::<_, 4, _>(|binding| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: matches!(binding, 0 | 2),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jxl-wgpu ICC pipeline layout"),
            bind_group_layouts: &[Some(&bindings)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu ICC processing"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("wg_x", f64::from(x)), ("wg_y", f64::from(y))],
                ..Default::default()
            },
            cache: None,
        });
        let prepare = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu ICC source-black detection"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("prepare_black_point"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            pipeline,
            prepare,
            variant,
        })
    }

    /// Validates views, records optional source-black detection and then pixel conversion.
    /// The returned resources belong to the caller's admitted transient allocation. A dynamic
    /// connection's mapped status must be validated before its output becomes authoritative.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        program: &ResidentIccProgram,
        inputs: ResidentIccInputs<'_>,
    ) -> Result<ResidentIccDispatch, ResidentIccError> {
        let limits = device.limits();
        let params = validate_inputs(
            inputs,
            program.input_channels,
            program.output_channels,
            &limits,
        )?;
        let (x, y) = self.variant.workgroup_size();
        let groups = [
            inputs.extent.width.div_ceil(x),
            inputs.extent.height.div_ceil(y),
        ];
        for required in groups {
            if required > limits.max_compute_workgroups_per_dimension {
                return Err(ResidentIccError::Limit {
                    resource: "workgroup count",
                    required: u64::from(required),
                    available: u64::from(limits.max_compute_workgroups_per_dimension),
                });
            }
        }
        let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu ICC dispatch"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu ICC bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inputs.input.resource(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inputs.output.resource(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: program.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: parameters.as_entire_binding(),
                },
            ],
        });
        let validation = if program.memory.validation_bytes != 0 {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jxl-wgpu ICC connection validation"),
                size: program.memory.validation_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu ICC source-black detection"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prepare);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);
            Some(buffer)
        } else {
            None
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu ICC processing"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], 1);
        drop(pass);
        if let Some(validation) = &validation {
            encoder.copy_buffer_to_buffer(
                &parameters,
                std::mem::offset_of!(DispatchParams, status) as u64,
                validation,
                0,
                4,
            );
        }
        Ok(ResidentIccDispatch {
            parameters,
            validation,
        })
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CurveParams {
    selectors: [u32; 4],
    parameters: [[f32; 4]; 2],
}

const MAX_CHANNELS: usize = 16;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DispatchParams {
    extent_channels: [u32; 4],
    input_offsets: [u32; MAX_CHANNELS],
    input_strides: [u32; MAX_CHANNELS],
    output_offsets: [u32; MAX_CHANNELS],
    output_strides: [u32; MAX_CHANNELS],
    connection_scale: [f32; 4],
    connection_offset: [f32; 3],
    status: u32,
}

fn validate_capabilities(limits: &wgpu::Limits) -> Result<(), ResidentIccError> {
    for (resource, required, available) in [
        (
            "storage bindings",
            4,
            u64::from(limits.max_storage_buffers_per_shader_stage),
        ),
        ("bind groups", 1, u64::from(limits.max_bind_groups)),
        (
            "binding slots",
            4,
            u64::from(limits.max_bindings_per_bind_group),
        ),
        (
            "dispatch storage bytes",
            std::mem::size_of::<DispatchParams>() as u64,
            limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
        ),
    ] {
        if required > available {
            return Err(ResidentIccError::Limit {
                resource,
                required,
                available,
            });
        }
    }
    Ok(())
}

fn validate_inputs(
    inputs: ResidentIccInputs<'_>,
    source_channels: usize,
    target_channels: usize,
    limits: &wgpu::Limits,
) -> Result<DispatchParams, ResidentIccError> {
    for (role, planes, expected, binding) in [
        ("input", inputs.input_planes, source_channels, inputs.input),
        (
            "output",
            inputs.output_planes,
            target_channels,
            inputs.output,
        ),
    ] {
        if planes.len() != expected {
            return Err(ResidentIccError::Channels {
                role,
                expected,
                actual: planes.len(),
            });
        }
        let alignment = u64::from(limits.min_storage_buffer_offset_alignment).max(4);
        let end = binding
            .offset
            .checked_add(binding.size.get())
            .ok_or(ResidentIccError::Addressing)?;
        if !binding.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            || !binding.offset.is_multiple_of(alignment)
            || !binding.size.get().is_multiple_of(4)
            || end > binding.buffer.size()
            || binding.size.get() > limits.max_storage_buffer_binding_size
        {
            return Err(ResidentIccError::Binding { role });
        }
        for (plane, view) in planes.iter().enumerate() {
            if inputs.extent.is_empty() || view.stride < inputs.extent.width {
                return Err(ResidentIccError::Plane { role, plane });
            }
            let end = plane_end(*view, inputs.extent)?;
            if end * 4 > binding.size.get() {
                return Err(ResidentIccError::Limit {
                    resource: role,
                    required: end * 4,
                    available: binding.size.get(),
                });
            }
        }
    }
    if inputs.input.buffer == inputs.output.buffer {
        return Err(ResidentIccError::Aliasing);
    }
    for (i, lhs) in inputs.output_planes.iter().enumerate() {
        for rhs in &inputs.output_planes[..i] {
            if u64::from(lhs.offset) < plane_end(*rhs, inputs.extent)?
                && u64::from(rhs.offset) < plane_end(*lhs, inputs.extent)?
            {
                return Err(ResidentIccError::OutputOverlap);
            }
        }
    }
    let offsets = |planes: &[ResidentIccPlane]| {
        std::array::from_fn(|i| planes.get(i).map_or(0, |plane| plane.offset))
    };
    let strides = |planes: &[ResidentIccPlane]| {
        std::array::from_fn(|i| planes.get(i).map_or(0, |plane| plane.stride))
    };
    Ok(DispatchParams {
        extent_channels: [
            inputs.extent.width,
            inputs.extent.height,
            source_channels as u32,
            target_channels as u32,
        ],
        input_offsets: offsets(inputs.input_planes),
        input_strides: strides(inputs.input_planes),
        output_offsets: offsets(inputs.output_planes),
        output_strides: strides(inputs.output_planes),
        connection_scale: [1.0; 4],
        connection_offset: [0.0; 3],
        status: 0,
    })
}

fn plane_end(plane: ResidentIccPlane, extent: Extent2d) -> Result<u64, ResidentIccError> {
    let end = u64::from(plane.offset)
        + u64::from(extent.height.saturating_sub(1)) * u64::from(plane.stride)
        + u64::from(extent.width);
    if end > u64::from(u32::MAX) {
        return Err(ResidentIccError::Addressing);
    }
    Ok(end)
}

const _: () = {
    assert!(std::mem::size_of::<CurveParams>() == 48);
    assert!(std::mem::offset_of!(CurveParams, parameters) == 16);
    assert!(std::mem::size_of::<DispatchParams>() == 304);
    assert!(std::mem::offset_of!(DispatchParams, input_offsets) == 16);
    assert!(std::mem::offset_of!(DispatchParams, output_strides) == 208);
    assert!(std::mem::offset_of!(DispatchParams, connection_scale) == 272);
    assert!(std::mem::offset_of!(DispatchParams, status) == 300);
};

#[cfg(test)]
mod tests;
