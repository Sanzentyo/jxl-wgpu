//! Direct execution of the general VarDCT renderer from GPU-resident artifacts.
//!
//! The regular scheduler accepts host-owned coefficient and task vectors. A
//! decoder entropy frontend must not read those artifacts back merely to upload
//! them again, so this module exposes the same `vardct_general.wgsl` kernel with
//! explicit resident-buffer bindings and indirect dispatch offsets.

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::TransformKind;
use wgpu::util::DeviceExt;

#[cfg(test)]
const WORKGROUP_SIZE: u32 = 64;
const GENERAL_TASK_BYTES: u64 = 64;
const INDIRECT_ARGUMENT_BYTES: u64 = 12;
const INDIRECT_STAGES: u64 = 3;

/// A checked subrange used as one WGSL storage binding.
#[derive(Clone, Copy, Debug)]
pub struct ResidentStorageBinding<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub offset: u64,
    pub size: NonZeroU64,
}

impl<'a> ResidentStorageBinding<'a> {
    /// Uses an entire non-empty buffer as one binding.
    ///
    /// # Errors
    ///
    /// Returns [`ResidentVarDctError::EmptyBinding`] for a zero-sized buffer.
    pub fn entire(buffer: &'a wgpu::Buffer) -> Result<Self, ResidentVarDctError> {
        let size = NonZeroU64::new(buffer.size()).ok_or(ResidentVarDctError::EmptyBinding {
            role: "storage buffer",
        })?;
        Ok(Self {
            buffer,
            offset: 0,
            size,
        })
    }

    fn resource(self) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.buffer,
            offset: self.offset,
            size: Some(self.size),
        })
    }
}

/// One GPU-resident F32 planar binding and its scalar geometry.
#[derive(Clone, Copy, Debug)]
pub struct ResidentF32Plane<'a> {
    pub storage: ResidentStorageBinding<'a>,
    pub width: u32,
    pub height: u32,
    /// Row stride in F32 scalars. Zero means tightly packed `width`.
    pub stride: u32,
}

impl ResidentF32Plane<'_> {
    #[must_use]
    pub const fn effective_stride(self) -> u32 {
        if self.stride == 0 {
            self.width
        } else {
            self.stride
        }
    }
}

/// Host-known fields that accompany GPU-produced tasks and coefficients.
#[derive(Clone, Copy, Debug)]
pub struct ResidentVarDctRenderConfig {
    /// Maximum task count across all compact strategy buckets. Each indirect Y dimension selects
    /// the exact live count for its bucket.
    pub task_capacity: u32,
    /// F32 scalar capacity of each transform scratch allocation.
    pub scratch_scalars: u32,
    /// Word offset of the first compact 64-byte task in `artifact`.
    pub task_word_offset: u32,
    /// Word offset of the first 16-byte strategy bucket in `artifact`.
    pub bucket_word_offset: u32,
    pub quant_offset: u32,
    pub correlation_offset: u32,
    pub lf_offset: u32,
    /// Row stride of the image-wide LF tuple grid.
    pub lf_stride: u32,
    pub correlation_width: u32,
    pub correlation_height: u32,
    pub quant_biases: [f32; 4],
}

/// GPU-resident inputs consumed by three regular inverse-transform dispatches.
#[derive(Clone, Copy, Debug)]
pub struct ResidentVarDctInputs<'a> {
    pub coefficients: ResidentStorageBinding<'a>,
    /// Complete GPU-produced artifact containing bucket descriptors and compact tasks.
    pub artifact: ResidentStorageBinding<'a>,
    pub resources: ResidentStorageBinding<'a>,
    pub outputs: [ResidentF32Plane<'a>; 3],
    pub indirect: &'a wgpu::Buffer,
    /// Byte offset of the first strategy's three `DispatchIndirectArgs` records.
    pub indirect_base_offset: u64,
    pub config: ResidentVarDctRenderConfig,
}

/// Exact transient allocation made by one resident regular-transform render.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentVarDctMemoryPlan {
    pub uniform_bytes: u64,
    pub scratch_buffer_bytes: u64,
    pub total_bytes: u64,
}

impl ResidentVarDctMemoryPlan {
    /// Computes the two F32 scratch buffers plus one fixed uniform record per strategy.
    ///
    /// # Errors
    ///
    /// Returns a typed arithmetic error when the requested capacity cannot be
    /// represented in a WebGPU buffer size.
    pub fn new(scratch_scalars: u32) -> Result<Self, ResidentVarDctError> {
        let scratch_buffer_bytes = u64::from(scratch_scalars)
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or(ResidentVarDctError::ArithmeticOverflow {
                field: "scratch buffer bytes",
            })?;
        let uniform_bytes = (TransformKind::ALL.len() as u64)
            .checked_mul(std::mem::size_of::<ResidentVarDctParams>() as u64)
            .ok_or(ResidentVarDctError::ArithmeticOverflow {
                field: "resident VarDCT uniform bytes",
            })?;
        let total_bytes = scratch_buffer_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(uniform_bytes))
            .ok_or(ResidentVarDctError::ArithmeticOverflow {
                field: "resident VarDCT transient bytes",
            })?;
        Ok(Self {
            uniform_bytes,
            scratch_buffer_bytes,
            total_bytes,
        })
    }
}

/// Scratch handles retained by a caller until its command buffer has been submitted.
#[derive(Debug)]
pub struct ResidentVarDctScratch {
    pub dequantized: wgpu::Buffer,
    pub horizontal: wgpu::Buffer,
    pub memory: ResidentVarDctMemoryPlan,
}

/// Typed validation failures for the GPU-resident VarDCT renderer seam.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResidentVarDctError {
    #[error("resident VarDCT task capacity must be nonzero")]
    ZeroTaskCapacity,
    #[error("resident VarDCT {role} binding is empty")]
    EmptyBinding { role: &'static str },
    #[error("resident VarDCT {role} buffer is missing required usage {required:?}")]
    MissingUsage {
        role: &'static str,
        required: wgpu::BufferUsages,
    },
    #[error("resident VarDCT {role} range {offset}..{end} exceeds buffer size {available}")]
    BindingRange {
        role: &'static str,
        offset: u64,
        end: u64,
        available: u64,
    },
    #[error("resident VarDCT {role} offset {offset} is not aligned to {alignment}")]
    BindingAlignment {
        role: &'static str,
        offset: u64,
        alignment: u64,
    },
    #[error("resident VarDCT {role} binding needs {required} bytes, device permits {available}")]
    StorageBindingLimit {
        role: &'static str,
        required: u64,
        available: u64,
    },
    #[error(
        "resident VarDCT artifact {role} range ends at word {required}, binding has {available} words"
    )]
    ArtifactWordRange {
        role: &'static str,
        required: u64,
        available: u64,
    },
    #[error("resident VarDCT output plane {plane} has invalid {width}x{height} stride {stride}")]
    OutputGeometry {
        plane: usize,
        width: u32,
        height: u32,
        stride: u32,
    },
    #[error("resident VarDCT output plane {plane} needs {required} bytes, binding has {available}")]
    OutputBindingSize {
        plane: usize,
        required: u64,
        available: u64,
    },
    #[error("resident VarDCT scratch needs at least {required} scalars, configured {available}")]
    ScratchCapacity { required: u64, available: u32 },
    #[error("resident VarDCT indirect offset {offset} is not four-byte aligned")]
    IndirectAlignment { offset: u64 },
    #[error("resident VarDCT indirect range {offset}..{end} exceeds buffer size {available}")]
    IndirectRange {
        offset: u64,
        end: u64,
        available: u64,
    },
    #[error("resident VarDCT arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
    #[error("resident VarDCT {resource} needs {required} bytes, device permits {available}")]
    DeviceBufferLimit {
        resource: &'static str,
        required: u64,
        available: u64,
    },
}

/// Reusable regular-transform pipelines backed by `vardct_general.wgsl`.
pub struct ResidentVarDctRenderer {
    dequantize: wgpu::ComputePipeline,
    horizontal: wgpu::ComputePipeline,
    vertical: wgpu::ComputePipeline,
    special: wgpu::ComputePipeline,
}

impl ResidentVarDctRenderer {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let module =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/vardct_general.wgsl"));
        let special_module =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/vardct_special.wgsl"));
        let pipeline = |label: &'static str, entry_point: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        Self {
            dequantize: pipeline("jxl-wgpu resident VarDCT dequantize", "dequantize"),
            horizontal: pipeline("jxl-wgpu resident VarDCT horizontal", "horizontal_idct"),
            vertical: pipeline("jxl-wgpu resident VarDCT vertical", "vertical_idct"),
            special: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu resident VarDCT special"),
                layout: None,
                module: &special_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }),
        }
    }

    /// Validates all resident ranges and records every regular and special strategy bucket.
    ///
    /// No coefficient, task, or resource bytes cross the host. The returned
    /// scratch handles make transient lifetime and byte accounting explicit.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: ResidentVarDctInputs<'_>,
    ) -> Result<ResidentVarDctScratch, ResidentVarDctError> {
        validate_inputs(device, inputs)?;
        let memory = ResidentVarDctMemoryPlan::new(inputs.config.scratch_scalars)?;
        let maximum = device.limits().max_buffer_size;
        if memory.scratch_buffer_bytes > maximum {
            return Err(ResidentVarDctError::DeviceBufferLimit {
                resource: "scratch buffer",
                required: memory.scratch_buffer_bytes,
                available: maximum,
            });
        }
        let scratch_descriptor = wgpu::BufferDescriptor {
            label: Some("jxl-wgpu resident VarDCT dequantized scratch"),
            size: memory.scratch_buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        };
        let dequantized = device.create_buffer(&scratch_descriptor);
        let horizontal = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu resident VarDCT horizontal scratch"),
            ..scratch_descriptor
        });
        for (strategy, transform) in TransformKind::ALL.into_iter().enumerate() {
            let params = resident_params(inputs, transform, strategy as u32)?;
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu resident VarDCT strategy params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let indirect = inputs
                .indirect_base_offset
                .checked_add(strategy as u64 * INDIRECT_STAGES * INDIRECT_ARGUMENT_BYTES)
                .ok_or(ResidentVarDctError::ArithmeticOverflow {
                    field: "strategy indirect offset",
                })?;
            if transform.is_special() {
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("jxl-wgpu resident special VarDCT bindings"),
                    layout: &self.special.get_bind_group_layout(0),
                    entries: &[
                        storage_entry(0, inputs.coefficients),
                        storage_entry(1, inputs.artifact),
                        storage_entry(2, inputs.resources),
                        storage_entry(5, inputs.outputs[0].storage),
                        storage_entry(6, inputs.outputs[1].storage),
                        storage_entry(7, inputs.outputs[2].storage),
                        entire_entry(8, &uniform),
                    ],
                });
                dispatch_indirect(
                    encoder,
                    "jxl-wgpu resident special VarDCT",
                    &self.special,
                    &bind_group,
                    inputs.indirect,
                    indirect,
                );
                continue;
            }

            let dequantize_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu resident VarDCT dequantize bindings"),
                layout: &self.dequantize.get_bind_group_layout(0),
                entries: &[
                    storage_entry(0, inputs.coefficients),
                    storage_entry(1, inputs.artifact),
                    storage_entry(2, inputs.resources),
                    entire_entry(3, &dequantized),
                    entire_entry(8, &uniform),
                ],
            });
            let horizontal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu resident VarDCT horizontal bindings"),
                layout: &self.horizontal.get_bind_group_layout(0),
                entries: &[
                    storage_entry(1, inputs.artifact),
                    entire_entry(3, &dequantized),
                    entire_entry(4, &horizontal),
                    entire_entry(8, &uniform),
                ],
            });
            let vertical_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu resident VarDCT vertical bindings"),
                layout: &self.vertical.get_bind_group_layout(0),
                entries: &[
                    storage_entry(1, inputs.artifact),
                    entire_entry(4, &horizontal),
                    storage_entry(5, inputs.outputs[0].storage),
                    storage_entry(6, inputs.outputs[1].storage),
                    storage_entry(7, inputs.outputs[2].storage),
                    entire_entry(8, &uniform),
                ],
            });
            for (label, pipeline, bind_group, offset) in [
                (
                    "jxl-wgpu resident VarDCT dequantize",
                    &self.dequantize,
                    &dequantize_bind_group,
                    indirect,
                ),
                (
                    "jxl-wgpu resident VarDCT horizontal",
                    &self.horizontal,
                    &horizontal_bind_group,
                    indirect + INDIRECT_ARGUMENT_BYTES,
                ),
                (
                    "jxl-wgpu resident VarDCT vertical",
                    &self.vertical,
                    &vertical_bind_group,
                    indirect + 2 * INDIRECT_ARGUMENT_BYTES,
                ),
            ] {
                dispatch_indirect(
                    encoder,
                    label,
                    pipeline,
                    bind_group,
                    inputs.indirect,
                    offset,
                );
            }
        }
        Ok(ResidentVarDctScratch {
            dequantized,
            horizontal,
            memory,
        })
    }
}

fn dispatch_indirect(
    encoder: &mut wgpu::CommandEncoder,
    label: &'static str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    indirect: &wgpu::Buffer,
    offset: u64,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups_indirect(indirect, offset);
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ResidentVarDctParams {
    task_base: u32,
    task_count: u32,
    transform_width: u32,
    transform_height: u32,
    transform_area: u32,
    lf_width: u32,
    lf_height: u32,
    quant_offset: u32,
    correlation_offset: u32,
    lf_offset: u32,
    output_width_x: u32,
    output_height_x: u32,
    output_stride_x: u32,
    output_width_y: u32,
    output_height_y: u32,
    output_stride_y: u32,
    output_width_b: u32,
    output_height_b: u32,
    output_stride_b: u32,
    transform_kind: u32,
    correlation_width: u32,
    correlation_height: u32,
    task_word_offset: u32,
    bucket_word_offset: u32,
    lf_stride: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    quant_biases: [f32; 4],
}

fn validate_inputs(
    device: &wgpu::Device,
    inputs: ResidentVarDctInputs<'_>,
) -> Result<(), ResidentVarDctError> {
    if inputs.config.task_capacity == 0 {
        return Err(ResidentVarDctError::ZeroTaskCapacity);
    }
    for (role, binding, writable) in [
        ("coefficient", inputs.coefficients, false),
        ("artifact", inputs.artifact, false),
        ("resource", inputs.resources, false),
        ("X output", inputs.outputs[0].storage, true),
        ("Y output", inputs.outputs[1].storage, true),
        ("B output", inputs.outputs[2].storage, true),
    ] {
        validate_storage_binding(device, role, binding, writable)?;
    }
    let available_words = inputs.artifact.size.get() / 4;
    let task_end = u64::from(inputs.config.task_word_offset)
        .checked_add(u64::from(inputs.config.task_capacity) * (GENERAL_TASK_BYTES / 4))
        .ok_or(ResidentVarDctError::ArithmeticOverflow {
            field: "artifact task range",
        })?;
    if task_end > available_words {
        return Err(ResidentVarDctError::ArtifactWordRange {
            role: "task",
            required: task_end,
            available: available_words,
        });
    }
    let bucket_end = u64::from(inputs.config.bucket_word_offset)
        .checked_add(TransformKind::ALL.len() as u64 * 4)
        .ok_or(ResidentVarDctError::ArithmeticOverflow {
            field: "artifact bucket range",
        })?;
    if bucket_end > available_words {
        return Err(ResidentVarDctError::ArtifactWordRange {
            role: "bucket",
            required: bucket_end,
            available: available_words,
        });
    }
    let required_scratch = inputs.coefficients.size.get().div_ceil(4);
    if required_scratch > u64::from(inputs.config.scratch_scalars) {
        return Err(ResidentVarDctError::ScratchCapacity {
            required: required_scratch,
            available: inputs.config.scratch_scalars,
        });
    }
    for (plane, output) in inputs.outputs.into_iter().enumerate() {
        let stride = output.effective_stride();
        if output.width == 0 || output.height == 0 || stride < output.width {
            return Err(ResidentVarDctError::OutputGeometry {
                plane,
                width: output.width,
                height: output.height,
                stride,
            });
        }
        let required = u64::from(output.height - 1)
            .checked_mul(u64::from(stride))
            .and_then(|value| value.checked_add(u64::from(output.width)))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or(ResidentVarDctError::ArithmeticOverflow {
                field: "output binding bytes",
            })?;
        if required > output.storage.size.get() {
            return Err(ResidentVarDctError::OutputBindingSize {
                plane,
                required,
                available: output.storage.size.get(),
            });
        }
    }
    if !inputs
        .indirect
        .usage()
        .contains(wgpu::BufferUsages::INDIRECT)
    {
        return Err(ResidentVarDctError::MissingUsage {
            role: "indirect",
            required: wgpu::BufferUsages::INDIRECT,
        });
    }
    if !inputs.indirect_base_offset.is_multiple_of(4) {
        return Err(ResidentVarDctError::IndirectAlignment {
            offset: inputs.indirect_base_offset,
        });
    }
    let indirect_bytes =
        TransformKind::ALL.len() as u64 * INDIRECT_STAGES * INDIRECT_ARGUMENT_BYTES;
    let indirect_end = inputs
        .indirect_base_offset
        .checked_add(indirect_bytes)
        .ok_or(ResidentVarDctError::ArithmeticOverflow {
            field: "indirect range",
        })?;
    if indirect_end > inputs.indirect.size() {
        return Err(ResidentVarDctError::IndirectRange {
            offset: inputs.indirect_base_offset,
            end: indirect_end,
            available: inputs.indirect.size(),
        });
    }
    Ok(())
}

fn resident_params(
    inputs: ResidentVarDctInputs<'_>,
    transform: TransformKind,
    strategy: u32,
) -> Result<ResidentVarDctParams, ResidentVarDctError> {
    let extent = transform.pixel_extent();
    let lf_extent = transform.lf_extent();
    let area = extent
        .area()
        .and_then(|area| u32::try_from(area).ok())
        .ok_or(ResidentVarDctError::ArithmeticOverflow {
            field: "transform area",
        })?;
    Ok(ResidentVarDctParams {
        task_base: 0,
        task_count: inputs.config.task_capacity,
        transform_width: extent.width,
        transform_height: extent.height,
        transform_area: area,
        lf_width: lf_extent.width,
        lf_height: lf_extent.height,
        quant_offset: inputs.config.quant_offset,
        correlation_offset: inputs.config.correlation_offset,
        lf_offset: inputs.config.lf_offset,
        output_width_x: inputs.outputs[0].width,
        output_height_x: inputs.outputs[0].height,
        output_stride_x: inputs.outputs[0].effective_stride(),
        output_width_y: inputs.outputs[1].width,
        output_height_y: inputs.outputs[1].height,
        output_stride_y: inputs.outputs[1].effective_stride(),
        output_width_b: inputs.outputs[2].width,
        output_height_b: inputs.outputs[2].height,
        output_stride_b: inputs.outputs[2].effective_stride(),
        transform_kind: strategy,
        correlation_width: inputs.config.correlation_width,
        correlation_height: inputs.config.correlation_height,
        task_word_offset: inputs.config.task_word_offset,
        bucket_word_offset: inputs.config.bucket_word_offset,
        lf_stride: inputs.config.lf_stride,
        _padding0: 0,
        _padding1: 0,
        _padding2: 0,
        quant_biases: inputs.config.quant_biases,
    })
}

fn validate_storage_binding(
    device: &wgpu::Device,
    role: &'static str,
    binding: ResidentStorageBinding<'_>,
    _writable: bool,
) -> Result<(), ResidentVarDctError> {
    if !binding.buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(ResidentVarDctError::MissingUsage {
            role,
            required: wgpu::BufferUsages::STORAGE,
        });
    }
    let alignment = u64::from(device.limits().min_storage_buffer_offset_alignment).max(4);
    if !binding.offset.is_multiple_of(alignment) {
        return Err(ResidentVarDctError::BindingAlignment {
            role,
            offset: binding.offset,
            alignment,
        });
    }
    let end = binding.offset.checked_add(binding.size.get()).ok_or(
        ResidentVarDctError::ArithmeticOverflow {
            field: "storage binding range",
        },
    )?;
    if end > binding.buffer.size() {
        return Err(ResidentVarDctError::BindingRange {
            role,
            offset: binding.offset,
            end,
            available: binding.buffer.size(),
        });
    }
    let maximum = device.limits().max_storage_buffer_binding_size;
    if binding.size.get() > maximum {
        return Err(ResidentVarDctError::StorageBindingLimit {
            role,
            required: binding.size.get(),
            available: maximum,
        });
    }
    Ok(())
}

fn storage_entry(binding: u32, storage: ResidentStorageBinding<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: storage.resource(),
    }
}

fn entire_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

const _: () = {
    assert!(std::mem::size_of::<ResidentVarDctParams>() == 128);
    assert!(std::mem::align_of::<ResidentVarDctParams>() == 16);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_and_task_contract_match_the_shared_shader() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<ResidentVarDctParams>();
        assert_eq!(std::mem::size_of::<ResidentVarDctParams>(), 128);
        assert_eq!(GENERAL_TASK_BYTES, 64);
        for shader in [
            include_str!("../shaders/vardct_general.wgsl"),
            include_str!("../shaders/vardct_special.wgsl"),
        ] {
            let module = naga::front::wgsl::parse_str(shader).expect("resident shader parses");
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .expect("resident shader validates");
        }
    }

    #[test]
    fn memory_plan_counts_both_scratch_allocations_and_uniform() {
        let plan = ResidentVarDctMemoryPlan::new(3 * 32 * 32).unwrap();
        assert_eq!(plan.scratch_buffer_bytes, 12_288);
        assert_eq!(plan.uniform_bytes, 3_456);
        assert_eq!(plan.total_bytes, 28_032);
    }

    #[test]
    fn zero_tasks_have_a_typed_error() {
        assert_eq!(
            ResidentVarDctError::ZeroTaskCapacity.to_string(),
            "resident VarDCT task capacity must be nonzero"
        );
    }

    #[test]
    fn dispatch_width_matches_shared_workgroup_contract() {
        for transform in TransformKind::ALL {
            if transform.is_special() {
                continue;
            }
            let area = u32::try_from(transform.pixel_extent().area().unwrap()).unwrap();
            assert!(area.div_ceil(WORKGROUP_SIZE) != 0);
            assert!(area.checked_mul(3).unwrap().div_ceil(WORKGROUP_SIZE) != 0);
        }
    }
}
