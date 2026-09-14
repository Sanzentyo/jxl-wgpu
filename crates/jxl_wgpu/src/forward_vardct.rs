//! GPU forward transforms and normative LF extraction for all VarDCT strategies.

mod basis;
#[cfg(test)]
mod tests;

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::TransformKind;
use wgpu::util::DeviceExt;

use crate::resident_vardct::validate_storage_binding;
use crate::{KernelVariant, ResidentF32Plane, ResidentStorageBinding, ResidentVarDctError};

const KERNEL: &str = "vardct_forward";
const SHADER: &str = include_str!("../shaders/forward_vardct.wgsl");

/// Exact temporary allocation; the caller supplies the input and output buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardVarDctMemoryPlan {
    pub parameter_bytes: u64,
    pub basis_bytes: u64,
    pub horizontal_bytes: u64,
    pub transient_bytes: u64,
    pub coefficient_bytes: u64,
    pub lf_bytes: u64,
}

impl ForwardVarDctMemoryPlan {
    #[must_use]
    pub const fn new(transform: TransformKind) -> Self {
        let extent = transform.pixel_extent();
        let lf = transform.lf_extent();
        let coefficient_bytes = extent.width as u64 * extent.height as u64 * 3 * 4;
        let lf_bytes = lf.width as u64 * lf.height as u64 * 3 * 4;
        let parameter_bytes = std::mem::size_of::<Params>() as u64;
        let (basis_bytes, horizontal_bytes) = if transform.is_special() {
            (64 * 64 * 4, 0)
        } else {
            (
                4 * (extent.width as u64 * extent.width as u64
                    + extent.height as u64 * extent.height as u64
                    + lf.width as u64 * lf.width as u64
                    + lf.height as u64 * lf.height as u64),
                coefficient_bytes,
            )
        };
        Self {
            parameter_bytes,
            basis_bytes,
            horizontal_bytes,
            transient_bytes: parameter_bytes + basis_bytes + horizontal_bytes,
            coefficient_bytes,
            lf_bytes,
        }
    }
}

/// A single transform of three planar F32 channels. Origins are scalar offsets
/// relative to each binding, allowing crops without unaligned storage bindings.
/// Coefficients and LF are contiguous X/Y/B planes in their respective bindings.
#[derive(Clone, Copy, Debug)]
pub struct ForwardVarDctInputs<'a> {
    pub transform: TransformKind,
    pub sources: [ResidentF32Plane<'a>; 3],
    pub origins: [u32; 3],
    pub coefficients: ResidentStorageBinding<'a>,
    pub low_frequency: ResidentStorageBinding<'a>,
}

/// Temporary GPU handles retained through submission by the caller's memory lease.
#[derive(Debug)]
pub struct ForwardVarDctScratch {
    pub parameters: wgpu::Buffer,
    pub basis: wgpu::Buffer,
    pub horizontal: Option<wgpu::Buffer>,
    pub memory: ForwardVarDctMemoryPlan,
}

#[derive(Debug, thiserror::Error)]
pub enum ForwardVarDctError {
    #[error(transparent)]
    Storage(#[from] ResidentVarDctError),
    #[error(transparent)]
    Kernel(#[from] crate::Error),
    #[error("forward VarDCT source {channel} has invalid extent or stride")]
    SourceGeometry { channel: usize },
    #[error("forward VarDCT {role} requires {required} bytes, binding has {available}")]
    BindingSize {
        role: &'static str,
        required: u64,
        available: u64,
    },
    #[error("forward VarDCT {role} binding must contain a whole number of F32 scalars")]
    ScalarAlignment { role: &'static str },
    #[error("forward VarDCT source {channel} address exceeds WGSL u32")]
    SourceAddress { channel: usize },
    #[error("forward VarDCT writable binding {write} overlaps {other}")]
    Alias {
        write: &'static str,
        other: &'static str,
    },
    #[error("forward VarDCT {name} requires {required}, device permits {available}")]
    DeviceLimit {
        name: &'static str,
        required: u64,
        available: u64,
    },
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    area: u32,
    lf_width: u32,
    lf_height: u32,
    lf_area: u32,
    reserved: u32,
    workgroups_x: u32,
    strides: [u32; 4],
    offsets: [u32; 4],
    basis_offsets: [u32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<Params>() == 80);
    assert!(std::mem::align_of::<Params>() == 16);
};

pub struct ForwardVarDctPipeline {
    horizontal: wgpu::ComputePipeline,
    vertical: wgpu::ComputePipeline,
    special: wgpu::ComputePipeline,
    lf: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl ForwardVarDctPipeline {
    pub fn new(device: &wgpu::Device, variant: KernelVariant) -> Result<Self, ForwardVarDctError> {
        if !variant.is_linear() {
            return Err(crate::Error::Unsupported(
                "forward VarDCT requires a linear workgroup".into(),
            )
            .into());
        }
        let limits = device.limits();
        variant.validate_for(KERNEL, &limits, 0)?;
        for (name, required, available) in [
            (
                "max_storage_buffers_per_shader_stage",
                5,
                u64::from(limits.max_storage_buffers_per_shader_stage),
            ),
            (
                "max_uniform_buffers_per_shader_stage",
                1,
                u64::from(limits.max_uniform_buffers_per_shader_stage),
            ),
            (
                "max_uniform_buffer_binding_size",
                std::mem::size_of::<Params>() as u64,
                limits.max_uniform_buffer_binding_size,
            ),
            (
                "max_buffer_size",
                std::mem::size_of::<Params>() as u64,
                limits.max_buffer_size,
            ),
        ] {
            if required <= available {
                continue;
            }
            return Err(ForwardVarDctError::DeviceLimit {
                name,
                required,
                available,
            });
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu forward VarDCT"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let constants = [("wg_x", f64::from(variant.workgroup_size().0))];
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        };
        Ok(Self {
            horizontal: pipeline("horizontal_dct"),
            vertical: pipeline("vertical_dct"),
            special: pipeline("special_transform"),
            lf: pipeline("extract_lf"),
            variant,
        })
    }

    /// Checks every binding and allocation before recording work. Source-dependent
    /// coefficients and LF stay resident; only strategy constants are uploaded.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        commands: &mut wgpu::CommandEncoder,
        inputs: ForwardVarDctInputs<'_>,
    ) -> Result<ForwardVarDctScratch, ForwardVarDctError> {
        let memory = ForwardVarDctMemoryPlan::new(inputs.transform);
        let extent = inputs.transform.pixel_extent();
        let lf = inputs.transform.lf_extent();
        let area = extent.width * extent.height;
        let groups = area.div_ceil(self.variant.workgroup_size().0);
        let limit = device.limits().max_compute_workgroups_per_dimension;
        let workgroups_x = groups.min(limit);
        if workgroups_x == 0 || groups.div_ceil(workgroups_x) > limit {
            return Err(ForwardVarDctError::DeviceLimit {
                name: "max_compute_workgroups_per_dimension",
                required: u64::from(groups),
                available: u64::from(limit),
            });
        }
        validate(device, inputs, memory)?;
        let basis_data = basis::matrix(inputs.transform);
        let params = Params {
            width: extent.width,
            height: extent.height,
            area,
            lf_width: lf.width,
            lf_height: lf.height,
            lf_area: lf.width * lf.height,
            reserved: 0,
            workgroups_x,
            strides: [
                inputs.sources[0].effective_stride(),
                inputs.sources[1].effective_stride(),
                inputs.sources[2].effective_stride(),
                0,
            ],
            offsets: [inputs.origins[0], inputs.origins[1], inputs.origins[2], 0],
            basis_offsets: basis_data.offsets,
        };
        let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu forward VarDCT parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let basis = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu forward VarDCT basis"),
            contents: bytemuck::cast_slice(&basis_data.weights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let horizontal = (memory.horizontal_bytes != 0).then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jxl-wgpu horizontal forward VarDCT scratch"),
                size: memory.horizontal_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let mut dispatch = |pipeline: &wgpu::ComputePipeline,
                            entries: Vec<wgpu::BindGroupEntry<'_>>,
                            count: u32| {
            let entries = entries
                .into_iter()
                .chain([
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: parameters.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: basis.as_entire_binding(),
                    },
                ])
                .collect::<Vec<_>>();
            let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu forward VarDCT bindings"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            // Each pass ends before consumers access its outputs, including LF extraction.
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu forward VarDCT stage"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &binding, &[]);
            let groups = count.div_ceil(self.variant.workgroup_size().0);
            pass.dispatch_workgroups(groups.min(workgroups_x), groups.div_ceil(workgroups_x), 1);
        };
        let source_entries = || {
            inputs
                .sources
                .iter()
                .enumerate()
                .map(|(index, plane)| wgpu::BindGroupEntry {
                    binding: index as u32,
                    resource: plane.storage.resource(),
                })
                .collect::<Vec<_>>()
        };
        if inputs.transform.is_special() {
            let mut entries = source_entries();
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: inputs.coefficients.resource(),
            });
            dispatch(&self.special, entries, area);
        } else if let Some(horizontal) = &horizontal {
            let mut entries = source_entries();
            entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: horizontal.as_entire_binding(),
            });
            dispatch(&self.horizontal, entries, area);
            dispatch(
                &self.vertical,
                vec![
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: horizontal.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: inputs.coefficients.resource(),
                    },
                ],
                area,
            );
        }
        dispatch(
            &self.lf,
            vec![
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: inputs.coefficients.resource(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: inputs.low_frequency.resource(),
                },
            ],
            params.lf_area,
        );
        Ok(ForwardVarDctScratch {
            parameters,
            basis,
            horizontal,
            memory,
        })
    }
}

fn validate(
    device: &wgpu::Device,
    inputs: ForwardVarDctInputs<'_>,
    memory: ForwardVarDctMemoryPlan,
) -> Result<(), ForwardVarDctError> {
    let extent = inputs.transform.pixel_extent();
    let bindings = [
        ("source X", inputs.sources[0].storage),
        ("source Y", inputs.sources[1].storage),
        ("source B", inputs.sources[2].storage),
        ("coefficients", inputs.coefficients),
        ("LF", inputs.low_frequency),
    ];
    for &(role, binding) in &bindings {
        validate_storage_binding(device, role, binding)?;
        if !binding.size.get().is_multiple_of(4) {
            return Err(ForwardVarDctError::ScalarAlignment { role });
        }
    }
    for (channel, source) in inputs.sources.iter().enumerate() {
        if source.width != extent.width
            || source.height != extent.height
            || source.effective_stride() < extent.width
        {
            return Err(ForwardVarDctError::SourceGeometry { channel });
        }
        let end = source
            .effective_stride()
            .checked_mul(extent.height - 1)
            .and_then(|rows| rows.checked_add(inputs.origins[channel]))
            .and_then(|offset| offset.checked_add(extent.width))
            .ok_or(ForwardVarDctError::SourceAddress { channel })?;
        let required = u64::from(end) * 4;
        if required > source.storage.size.get() {
            return Err(ForwardVarDctError::BindingSize {
                role: bindings[channel].0,
                required,
                available: source.storage.size.get(),
            });
        }
    }
    for (role, binding, required) in [
        (
            "coefficients",
            inputs.coefficients,
            memory.coefficient_bytes,
        ),
        ("LF", inputs.low_frequency, memory.lf_bytes),
    ] {
        if required > binding.size.get() {
            return Err(ForwardVarDctError::BindingSize {
                role,
                required,
                available: binding.size.get(),
            });
        }
    }
    for &(write, left) in &bindings[3..] {
        for &(other, right) in &bindings {
            if write != other
                && left.buffer == right.buffer
                && left.offset < right.offset + right.size.get()
                && right.offset < left.offset + left.size.get()
            {
                return Err(ForwardVarDctError::Alias { write, other });
            }
        }
    }
    let limits = device.limits();
    for required in [memory.basis_bytes, memory.horizontal_bytes] {
        for (name, available) in [
            ("max_buffer_size", limits.max_buffer_size),
            (
                "max_storage_buffer_binding_size",
                limits.max_storage_buffer_binding_size,
            ),
        ] {
            if required > available {
                return Err(ForwardVarDctError::DeviceLimit {
                    name,
                    required,
                    available,
                });
            }
        }
    }
    Ok(())
}
