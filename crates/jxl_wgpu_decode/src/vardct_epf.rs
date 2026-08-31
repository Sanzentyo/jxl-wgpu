//! GPU construction of the per-block VarDCT EPF inverse-sigma field.

use bytemuck::{Pod, Zeroable};
use jxl_wgpu::KernelVariant;
use thiserror::Error;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("vardct_epf.wgsl");

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpfSigmaConfig {
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub task_count: u32,
    pub sharpness_offset_words: u32,
    pub artifact_status_offset_words: u32,
    pub task_metadata_offset_words: u32,
    pub global_scale: u32,
    pub quant_mul: f32,
    pub sharp_lut: [f32; 8],
}

impl EpfSigmaConfig {
    pub fn plan(self) -> Result<EpfSigmaMemoryPlan, EpfSigmaError> {
        let sigma_bytes = u64::from(self.blocks_x)
            .checked_mul(u64::from(self.blocks_y))
            .and_then(|blocks| blocks.checked_mul(4))
            .ok_or(EpfSigmaError::ArithmeticOverflow {
                field: "sigma plane bytes",
            })?;
        if sigma_bytes == 0 {
            return Err(EpfSigmaError::EmptyBlockGrid);
        }
        if self.task_count == 0 {
            return Err(EpfSigmaError::EmptyTaskSet);
        }
        if self.global_scale == 0 {
            return Err(EpfSigmaError::ZeroGlobalScale);
        }
        if !self.quant_mul.is_finite() || self.sharp_lut.into_iter().any(|value| !value.is_finite())
        {
            return Err(EpfSigmaError::NonFiniteParameters);
        }
        Ok(EpfSigmaMemoryPlan {
            sigma_bytes,
            uniform_bytes: std::mem::size_of::<EpfSigmaUniform>() as u64,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpfSigmaMemoryPlan {
    pub sigma_bytes: u64,
    pub uniform_bytes: u64,
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum EpfSigmaError {
    #[error("EPF sigma construction requires a linear workgroup, got {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    #[error("EPF sigma workgroup variant {variant:?} exceeds device limits")]
    WorkgroupVariant { variant: KernelVariant },
    #[error("EPF sigma construction requires a nonempty block grid")]
    EmptyBlockGrid,
    #[error("EPF sigma construction requires at least one transform task")]
    EmptyTaskSet,
    #[error("EPF sigma construction requires a nonzero global quantizer scale")]
    ZeroGlobalScale,
    #[error("EPF sigma parameters contain a non-finite value")]
    NonFiniteParameters,
    #[error("EPF sigma buffer needs {required} bytes, device permits {available}")]
    StorageBindingLimit { required: u64, available: u64 },
    #[error("EPF sigma dispatch count {required} exceeds device limit {available}")]
    WorkgroupCount { required: u32, available: u32 },
    #[error("EPF sigma arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
}

pub struct EpfSigmaPipeline {
    pipeline: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl EpfSigmaPipeline {
    pub fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, EpfSigmaError> {
        if !variant.is_linear() {
            return Err(EpfSigmaError::WorkgroupShape { variant });
        }
        variant
            .validate_for("vardct_epf_sigma", &device.limits(), 0)
            .map_err(|_| EpfSigmaError::WorkgroupVariant { variant })?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu VarDCT EPF sigma"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let (workgroup_x, _) = variant.workgroup_size();
        let constants = [("wg_x", f64::from(workgroup_x))];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu VarDCT EPF sigma"),
            layout: None,
            module: &module,
            entry_point: Some("build_epf_sigma"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self { pipeline, variant })
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        raw_metadata: &wgpu::Buffer,
        artifact: &wgpu::Buffer,
        sigma: &wgpu::Buffer,
        config: EpfSigmaConfig,
    ) -> Result<wgpu::Buffer, EpfSigmaError> {
        let plan = config.plan()?;
        let maximum_binding = device.limits().max_storage_buffer_binding_size;
        if plan.sigma_bytes > maximum_binding {
            return Err(EpfSigmaError::StorageBindingLimit {
                required: plan.sigma_bytes,
                available: maximum_binding,
            });
        }
        let dispatch = config.task_count.div_ceil(self.variant.workgroup_size().0);
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        if dispatch > maximum {
            return Err(EpfSigmaError::WorkgroupCount {
                required: dispatch,
                available: maximum,
            });
        }
        let uniform_data = EpfSigmaUniform {
            geometry: [
                config.blocks_x,
                config.blocks_y,
                config.task_count,
                config.sharpness_offset_words,
            ],
            artifact_status_offset_words: config.artifact_status_offset_words,
            task_metadata_offset_words: config.task_metadata_offset_words,
            global_scale: config.global_scale,
            quant_mul: config.quant_mul,
            sharp_lut: [
                config.sharp_lut[..4].try_into().unwrap(),
                config.sharp_lut[4..].try_into().unwrap(),
            ],
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu VarDCT EPF sigma params"),
            contents: bytemuck::bytes_of(&uniform_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu VarDCT EPF sigma bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                binding(0, raw_metadata),
                binding(1, artifact),
                binding(2, sigma),
                binding(3, &uniform),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu VarDCT EPF sigma"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch, 1, 1);
        drop(pass);
        Ok(uniform)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct EpfSigmaUniform {
    geometry: [u32; 4],
    artifact_status_offset_words: u32,
    task_metadata_offset_words: u32,
    global_scale: u32,
    quant_mul: f32,
    sharp_lut: [[f32; 4]; 2],
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

const _: () = {
    assert!(std::mem::size_of::<EpfSigmaUniform>() == 64);
    assert!(std::mem::align_of::<EpfSigmaUniform>() == 16);
    assert!(std::mem::offset_of!(EpfSigmaUniform, geometry) == 0);
    assert!(std::mem::offset_of!(EpfSigmaUniform, artifact_status_offset_words) == 16);
    assert!(std::mem::offset_of!(EpfSigmaUniform, task_metadata_offset_words) == 20);
    assert!(std::mem::offset_of!(EpfSigmaUniform, global_scale) == 24);
    assert!(std::mem::offset_of!(EpfSigmaUniform, quant_mul) == 28);
    assert!(std::mem::offset_of!(EpfSigmaUniform, sharp_lut) == 32);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_plan_is_exact_and_rejects_non_finite_inputs() {
        let config = EpfSigmaConfig {
            blocks_x: 3,
            blocks_y: 5,
            task_count: 15,
            sharpness_offset_words: 20,
            artifact_status_offset_words: 0,
            task_metadata_offset_words: 16,
            global_scale: 4096,
            quant_mul: 0.46,
            sharp_lut: [
                0.0,
                1.0 / 7.0,
                2.0 / 7.0,
                3.0 / 7.0,
                4.0 / 7.0,
                5.0 / 7.0,
                6.0 / 7.0,
                1.0,
            ],
        };
        assert_eq!(
            config.plan().unwrap(),
            EpfSigmaMemoryPlan {
                sigma_bytes: 60,
                uniform_bytes: 64,
            }
        );
        assert_eq!(
            EpfSigmaConfig {
                quant_mul: f32::NAN,
                ..config
            }
            .plan()
            .unwrap_err(),
            EpfSigmaError::NonFiniteParameters
        );
    }

    #[test]
    fn sigma_shader_has_a_portable_semantic_module() {
        let module = naga::front::wgsl::parse_str(SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}
