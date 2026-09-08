//! Deterministic JPEG XL noise generation and synthesis on resident color planes.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::Extent2d;
use wgpu::util::DeviceExt;

use crate::{ResidentF32Plane, ResidentStorageBinding};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidentNoiseParameters {
    pub lut: [f32; 8],
    /// Base Y-to-X and Y-to-B correlation, without the LF-specific factor adjustment.
    pub correlation: [f32; 2],
    /// Visible and nonvisible frame counters, after advancing to this physical frame.
    pub frame_seed: [u32; 2],
    pub group_dimension: u32,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResidentNoiseError {
    #[error("invalid resident noise {0}")]
    Invalid(&'static str),
    #[error("resident noise {resource} requires {required}, available {available}")]
    Limit {
        resource: &'static str,
        required: u64,
        available: u64,
    },
    #[error("invalid resident noise plane {plane}: {reason}")]
    Plane { plane: usize, reason: &'static str },
}

type Result<T> = std::result::Result<T, ResidentNoiseError>;

/// Exact explicit storage and uniform requirement for two ordered GPU dispatches.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidentNoisePlan {
    extent: Extent2d,
    parameters: ResidentNoiseParameters,
    storage_bytes: u64,
}

impl ResidentNoisePlan {
    pub const UNIFORM_BYTES: u64 = std::mem::size_of::<NoiseParams>() as u64;

    pub fn new(
        extent: Extent2d,
        parameters: ResidentNoiseParameters,
        limits: &wgpu::Limits,
    ) -> Result<Self> {
        if extent.width == 0
            || extent.height == 0
            || extent.width > i32::MAX as u32 / 2
            || extent.height > i32::MAX as u32 / 2
        {
            return Err(ResidentNoiseError::Invalid("extent"));
        }
        if !matches!(parameters.group_dimension, 128 | 256 | 512 | 1024)
            || parameters
                .lut
                .iter()
                .any(|x| !x.is_finite() || !(0.0..1.0).contains(x))
            || parameters.correlation.iter().any(|x| !x.is_finite())
        {
            return Err(ResidentNoiseError::Invalid("model parameters"));
        }
        crate::KernelVariant::Tile16x16
            .validate_for("resident_noise", limits, 0)
            .map_err(|_| ResidentNoiseError::Invalid("workgroup limits"))?;
        let words = u64::from(extent.width) * u64::from(extent.height) * 3;
        require("scratch address words", words, u64::from(u32::MAX))?;
        let storage_bytes = words * 4;
        require(
            "scratch bytes",
            storage_bytes,
            limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size),
        )?;
        require(
            "uniform bytes",
            Self::UNIFORM_BYTES,
            limits.max_uniform_buffer_binding_size,
        )?;
        for dimension in [extent.width, extent.height] {
            require(
                "workgroups",
                u64::from(dimension.div_ceil(16)),
                u64::from(limits.max_compute_workgroups_per_dimension),
            )?;
        }
        Ok(Self {
            extent,
            parameters,
            storage_bytes,
        })
    }

    pub const fn total_bytes(&self) -> u64 {
        self.storage_bytes + Self::UNIFORM_BYTES
    }

    pub const fn storage_bytes(&self) -> u64 {
        self.storage_bytes
    }

    pub fn allocate(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu random noise planes"),
            size: self.storage_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }
}

pub struct ResidentNoiseInputs<'a> {
    pub plan: &'a ResidentNoisePlan,
    pub planes: [ResidentF32Plane<'a>; 3],
    pub scratch: &'a wgpu::Buffer,
}

pub struct ResidentNoisePipeline {
    generate: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
}

impl ResidentNoisePipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        crate::KernelVariant::Tile16x16
            .validate_for("resident_noise", &device.limits(), 0)
            .map_err(|_| ResidentNoiseError::Invalid("workgroup limits"))?;
        let module = device.create_shader_module(wgpu::include_wgsl!("../shaders/noise.wgsl"));
        let create = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            generate: create("generate"),
            apply: create("apply"),
        })
    }

    /// Records random generation and fused convolution/addition; returns the retained uniform.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: ResidentNoiseInputs<'_>,
    ) -> Result<wgpu::Buffer> {
        let plan = inputs.plan;
        // A caller may have built the plan against a different device.
        ResidentNoisePlan::new(plan.extent, plan.parameters, &device.limits())?;
        if !inputs.scratch.usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(ResidentNoiseError::Invalid("scratch usage"));
        }
        require("scratch binding", plan.storage_bytes, inputs.scratch.size())?;
        let scratch = ResidentStorageBinding {
            buffer: inputs.scratch,
            offset: 0,
            size: std::num::NonZeroU64::new(plan.storage_bytes).unwrap(),
        };
        for (index, plane) in inputs.planes.iter().enumerate() {
            validate_plane(device, plan, index, *plane)?;
            if plane.storage.buffer == inputs.scratch {
                return Err(ResidentNoiseError::Invalid("scratch aliases color"));
            }
            for previous in &inputs.planes[..index] {
                if plane.storage.buffer == previous.storage.buffer
                    && plane.storage.offset < previous.storage.offset + previous.storage.size.get()
                    && previous.storage.offset < plane.storage.offset + plane.storage.size.get()
                {
                    return Err(ResidentNoiseError::Invalid("overlapping color planes"));
                }
            }
        }
        let params = NoiseParams {
            geometry: [
                plan.extent.width,
                plan.extent.height,
                plan.parameters.group_dimension,
                plan.extent.width * plan.extent.height,
            ],
            seed: [
                plan.parameters.frame_seed[0],
                plan.parameters.frame_seed[1],
                0,
                0,
            ],
            strides: [
                inputs.planes[0].effective_stride(),
                inputs.planes[1].effective_stride(),
                inputs.planes[2].effective_stride(),
                0,
            ],
            correlation: [
                plan.parameters.correlation[0],
                plan.parameters.correlation[1],
                0.0,
                0.0,
            ],
            lut: plan.parameters.lut,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu noise model and seed"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let resources = [
            uniform.as_entire_binding(),
            scratch.resource(),
            inputs.planes[0].storage.resource(),
            inputs.planes[1].storage.resource(),
            inputs.planes[2].storage.resource(),
        ];
        for (pipeline, count, dispatch) in [
            (
                &self.generate,
                2,
                [
                    plan.extent.width.div_ceil(plan.parameters.group_dimension),
                    plan.extent.height.div_ceil(plan.parameters.group_dimension),
                ],
            ),
            (
                &self.apply,
                5,
                [
                    plan.extent.width.div_ceil(16),
                    plan.extent.height.div_ceil(16),
                ],
            ),
        ] {
            let entries: Vec<_> = resources[..count]
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, resource)| wgpu::BindGroupEntry {
                    binding: index as u32,
                    resource,
                })
                .collect();
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu noise bindings"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(dispatch[0], dispatch[1], 1);
        }
        Ok(uniform)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NoiseParams {
    geometry: [u32; 4],
    seed: [u32; 4],
    strides: [u32; 4],
    correlation: [f32; 4],
    lut: [f32; 8],
}
const _: () = assert!(std::mem::size_of::<NoiseParams>() == 96);

fn require(resource: &'static str, required: u64, available: u64) -> Result<()> {
    if required > available {
        Err(ResidentNoiseError::Limit {
            resource,
            required,
            available,
        })
    } else {
        Ok(())
    }
}

fn validate_plane(
    device: &wgpu::Device,
    plan: &ResidentNoisePlan,
    index: usize,
    plane: ResidentF32Plane<'_>,
) -> Result<()> {
    let invalid = |reason| ResidentNoiseError::Plane {
        plane: index,
        reason,
    };
    if plane.width != plan.extent.width
        || plane.height != plan.extent.height
        || plane.effective_stride() < plane.width
    {
        return Err(invalid("geometry"));
    }
    let binding = plane.storage;
    if !binding.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
        || !binding
            .offset
            .is_multiple_of(u64::from(device.limits().min_storage_buffer_offset_alignment).max(4))
        || !binding.size.get().is_multiple_of(4)
        || binding
            .offset
            .checked_add(binding.size.get())
            .is_none_or(|end| end > binding.buffer.size())
    {
        return Err(invalid("storage binding"));
    }
    let words =
        u64::from(plane.height - 1) * u64::from(plane.effective_stride()) + u64::from(plane.width);
    require("color address words", words, u64::from(u32::MAX))?;
    require("color binding bytes", words * 4, binding.size.get())?;
    require(
        "color storage limit",
        binding.size.get(),
        device.limits().max_storage_buffer_binding_size,
    )
}

#[cfg(test)]
mod tests;
