//! Normative JPEG XL 5×5 interpolation between resident F32 planes.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{KernelVariant, ResidentF32Plane, ResidentStorageBinding};

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResidentUpsampleError {
    #[error("upsampling factor {factor} must be 2, 4, or 8")]
    Factor { factor: u32 },
    #[error("{factor}x upsampling requires {expected} compact weights, got {actual}")]
    WeightCount {
        factor: u32,
        expected: usize,
        actual: usize,
    },
    #[error("upsampling weight {index} must be finite")]
    NonFiniteWeight { index: usize },
    #[error("resident upsampling {role} plane has invalid geometry or addressing")]
    PlaneGeometry { role: &'static str },
    #[error("resident upsampling {role} plane has an invalid storage binding")]
    Binding { role: &'static str },
    #[error("resident upsampling extents do not match factor {factor}")]
    Extent { factor: u32 },
    #[error("resident upsampling requires a supported tiled workgroup, got {variant:?}")]
    WorkgroupVariant { variant: KernelVariant },
    #[error("resident upsampling {resource} requires {required}, device permits {available}")]
    DeviceLimit {
        resource: &'static str,
        required: u64,
        available: u64,
    },
}

/// Validated, phase-major 5×5 kernels expanded from the image-header triangle.
/// Expansion touches only the bounded scalar weights, never image samples.
#[derive(Clone, Debug)]
pub struct ResidentUpsampleKernel {
    factor: u32,
    weights: Vec<f32>,
}

impl ResidentUpsampleKernel {
    /// Resolves all transposed/reflected phases from 15, 55, or 210 compact weights.
    pub fn from_compact(factor: u32, compact: &[f32]) -> Result<Self, ResidentUpsampleError> {
        if !matches!(factor, 2 | 4 | 8) {
            return Err(ResidentUpsampleError::Factor { factor });
        }
        let n = factor as usize;
        let half = n / 2;
        let side = 5 * half;
        let expected = side * (side + 1) / 2;
        if compact.len() != expected {
            return Err(ResidentUpsampleError::WeightCount {
                factor,
                expected,
                actual: compact.len(),
            });
        }
        if let Some(index) = compact.iter().position(|weight| !weight.is_finite()) {
            return Err(ResidentUpsampleError::NonFiniteWeight { index });
        }
        let mut weights = vec![0.0; n * n * 25];
        for phase_y in 0..n {
            for phase_x in 0..n {
                let phase = (phase_y * n + phase_x) * 25;
                for y in 0..5 {
                    for x in 0..5 {
                        let row = phase_y.min(n - phase_y - 1) * 5
                            + if phase_y < half { y } else { 4 - y };
                        let col = phase_x.min(n - phase_x - 1) * 5
                            + if phase_x < half { x } else { 4 - x };
                        let low = row.min(col);
                        let high = row.max(col);
                        let index = low * (2 * side - low + 1) / 2 + high - low;
                        weights[phase + y * 5 + x] = compact[index];
                    }
                }
            }
        }
        Ok(Self { factor, weights })
    }

    #[must_use]
    pub const fn factor(&self) -> u32 {
        self.factor
    }

    /// Exact storage bytes shared by every channel using this kernel.
    #[must_use]
    pub fn weight_bytes(&self) -> u64 {
        self.weights.len() as u64 * 4
    }

    pub fn upload(
        &self,
        device: &wgpu::Device,
    ) -> Result<ResidentUpsampleWeights, ResidentUpsampleError> {
        let limits = device.limits();
        check_limit(
            "weight buffer bytes",
            self.weight_bytes(),
            limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size),
        )?;
        Ok(ResidentUpsampleWeights {
            factor: self.factor,
            buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu resident upsampling weights"),
                contents: bytemuck::cast_slice(&self.weights),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        })
    }
}

/// One immutable GPU weight buffer, reusable across channels in a frame.
pub struct ResidentUpsampleWeights {
    factor: u32,
    buffer: wgpu::Buffer,
}

/// A scalar view into planar or interleaved F32 storage. All addressing is in scalar words,
/// relative to the storage binding. The filter validates the complete view before recording work.
#[derive(Clone, Copy, Debug)]
pub struct ResidentUpsampleSource<'a> {
    pub storage: ResidentStorageBinding<'a>,
    pub width: u32,
    pub height: u32,
    pub offset: u32,
    pub row_stride: u32,
    pub sample_stride: u32,
}

impl<'a> From<ResidentF32Plane<'a>> for ResidentUpsampleSource<'a> {
    fn from(plane: ResidentF32Plane<'a>) -> Self {
        Self {
            storage: plane.storage,
            width: plane.width,
            height: plane.height,
            offset: 0,
            row_stride: plane.effective_stride(),
            sample_stride: 1,
        }
    }
}

pub struct ResidentUpsampleInputs<'a> {
    pub input: ResidentUpsampleSource<'a>,
    /// A distinct destination, optionally cropped at the right/bottom edge.
    pub output: ResidentF32Plane<'a>,
    pub weights: &'a ResidentUpsampleWeights,
}

/// Records the same filter used by the render-graph scheduler without host pixel uploads.
pub struct ResidentUpsamplePipeline {
    pipeline: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl ResidentUpsamplePipeline {
    pub const UNIFORM_BYTES: u64 = std::mem::size_of::<UpsampleUniform>() as u64;

    pub fn new(device: &wgpu::Device) -> Result<Self, ResidentUpsampleError> {
        Self::with_variant(device, KernelVariant::Tile16x16)
    }

    pub fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, ResidentUpsampleError> {
        if variant.is_linear()
            || variant
                .validate_for("resident_upsample", &device.limits(), 0)
                .is_err()
        {
            return Err(ResidentUpsampleError::WorkgroupVariant { variant });
        }
        check_limit(
            "uniform bytes",
            Self::UNIFORM_BYTES,
            device.limits().max_uniform_buffer_binding_size,
        )?;
        let module = device.create_shader_module(wgpu::include_wgsl!("../shaders/upsample.wgsl"));
        let (x, y) = variant.workgroup_size();
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu resident frame upsampling"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("wg_x", f64::from(x)), ("wg_y", f64::from(y))],
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self { pipeline, variant })
    }

    /// Validates both planes and records one dispatch; retain its uniform until completion.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: ResidentUpsampleInputs<'_>,
    ) -> Result<wgpu::Buffer, ResidentUpsampleError> {
        validate_source(device, "input", inputs.input)?;
        validate_source(device, "output", inputs.output.into())?;
        let factor = inputs.weights.factor;
        if inputs.output.width.div_ceil(factor) != inputs.input.width
            || inputs.output.height.div_ceil(factor) != inputs.input.height
        {
            return Err(ResidentUpsampleError::Extent { factor });
        }
        let (x, y) = self.variant.workgroup_size();
        let dispatch_x = inputs.output.width.div_ceil(x);
        let dispatch_y = inputs.output.height.div_ceil(y);
        let maximum = u64::from(device.limits().max_compute_workgroups_per_dimension);
        check_limit("X workgroups", u64::from(dispatch_x), maximum)?;
        check_limit("Y workgroups", u64::from(dispatch_y), maximum)?;
        let params = UpsampleUniform {
            input_width: inputs.input.width,
            input_height: inputs.input.height,
            output_width: inputs.output.width,
            output_height: inputs.output.height,
            input_stride: inputs.input.row_stride,
            output_stride: inputs.output.effective_stride(),
            factor,
            input_offset: inputs.input.offset,
            input_sample_stride: inputs.input.sample_stride,
            _padding: [0; 3],
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu resident frame upsampling params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let resources = [
            inputs.input.storage.resource(),
            inputs.weights.buffer.as_entire_binding(),
            inputs.output.storage.resource(),
            uniform.as_entire_binding(),
        ];
        let entries: Vec<_> = resources
            .into_iter()
            .enumerate()
            .map(|(index, resource)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource,
            })
            .collect();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu resident frame upsampling bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu resident frame upsampling"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        drop(pass);
        Ok(uniform)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct UpsampleUniform {
    pub(crate) input_width: u32,
    pub(crate) input_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) input_stride: u32,
    pub(crate) output_stride: u32,
    pub(crate) factor: u32,
    pub(crate) input_offset: u32,
    pub(crate) input_sample_stride: u32,
    pub(crate) _padding: [u32; 3],
}

fn check_limit(
    resource: &'static str,
    required: u64,
    available: u64,
) -> Result<(), ResidentUpsampleError> {
    if required > available {
        return Err(ResidentUpsampleError::DeviceLimit {
            resource,
            required,
            available,
        });
    }
    Ok(())
}

fn validate_source(
    device: &wgpu::Device,
    role: &'static str,
    plane: ResidentUpsampleSource<'_>,
) -> Result<(), ResidentUpsampleError> {
    let invalid = || ResidentUpsampleError::PlaneGeometry { role };
    if plane.width == 0
        || plane.height == 0
        || plane.sample_stride == 0
        || plane.width > i32::MAX as u32 / 2
        || plane.height > i32::MAX as u32 / 2
    {
        return Err(invalid());
    }
    let row_words = (plane.width - 1)
        .checked_mul(plane.sample_stride)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(invalid)?;
    if plane.row_stride < row_words {
        return Err(invalid());
    }
    let words = (plane.height - 1)
        .checked_mul(plane.row_stride)
        .and_then(|v| v.checked_add(row_words))
        .and_then(|v| v.checked_add(plane.offset))
        .ok_or_else(invalid)?;
    let required = u64::from(words) * 4;
    let storage = plane.storage;
    let limits = device.limits();
    if !storage.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
        || !storage
            .offset
            .is_multiple_of(u64::from(limits.min_storage_buffer_offset_alignment).max(4))
        || !storage.size.get().is_multiple_of(4)
        || storage
            .offset
            .checked_add(storage.size.get())
            .is_none_or(|end| end > storage.buffer.size())
        || required > storage.size.get()
    {
        return Err(ResidentUpsampleError::Binding { role });
    }
    check_limit(
        "plane binding bytes",
        storage.size.get(),
        limits.max_storage_buffer_binding_size,
    )
}

const _: () = {
    assert!(std::mem::size_of::<UpsampleUniform>() == 48);
    assert!(std::mem::align_of::<UpsampleUniform>() == 16);
    assert!(std::mem::offset_of!(UpsampleUniform, input_offset) == 28);
    assert!(std::mem::offset_of!(UpsampleUniform, input_sample_stride) == 32);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_kernels_validate_and_expand_every_phase() {
        for factor in [2_u32, 4, 8] {
            let side = 5 * factor as usize / 2;
            let compact: Vec<_> = (0..side * (side + 1) / 2).map(|i| i as f32).collect();
            let kernel = ResidentUpsampleKernel::from_compact(factor, &compact).unwrap();
            let n = factor as usize;
            assert_eq!(kernel.weight_bytes(), u64::from(factor * factor) * 25 * 4);
            // Reconstruct a symmetric matrix by walking its triangle rather than using the
            // production index expression, then compare all phase/reflection coordinates.
            let mut matrix = vec![vec![0.0; side]; side];
            let mut values = compact.iter();
            let mut remaining = matrix.as_mut_slice();
            let mut y = 0;
            while let Some((row, below)) = remaining.split_first_mut() {
                row[y] = *values.next().unwrap();
                for (cell, other_row) in row[y + 1..].iter_mut().zip(below.iter_mut()) {
                    let value = *values.next().unwrap();
                    *cell = value;
                    other_row[y] = value;
                }
                y += 1;
                remaining = below;
            }
            for py in 0..n {
                for px in 0..n {
                    for y in 0..5 {
                        for x in 0..5 {
                            let row = if py < n / 2 {
                                py * 5 + y
                            } else {
                                (n - py) * 5 - y - 1
                            };
                            let col = if px < n / 2 {
                                px * 5 + x
                            } else {
                                (n - px) * 5 - x - 1
                            };
                            assert_eq!(
                                kernel.weights[(py * n + px) * 25 + y * 5 + x],
                                matrix[row][col]
                            );
                        }
                    }
                }
            }
            assert!(matches!(
                ResidentUpsampleKernel::from_compact(factor, &compact[1..]),
                Err(ResidentUpsampleError::WeightCount { .. })
            ));
            let mut bad = compact;
            bad[0] = f32::NAN;
            assert!(matches!(
                ResidentUpsampleKernel::from_compact(factor, &bad),
                Err(ResidentUpsampleError::NonFiniteWeight { index: 0 })
            ));
        }
        assert!(matches!(
            ResidentUpsampleKernel::from_compact(3, &[]),
            Err(ResidentUpsampleError::Factor { factor: 3 })
        ));
        let module =
            naga::front::wgsl::parse_str(include_str!("../shaders/upsample.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}
