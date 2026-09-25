//! Checked full-resolution Modular side-plane planning, GPU encoding and validated fragments.
//!
//! The color pipeline never sees these samples. Rows are independent GPU tasks, but prediction
//! reads the original preceding row within the same group. Concatenating the compressed rows
//! therefore produces one standard Gradient stream without restarting the predictor.

use jxl_gpu_bitstream::BitWriter;
use wgpu::util::DeviceExt;

use super::entropy::{
    VarDctPrefixCode, prefix_entries, read_fragment_slice, validate_fragment_padding,
};
use super::types::{GpuPrefixEntry, VarDctFrameLayout};
use crate::{BackendError, EncodeError, ProgressivePlan, UnsupportedFeature};

const READY: u32 = 0x4d504c4e;
const GROUP_DIM: u32 = 256;

/// Additional ownership for one full-resolution alpha plane. Readback is also included in
/// `VarDctMemoryPlan::readback_bytes`; `total_bytes` counts it exactly once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctAlphaMemoryPlan {
    pub parameter_bytes: u64,
    pub artifact_bytes: u64,
    pub readback_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    params: Params,
    rows: u32,
    global: bool,
    pass: u32,
    pub(super) memory: VarDctAlphaMemoryPlan,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    source: crate::source::SourceParams,
    width: u32,
    height: u32,
    groups_x: u32,
    row_words: u32,
    big_endian: u32,
    sample_mask: u32,
    prefix: [GpuPrefixEntry; 33],
}

const _: () = assert!(std::mem::size_of::<Params>() == 312);

impl Plan {
    pub(super) fn new(
        frame: VarDctFrameLayout,
        source: crate::source::SourceParams,
        sample_mask: u32,
        big_endian: bool,
        progressive: &ProgressivePlan,
        code: &VarDctPrefixCode,
    ) -> Result<Self, EncodeError> {
        let overflow = || EncodeError::InvalidConfiguration("Modular side-plane size overflow");
        let prefix = prefix_entries(code);
        let max_bits = prefix
            .iter()
            .enumerate()
            .map(|(symbol, entry)| entry.bit_len + symbol.saturating_sub(1) as u32)
            .max()
            .expect("complete raw alphabet");
        let row_words = 4 + frame
            .width
            .min(GROUP_DIM)
            .checked_mul(max_bits)
            .ok_or_else(overflow)?
            .div_ceil(32);
        let rows = frame
            .height
            .checked_mul(frame.ac_groups_x)
            .ok_or_else(overflow)?;
        // Every GPU word address, including the final row's padding, fits u32.
        let words = rows.checked_mul(row_words).ok_or_else(overflow)?;
        let parameter_bytes = std::mem::size_of::<Params>() as u64;
        let artifact_bytes = u64::from(words) * 4;
        let pass = progressive
            .downsampling()
            .iter()
            .find(|point| point.factor == 1)
            .map_or(progressive.passes().len() as u32 - 1, |point| {
                u32::from(point.last_pass)
            });
        Ok(Self {
            params: Params {
                source,
                width: frame.width,
                height: frame.height,
                groups_x: frame.ac_groups_x,
                row_words,
                big_endian: u32::from(big_endian),
                sample_mask,
                prefix,
            },
            rows,
            global: frame.width <= GROUP_DIM && frame.height <= GROUP_DIM,
            pass,
            memory: VarDctAlphaMemoryPlan {
                parameter_bytes,
                artifact_bytes,
                readback_bytes: artifact_bytes,
                total_bytes: parameter_bytes + 2 * artifact_bytes,
            },
        })
    }

    pub(super) fn validate_limits(
        &self,
        buffer_limit: u64,
        binding_limit: u64,
        workgroups: u32,
    ) -> Result<(), EncodeError> {
        for required in [self.memory.parameter_bytes, self.memory.artifact_bytes] {
            for (name, available) in [
                ("max_buffer_size", buffer_limit),
                ("max_storage_buffer_binding_size", binding_limit),
            ] {
                if required > available {
                    return Err(UnsupportedFeature::DeviceLimit {
                        name,
                        required,
                        available,
                    }
                    .into());
                }
            }
        }
        let required = self.params.groups_x.max(self.params.height);
        if required > workgroups {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_compute_workgroups_per_dimension",
                required: u64::from(required),
                available: u64::from(workgroups),
            }
            .into());
        }
        Ok(())
    }

    /// Validates identity, bounds, exact token termination and zero padding. It neither
    /// reconstructs source samples nor creates entropy from host image data.
    pub(super) fn validate<'a>(
        &self,
        bytes: &'a [u8],
        code: &VarDctPrefixCode,
    ) -> Result<Fragments<'a>, BackendError> {
        let invalid = || BackendError::InvalidArtifact("Modular side-plane artifact is malformed");
        if bytes.len() as u64 != self.memory.artifact_bytes {
            return Err(invalid());
        }
        let words: &[u32] = bytemuck::try_cast_slice(bytes).map_err(|_| invalid())?;
        let max_prefix = code
            .raw_entries()
            .iter()
            .map(|entry| entry.bit_len)
            .max()
            .ok_or_else(invalid)?;
        if max_prefix > 15 || max_prefix == 0 {
            return Err(invalid());
        }
        let mut lookup = vec![None; 1usize << max_prefix];
        for (symbol, entry) in code.raw_entries().iter().enumerate() {
            for suffix in 0..1usize << (max_prefix - entry.bit_len) {
                lookup[usize::from(entry.bits) | suffix << entry.bit_len] =
                    Some((symbol as u32, u32::from(entry.bit_len)));
            }
        }
        for row in 0..self.rows {
            let start = (row * self.params.row_words) as usize;
            let fragment = &words[start + 4..start + self.params.row_words as usize];
            let x = row / self.params.height * GROUP_DIM;
            let samples = (self.params.width - x).min(GROUP_DIM);
            let header = &words[start..start + 4];
            if header[..3] != [READY, row, samples] {
                return Err(invalid());
            }
            let bit_len = header[3];
            if u64::from(bit_len) > fragment.len() as u64 * 32 {
                return Err(invalid());
            }
            let mut cursor = 0;
            for _ in 0..samples {
                let count = u32::from(max_prefix).min(bit_len.saturating_sub(cursor));
                let bits = read_fragment_slice(fragment, bit_len, cursor, count)?;
                let (symbol, prefix_bits) = lookup[bits as usize].ok_or_else(invalid)?;
                cursor += prefix_bits + symbol.saturating_sub(1);
                if cursor > bit_len {
                    return Err(invalid());
                }
            }
            if cursor != bit_len {
                return Err(invalid());
            }
            validate_fragment_padding(fragment, bit_len)?;
        }
        Ok(Fragments {
            words,
            plan: Some(*self),
        })
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct Fragments<'a> {
    words: &'a [u32],
    plan: Option<Plan>,
}

impl Fragments<'_> {
    pub(super) fn is_present(self) -> bool {
        self.plan.is_some()
    }

    pub(super) fn write_global(self, output: &mut BitWriter) -> Result<(), EncodeError> {
        if let Some(plan) = self.plan {
            super::bitstream::write_local_modular_header(output)?;
            if plan.global {
                self.append_group(output, 0)?;
            }
        }
        Ok(())
    }

    pub(super) fn write_group(
        self,
        output: &mut BitWriter,
        group: u32,
        pass: u32,
    ) -> Result<(), EncodeError> {
        if self
            .plan
            .is_some_and(|plan| !plan.global && pass == plan.pass)
        {
            super::bitstream::write_local_modular_header(output)?;
            self.append_group(output, group)?;
        }
        Ok(())
    }

    fn append_group(self, output: &mut BitWriter, group: u32) -> Result<(), EncodeError> {
        let plan = self
            .plan
            .ok_or(BackendError::InvalidArtifact("missing Modular side plane"))?;
        let y = group / plan.params.groups_x * GROUP_DIM;
        let x = group % plan.params.groups_x;
        if y >= plan.params.height {
            return Err(BackendError::Invariant("side-plane group out of range").into());
        }
        for row in y..(y + GROUP_DIM).min(plan.params.height) {
            let start = ((x * plan.params.height + row) * plan.params.row_words) as usize;
            super::bitstream::append_gpu_fragment(
                output,
                &self.words[start + 4..start + plan.params.row_words as usize],
                0,
                self.words[start + 3],
            )?;
        }
        Ok(())
    }
}

pub(super) struct Pipeline(wgpu::ComputePipeline);

pub(super) struct Scratch {
    _parameters: wgpu::Buffer,
    _artifact: wgpu::Buffer,
}

impl Pipeline {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VarDCT Modular side-plane encoder"),
            source: wgpu::ShaderSource::Wgsl(
                crate::source::shader(include_str!("modular_plane.wgsl")).into(),
            ),
        });
        Self(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("VarDCT Modular side-plane encoder"),
                layout: None,
                module: &module,
                entry_point: Some("encode"),
                compilation_options: Default::default(),
                cache: None,
            }),
        )
    }

    pub(super) fn encode(
        &self,
        device: &wgpu::Device,
        commands: &mut wgpu::CommandEncoder,
        plan: Plan,
        sources: [wgpu::BindGroupEntry<'_>; 4],
        readback: &wgpu::Buffer,
        offset: u64,
    ) -> Scratch {
        let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Modular side-plane parameters"),
            contents: bytemuck::bytes_of(&plan.params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let artifact = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modular side-plane fragments"),
            size: plan.memory.artifact_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        commands.clear_buffer(&artifact, 0, None);
        let entries: Vec<_> = sources
            .into_iter()
            .chain([
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parameters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: artifact.as_entire_binding(),
                },
            ])
            .collect();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Modular side-plane bindings"),
            layout: &self.0.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Modular side-plane prediction and entropy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.0);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(plan.params.groups_x, plan.params.height, 1);
        }
        commands.copy_buffer_to_buffer(&artifact, 0, readback, offset, plan.memory.artifact_bytes);
        Scratch {
            _parameters: parameters,
            _artifact: artifact,
        }
    }
}
