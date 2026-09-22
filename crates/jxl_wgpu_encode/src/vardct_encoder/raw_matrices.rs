//! GPU Modular encoding of bounded caller-selected raw matrix side images.

use jxl_gpu_bitstream::BitWriter;
use wgpu::util::DeviceExt;

use super::VarDctDequantMatrices;
use super::bitstream::append_gpu_fragment;
use super::dispatch::{gradient_residual_i32, signed_token};
use super::entropy::{VarDctPrefixCode, read_fragment_slice, validate_fragment_padding};
use crate::{BackendError, EncodeError, UnsupportedFeature};

const READY: u32 = 0x524d4154;
const TASK_START: usize = 1 + 2 * 33;

pub(super) struct Plan {
    input: Vec<u32>,
    slots: Vec<Slot>,
    artifact_words: u32,
}

struct Slot {
    family: usize,
    width: usize,
    area: usize,
    input: usize,
    output: usize,
    capacity: usize,
}

impl Plan {
    pub(super) fn new(
        matrices: &VarDctDequantMatrices,
        code: &VarDctPrefixCode,
    ) -> Result<Option<Self>, EncodeError> {
        let families: Vec<_> = (0..17)
            .filter_map(|family| matrices.raw_family(family).map(|raw| (family, raw)))
            .collect();
        if families.is_empty() {
            return Ok(None);
        }
        let mut input = vec![0; TASK_START + 5 * families.len()];
        input[0] = families.len() as u32;
        for (symbol, entry) in code.raw_entries().iter().enumerate() {
            input[1 + 2 * symbol] = u32::from(entry.bits);
            input[2 + 2 * symbol] = u32::from(entry.bit_len);
        }
        let max_bits = code
            .raw_entries()
            .iter()
            .enumerate()
            .map(|(symbol, entry)| u32::from(entry.bit_len) + symbol.saturating_sub(1) as u32)
            .max()
            .expect("nonempty prefix alphabet");
        let mut artifact_words = 4 * families.len() as u32;
        let mut slots = Vec::with_capacity(families.len());
        let overflow = || EncodeError::InvalidConfiguration("raw matrix buffer size overflow");
        for (index, (family, raw)) in families.into_iter().enumerate() {
            let area = raw.channels[0].len();
            let area_words = u32::try_from(area).map_err(|_| overflow())?;
            let samples = area_words.checked_mul(3).ok_or_else(overflow)?;
            let capacity = samples
                .checked_mul(max_bits)
                .ok_or_else(overflow)?
                .div_ceil(32);
            let sample_start = input.len();
            let sample_word = u32::try_from(sample_start).map_err(|_| overflow())?;
            sample_word.checked_add(samples).ok_or_else(overflow)?;
            input[TASK_START + 5 * index..TASK_START + 5 * index + 5].copy_from_slice(&[
                raw.width,
                area_words,
                sample_word,
                artifact_words,
                capacity,
            ]);
            input.extend(raw.channels.iter().flatten().map(|&sample| sample as u32));
            slots.push(Slot {
                family,
                width: raw.width as usize,
                area,
                input: sample_start,
                output: artifact_words as usize,
                capacity: capacity as usize,
            });
            artifact_words = artifact_words.checked_add(capacity).ok_or_else(overflow)?;
        }
        Ok(Some(Self {
            input,
            slots,
            artifact_words,
        }))
    }

    pub(super) fn input_bytes(&self) -> u64 {
        self.input.len() as u64 * 4
    }
    pub(super) fn artifact_bytes(&self) -> u64 {
        u64::from(self.artifact_words) * 4
    }

    pub(super) fn validate_limits(&self, limits: &wgpu::Limits) -> Result<(), EncodeError> {
        for required in [self.input_bytes(), self.artifact_bytes()] {
            for (name, available) in [
                ("max_buffer_size", limits.max_buffer_size),
                (
                    "max_storage_buffer_binding_size",
                    limits.max_storage_buffer_binding_size,
                ),
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
        if self.slots.len() as u64 > u64::from(limits.max_compute_workgroups_per_dimension) {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_compute_workgroups_per_dimension",
                required: self.slots.len() as u64,
                available: u64::from(limits.max_compute_workgroups_per_dimension),
            }
            .into());
        }
        Ok(())
    }

    /// Validates GPU output against the caller's metadata without emitting entropy
    /// or reconstructing image coefficients. Only checked GPU fragments are handed off.
    pub(super) fn validate<'a>(&'a self, bytes: &'a [u8]) -> Result<Fragments<'a>, BackendError> {
        let invalid = || BackendError::InvalidArtifact("raw matrix artifact is malformed");
        if bytes.len() as u64 != self.artifact_bytes() {
            return Err(invalid());
        }
        let words: &[u32] = bytemuck::try_cast_slice(bytes).map_err(|_| invalid())?;
        for (index, slot) in self.slots.iter().enumerate() {
            let header = &words[index * 4..index * 4 + 4];
            if header[..3] != [READY, index as u32, 3 * slot.area as u32] {
                return Err(invalid());
            }
            let bit_len = header[3];
            let fragment = &words[slot.output..slot.output + slot.capacity];
            let mut cursor = 0;
            for channel in 0..3 {
                let base = slot.input + channel * slot.area;
                let samples = &self.input[base..base + slot.area];
                for (i, &sample) in samples.iter().enumerate() {
                    let x = i % slot.width;
                    let y = i / slot.width;
                    let left = if x != 0 {
                        samples[i - 1]
                    } else if y != 0 {
                        samples[i - slot.width]
                    } else {
                        0
                    };
                    let top = if y != 0 {
                        samples[i - slot.width]
                    } else {
                        left
                    };
                    let top_left = if x != 0 && y != 0 {
                        samples[i - slot.width - 1]
                    } else {
                        left
                    };
                    let (token, count, extra) = signed_token(gradient_residual_i32(
                        sample as i32,
                        top as i32,
                        left as i32,
                        top_left as i32,
                    ));
                    let prefix = self.input[1 + 2 * token as usize];
                    let prefix_bits = self.input[2 + 2 * token as usize];
                    if read_fragment_slice(fragment, bit_len, cursor, prefix_bits)? != prefix {
                        return Err(invalid());
                    }
                    cursor += prefix_bits;
                    if read_fragment_slice(fragment, bit_len, cursor, count)? != extra {
                        return Err(invalid());
                    }
                    cursor += count;
                }
            }
            if cursor != bit_len {
                return Err(invalid());
            }
            validate_fragment_padding(fragment, bit_len)?;
        }
        Ok(Fragments {
            words,
            plan: Some(self),
        })
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct Fragments<'a> {
    words: &'a [u32],
    plan: Option<&'a Plan>,
}

impl Fragments<'_> {
    pub(super) fn append(self, output: &mut BitWriter, family: usize) -> Result<(), EncodeError> {
        let (index, slot) = self
            .plan
            .and_then(|plan| {
                plan.slots
                    .iter()
                    .enumerate()
                    .find(|(_, slot)| slot.family == family)
            })
            .ok_or(BackendError::InvalidArtifact(
                "missing GPU raw matrix fragment",
            ))?;
        append_gpu_fragment(
            output,
            &self.words[slot.output..slot.output + slot.capacity],
            0,
            self.words[index * 4 + 3],
        )
    }
}

pub(super) struct Pipeline(wgpu::ComputePipeline);

/// Kept with the enclosing job's permit until the single completion map resolves.
pub(super) struct Scratch {
    _input: wgpu::Buffer,
    _artifact: wgpu::Buffer,
}

impl Pipeline {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raw VarDCT matrix Modular encoder"),
            source: wgpu::ShaderSource::Wgsl(include_str!("raw_matrices.wgsl").into()),
        });
        Self(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("raw VarDCT matrix Modular encoder"),
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
        plan: &Plan,
        readback: &wgpu::Buffer,
        readback_offset: u64,
    ) -> Scratch {
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("raw matrix samples and prefix metadata"),
            contents: bytemuck::cast_slice(&plan.input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let artifact = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw matrix GPU entropy fragments"),
            size: plan.artifact_bytes(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        commands.clear_buffer(&artifact, 0, None);
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raw matrix encoder bindings"),
            layout: &self.0.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: artifact.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raw matrix prediction and entropy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.0);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(plan.slots.len() as u32, 1, 1);
        }
        commands.copy_buffer_to_buffer(
            &artifact,
            0,
            readback,
            readback_offset,
            plan.artifact_bytes(),
        );
        Scratch {
            _input: input,
            _artifact: artifact,
        }
    }
}
