//! Frame-wide entropy policy and bounded GPU serialization of complete Modular groups.
use std::sync::Arc;

use jxl_gpu_bitstream::BitWriter;

use super::dispatch::{ModularDispatchBatch, ModularDispatchPlan};
use super::lz77::LosslessModularLz77;
use super::serializer::{DistanceCode, ValidatedModularArtifact};
use crate::ans::{ALPHABET, AnsCode, TABLE_WORDS};
use crate::prefix::{LZ77_SYMBOLS, PrefixCode, RAW_SYMBOLS};
use crate::{BackendError, EncodeError, WgpuContext};

mod clustering;
mod hybrid;
pub(super) use hybrid::{FrameHistograms, PROFILE_BYTES, PROFILES};

/// Entropy coding after lossless GPU tokenization. This selection never changes source words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LosslessModularEntropyCoding {
    #[default]
    Prefix,
    /// GPU ANS uses one state per complete group and deterministic clustered frame histograms.
    Ans,
}

pub(super) enum EntropyCode {
    Prefix {
        codes: Box<[PrefixCode; 4]>,
        distance: Option<DistanceCode>,
    },
    Ans(Box<AnsCodebook>),
}

impl EntropyCode {
    pub(super) fn from_histograms(
        plan: &ModularDispatchPlan,
        histograms: &FrameHistograms,
    ) -> Result<Self, EncodeError> {
        match plan.entropy {
            LosslessModularEntropyCoding::Prefix => Ok(Self::Prefix {
                codes: Box::new(super::serializer::build_prefix_codes(
                    plan.format,
                    plan.bits_per_sample,
                    plan.predictor,
                    &plan.transforms,
                    &histograms.raw,
                    &histograms.lz77,
                )?),
                distance: super::serializer::build_distance_code(plan.lz77, &histograms.distance)?,
            }),
            LosslessModularEntropyCoding::Ans => Ok(Self::Ans(Box::new(AnsCodebook::new(
                plan.lz77,
                histograms,
                1 + if plan.group_grid.groups > 1
                    && plan.tree_mode == super::types::LosslessModularTreeMode::LocalPerGroup
                {
                    u64::from(plan.group_grid.groups)
                } else {
                    0
                },
            )?))),
        }
    }

    pub(super) fn matches_lz77(&self, mode: LosslessModularLz77) -> bool {
        match self {
            Self::Prefix { distance, .. } => {
                distance.is_some() == (mode == LosslessModularLz77::Greedy)
            }
            Self::Ans(codebook) => codebook.mode == mode,
        }
    }

    pub(super) fn ans(&self) -> Option<&AnsCodebook> {
        match self {
            Self::Ans(codebook) => Some(codebook),
            _ => None,
        }
    }

    pub(super) fn write_context_map(&self, writer: &mut BitWriter) -> Result<(), EncodeError> {
        let map = self
            .ans()
            .map_or(&[0, 1, 2, 3, 4], |codebook| &codebook.context_map);
        clustering::write_context_map(writer, map)
    }

    pub(super) fn write_empty_stream(&self, writer: &mut BitWriter) -> Result<(), EncodeError> {
        // Match libjxl WriteTokens: an empty ANS stream retains its 32-bit terminal state.
        if self.ans().is_some() {
            writer.write_bits(0x13 << 16, 32)?;
        }
        Ok(())
    }

    pub(super) fn write_stream(
        &self,
        writer: &mut BitWriter,
        artifacts: &[ValidatedModularArtifact<'_>],
        encoded: Option<EncodedGroup<'_>>,
    ) -> Result<(), EncodeError> {
        match (self, encoded) {
            (Self::Prefix { codes, distance }, None) => {
                for (channel, artifact) in artifacts.iter().enumerate() {
                    super::serializer::write_events(
                        writer,
                        &codes[channel.min(3)],
                        distance.as_ref(),
                        artifact.events,
                    )?;
                }
            }
            (Self::Ans(_), Some(encoded)) => {
                // Copy already-compressed bits without host token coding or byte alignment.
                for (index, &byte) in encoded.bytes.iter().enumerate() {
                    writer
                        .write_bits(u64::from(byte), (encoded.bit_len - index * 8).min(8) as u8)?;
                }
            }
            _ => {
                return Err(BackendError::Invariant(
                    "entropy result does not match selected coder",
                )
                .into());
            }
        }
        Ok(())
    }
}

pub(super) struct AnsCodebook {
    tables: Vec<hybrid::HybridCode>,
    /// Distance, then channel 0/1/2/3+. Shared by wire metadata and GPU lowering.
    context_map: clustering::ContextMap,
    mode: LosslessModularLz77,
}

impl AnsCodebook {
    pub(super) fn new(
        mode: LosslessModularLz77,
        histograms: &FrameHistograms,
        header_copies: u64,
    ) -> Result<Self, EncodeError> {
        let profiles = histograms.candidates(mode)?;
        let (tables, context_map) = clustering::cluster(&profiles, header_copies)?;
        Ok(Self {
            tables,
            context_map,
            mode,
        })
    }

    /// The selected table owns both its wire configuration and GPU recoding metadata.
    pub(super) fn write_histograms(&self, writer: &mut BitWriter) -> Result<(), EncodeError> {
        writer.write_bits(0, 1)?; // ANS
        writer.write_bits(3, 2)?; // log alphabet size = 8
        for table in &self.tables {
            table.config.write(writer)?;
        }
        for table in &self.tables {
            table.code.write_histogram(writer)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct EntropyArtifactPlan {
    /// Byte offset inside the first channel's artifact allocation.
    pub(super) byte_offset: u64,
    pub(super) capacity_words: u32,
}

impl EntropyArtifactPlan {
    pub(super) fn for_events(byte_offset: u64, events: u64) -> Result<Self, EncodeError> {
        // A zero-run event expands to three ANS symbols; every other event to one.
        // Each symbol can renormalize once (16 bits), plus at most 31 extra bits per event.
        let bits = events
            .checked_mul(80)
            .and_then(|bits| bits.checked_add(32))
            .ok_or(EncodeError::InvalidSource("ANS bit capacity overflow"))?;
        let capacity_words = u32::try_from(bits.div_ceil(32))
            .ok()
            .filter(|&words| words <= u32::MAX / 32)
            .ok_or(EncodeError::InvalidSource(
                "ANS capacity exceeds WGSL bit indexing",
            ))?;
        Ok(Self {
            byte_offset,
            capacity_words,
        })
    }
    pub(super) fn bytes(self) -> u64 {
        4 * (4 + u64::from(self.capacity_words))
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct EntropyBatchPlan {
    pub(super) parameter_offset: u64,
    pub(super) group_count: u32,
    pub(super) profile_byte_offset: u64,
}

impl EntropyBatchPlan {
    pub(super) fn bytes(self, channels: usize) -> u64 {
        4 * (8
            + hybrid::PROFILES as u64
            + 4 * u64::from(self.group_count)
            + 4 * channels as u64
            + clustering::CONTEXTS as u64 * (TABLE_WORDS as u64 + 1))
    }
}

pub(super) struct EncodedGroup<'a> {
    bytes: &'a [u8],
    bit_len: usize,
}

pub(super) fn validate_encoded_group<'a>(
    entropy: EntropyArtifactPlan,
    bytes: &'a [u8],
    artifacts: &[ValidatedModularArtifact<'_>],
) -> Result<EncodedGroup<'a>, EncodeError> {
    let end = entropy
        .byte_offset
        .checked_add(entropy.bytes())
        .and_then(|end| usize::try_from(end).ok())
        .ok_or(BackendError::InvalidArtifact("ANS output range overflow"))?;
    let start = usize::try_from(entropy.byte_offset)
        .map_err(|_| BackendError::InvalidArtifact("ANS output offset overflow"))?;

    let data = bytes.get(start..end).ok_or(BackendError::InvalidArtifact(
        "truncated ANS output allocation",
    ))?;
    let word =
        |index: usize| u32::from_le_bytes(data[index * 4..index * 4 + 4].try_into().unwrap());
    let (status, bits, symbols, reserved) = (word(0), word(1), word(2), word(3));
    if status != 1 || reserved != 0 || bits < 32 || bits > entropy.capacity_words * 32 {
        return Err(
            BackendError::InvalidArtifact("invalid GPU ANS completion or bit count").into(),
        );
    }
    let expected_symbols = artifacts
        .iter()
        .flat_map(|artifact| artifact.events)
        .map(|event| if event.kind == 1 { 3u64 } else { 1 })
        .sum::<u64>();
    if u64::from(symbols) != expected_symbols {
        return Err(BackendError::InvalidArtifact("GPU ANS did not consume every event").into());
    }
    let bytes = &data[16..16 + bits.div_ceil(8) as usize];
    if bits % 8 != 0 && bytes.last().unwrap() >> (bits % 8) != 0 {
        return Err(BackendError::InvalidArtifact("GPU ANS padding is not zero").into());
    }
    Ok(EncodedGroup {
        bytes,
        bit_len: bits as usize,
    })
}

fn shader_source(body: &str) -> String {
    format!("{}\n{body}", include_str!("entropy/hybrid.wgsl"))
}

pub(super) struct AnsPipelines {
    pub(super) encode: Arc<wgpu::ComputePipeline>,
    pub(super) profile: wgpu::ComputePipeline,
}

impl AnsPipelines {
    pub(super) fn new(context: &WgpuContext) -> Arc<Self> {
        let module = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Modular hybrid histogram shader"),
                source: wgpu::ShaderSource::Wgsl(
                    shader_source(include_str!("entropy/profile.wgsl")).into(),
                ),
            });
        Arc::new(Self {
            encode: pipeline(context),
            profile: context
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Modular hybrid histogram pipeline"),
                    layout: None,
                    module: &module,
                    entry_point: Some("profile"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        })
    }
}

pub(super) fn pipeline(context: &WgpuContext) -> Arc<wgpu::ComputePipeline> {
    let module = context
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Modular GPU ANS serialization"),
            source: wgpu::ShaderSource::Wgsl(shader_source(include_str!("entropy.wgsl")).into()),
        });
    Arc::new(
        context
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Modular GPU ANS pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("encode"),
                compilation_options: Default::default(),
                cache: None,
            }),
    )
}

pub(super) struct AnsSubmission<'a> {
    pub(super) plan: &'a ModularDispatchPlan,
    pub(super) batch: &'a ModularDispatchBatch,
    pub(super) codebook: Option<&'a AnsCodebook>,
    pub(super) context: &'a WgpuContext,
    pub(super) pipeline: &'a wgpu::ComputePipeline,
    pub(super) parameters: &'a wgpu::Buffer,
    pub(super) artifact: &'a wgpu::Buffer,
}

pub(super) fn record(
    submission: AnsSubmission<'_>,
    commands: &mut wgpu::CommandEncoder,
) -> Result<(), EncodeError> {
    let AnsSubmission {
        plan,
        batch,
        codebook,
        context,
        pipeline,
        parameters,
        artifact,
    } = submission;

    let entropy = batch.entropy.ok_or(BackendError::Invariant(
        "ANS batch has no metadata allocation",
    ))?;
    let groups = &plan.groups[batch.first_dispatch..batch.first_dispatch + batch.dispatch_count];
    let channel_start = 8 + 4 * entropy.group_count as usize;
    let profiles_start = channel_start + 4 * groups.len();
    let tables_start = profiles_start + hybrid::PROFILES;
    let mut metadata = vec![0u32; tables_start];
    metadata[..8].copy_from_slice(&[
        entropy.group_count,
        tables_start as u32,
        u32::from(plan.lz77 == LosslessModularLz77::Greedy),
        u32::from(codebook.map_or(0, |code| code.context_map[0])),
        channel_start as u32,
        (entropy.profile_byte_offset / 4) as u32,
        profiles_start as u32,
        groups.len() as u32,
    ]);
    for (target, &config) in metadata[profiles_start..tables_start]
        .iter_mut()
        .zip(hybrid::HybridConfig::candidates())
    {
        *target = config.packed();
    }
    let mut job = 0;
    for (index, group) in groups.iter().enumerate() {
        let base = (group.artifact_byte_offset - batch.artifact_byte_offset) / 4;
        metadata[channel_start + index * 4..channel_start + index * 4 + 4].copy_from_slice(&[
            base as u32 + super::types::OUTPUT_HEADER_WORDS as u32,
            base as u32,
            group.max_events as u32,
            u32::from(codebook.map_or(group.channel.min(3) as u8 + 1, |code| {
                code.context_map[group.channel.min(3) as usize + 1]
            })),
        ]);
        if let Some(output) = group.entropy {
            let count = groups[index..]
                .iter()
                .take_while(|next| next.group_index == group.group_index)
                .count();
            metadata[8 + job * 4..12 + job * 4].copy_from_slice(&[
                (channel_start + index * 4) as u32,
                count as u32,
                (base + output.byte_offset / 4) as u32,
                output.capacity_words,
            ]);
            job += 1;
        }
    }
    if job != entropy.group_count as usize {
        return Err(BackendError::Invariant("ANS group inventory mismatch").into());
    }
    if let Some(codebook) = codebook {
        for table in &codebook.tables {
            metadata.push(table.config.packed());
            metadata.extend_from_slice(table.code.gpu_words());
        }
    }
    let metadata_bytes = metadata.len() as u64 * 4;
    if metadata_bytes > entropy.bytes(groups.len()) {
        return Err(BackendError::Invariant("ANS metadata exceeds admission").into());
    }
    context.queue().write_buffer(
        parameters,
        entropy.parameter_offset,
        bytemuck::cast_slice(&metadata),
    );
    let bindings = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Modular ANS bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: artifact.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: parameters,
                        offset: entropy.parameter_offset,
                        size: std::num::NonZeroU64::new(metadata_bytes),
                    }),
                },
            ],
        });
    let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(if codebook.is_some() {
            "Modular ANS group serialization"
        } else {
            "Modular hybrid histogram profiling"
        }),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bindings, &[]);
    if codebook.is_some() {
        pass.dispatch_workgroups(entropy.group_count, 1, 1);
    } else {
        pass.dispatch_workgroups(groups.len() as u32, hybrid::PROFILES as u32, 1);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
