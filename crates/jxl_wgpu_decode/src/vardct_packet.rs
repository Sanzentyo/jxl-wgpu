//! GPU entropy frontend for the bounded standard zero-AC regular-VarDCT packet profile.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{BitRange, BitReader, CodestreamInventory};
use jxl_gpu_protocol::TransformKind;
use thiserror::Error;

use crate::modular_tree::{MaTreeLimits, MaTreeNodeIr, PackedModularMetadata, parse_ma_config};
use crate::vardct_frontend::{
    LfGlobalPrefix, StandardVarDctProfile, VarDctFrontendError, VarDctPacketError,
    VarDctSectionLayout,
};

const SHADER_TEMPLATE: &str = include_str!("vardct_packet.wgsl");
const MODULAR_ENTROPY: &str = include_str!("modular_entropy.wgsl");
const MODULAR_RECONSTRUCT: &str = include_str!("modular_reconstruct.wgsl");
const ENTROPY_MARKER: &str = "/*__JXL_MODULAR_ENTROPY__*/";
const RECONSTRUCT_MARKER: &str = "/*__JXL_MODULAR_RECONSTRUCT__*/";

const GLOBAL_SCALE: u32 = 8_813;
const QUANT_LF: u32 = 10;
const RAW_HF_MULTIPLIER: u32 = 5;
const ZERO_AC_HF_GLOBAL: u32 = 0x2495;

/// A standard feature excluded from the deliberately bounded regular-VarDCT packet profile.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum UnsupportedVarDctPacketFeature {
    #[error("the bounded tiled-VarDCT decoder requires exactly one LF group")]
    MultipleLfGroups,
    #[error("pass group {group} contains {bits} AC payload bits; the bounded profile is zero-AC")]
    NonEmptyPassGroup { group: u32, bits: u64 },
    #[error("the bounded regular-VarDCT decoder currently accepts 8-bit samples")]
    BitDepth,
    #[error("the one-entry packet extent is not one implemented regular VarDCT transform")]
    TransformExtent,
    #[error("the standard packet requests skip-adaptive-LF smoothing")]
    SkipAdaptiveLfSmoothing,
    #[error("the packet does not use the fixed global_scale=8813 and quant_lf=10 profile")]
    Quantization,
    #[error("the MA tree uses property {property}; only channel and stream routing are supported")]
    MaProperty { property: u32 },
    #[error("the MA tree uses self-correcting prediction")]
    SelfCorrectingPredictor,
}

/// Host-side failure before image entropy is submitted to the GPU.
#[derive(Debug, Error)]
pub enum BoundedVarDctPacketError {
    #[error(transparent)]
    Frontend(#[from] VarDctFrontendError),
    #[error(transparent)]
    Packet(#[from] VarDctPacketError),
    #[error(transparent)]
    Unsupported(#[from] UnsupportedVarDctPacketFeature),
    #[error("failed to position the bounded MA-tree parser: {0}")]
    Bitstream(#[from] jxl_gpu_bitstream::Error),
    #[error("failed to parse the bounded MA-tree descriptor: {0}")]
    ModularTree(String),
    #[error("the packed MA-tree metadata ABI is malformed")]
    PackedMetadata,
    #[error("VarDCT packet arithmetic overflowed while computing {field}")]
    ArithmeticOverflow { field: &'static str },
}

/// GPU-reported validation failure. No output is authoritative after this error.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum GpuVarDctPacketError {
    #[error("GPU VarDCT packet entropy failed with status {code} at bit {cursor}/{end}")]
    Entropy { code: u32, cursor: u32, end: u32 },
    #[error("GPU VarDCT packet has invalid LF local header")]
    LfHeader,
    #[error("GPU VarDCT packet does not declare the negotiated bounded first-block count")]
    FirstBlock,
    #[error("GPU VarDCT packet has invalid HF metadata local header")]
    HfHeader,
    #[error("GPU VarDCT packet uses nonzero HF chroma correlation {value}")]
    Correlation { value: u32 },
    #[error("GPU VarDCT packet selects strategy {actual}, expected {expected}")]
    Strategy { actual: u32, expected: u32 },
    #[error("GPU VarDCT packet reconstructs raw HF multiplier {actual}, expected 5")]
    HfMultiplier { actual: u32 },
    #[error("GPU VarDCT packet uses nonzero EPF sharpness {value}")]
    Sharpness { value: u32 },
    #[error("GPU VarDCT packet does not contain the standard zero-AC HF-global bundle")]
    HfGlobal,
    #[error("GPU VarDCT packet returned unknown status {code}")]
    Unknown { code: u32 },
}

/// Parsed host metadata and untouched image entropy for one strict packet.
#[derive(Clone, Debug)]
pub struct BoundedVarDctPacketPlan {
    pub profile: StandardVarDctProfile,
    /// Regular transform shared by every first block in this bounded packet.
    pub transform: TransformKind,
    /// Number of non-overlapping first blocks reconstructed from HF metadata.
    pub task_count: u32,
    coefficient_words: u32,
    /// LF-global packet containing the scalar quantizer fields and global MA descriptor.
    pub lf_global: BitRange,
    /// LF-group packet containing quantized LF and HF-metadata Modular streams.
    pub lf_group: BitRange,
    /// Separate HF-global packet, or `None` when all three packets share a single TOC entry.
    pub hf_global: Option<BitRange>,
    /// Descriptor end used as the LF-group start by the one-entry TOC form.
    pub entropy_bit_offset: u32,
    pub lf_stream_index: u32,
    pub hf_stream_index: u32,
    pub modular_metadata: Vec<u32>,
    /// Physical power-of-two history ring used by both sequential Modular streams.
    pub lz77_window_words: u32,
}

impl BoundedVarDctPacketPlan {
    /// Parses bounded scalar metadata only. Image symbols remain encoded for the GPU.
    pub fn parse(
        codestream: &[u8],
        inventory: &CodestreamInventory,
    ) -> Result<Self, BoundedVarDctPacketError> {
        let profile = StandardVarDctProfile::negotiate(inventory)?;
        if profile.low_frequency_group_count != 1 {
            return Err(UnsupportedVarDctPacketFeature::MultipleLfGroups.into());
        }
        if profile.bits_per_sample != 8 {
            return Err(UnsupportedVarDctPacketFeature::BitDepth.into());
        }
        if !profile.adaptive_lf_smoothing {
            return Err(UnsupportedVarDctPacketFeature::SkipAdaptiveLfSmoothing.into());
        }
        let (transform, task_count, lf_global_packet, lf_group, hf_global) = match &profile.sections
        {
            VarDctSectionLayout::Single { packet } => (
                transform_for_extent(profile.width, profile.height)
                    .ok_or(UnsupportedVarDctPacketFeature::TransformExtent)?,
                1,
                *packet,
                *packet,
                None,
            ),
            VarDctSectionLayout::Sections {
                lf_global,
                lf_groups,
                hf_global,
                pass_groups,
            } => {
                let lf_group = *lf_groups
                    .first()
                    .ok_or(UnsupportedVarDctPacketFeature::MultipleLfGroups)?;
                for (group, packet) in pass_groups.iter().copied().enumerate() {
                    if packet.length != 0 {
                        return Err(UnsupportedVarDctPacketFeature::NonEmptyPassGroup {
                            group: u32::try_from(group).map_err(|_| {
                                BoundedVarDctPacketError::ArithmeticOverflow {
                                    field: "pass-group index",
                                }
                            })?,
                            bits: packet.length,
                        }
                        .into());
                    }
                }
                let blocks = profile
                    .width
                    .div_ceil(8)
                    .checked_mul(profile.height.div_ceil(8))
                    .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "tiled DCT8 task count",
                    })?;
                (
                    TransformKind::Dct8,
                    blocks,
                    *lf_global,
                    lf_group,
                    Some(*hf_global),
                )
            }
        };
        let lf_global = LfGlobalPrefix::parse(codestream, lf_global_packet)?;
        if lf_global.global_scale != GLOBAL_SCALE || lf_global.quant_lf != QUANT_LF {
            return Err(UnsupportedVarDctPacketFeature::Quantization.into());
        }
        let mut reader = BitReader::new(codestream);
        reader.skip_bits(lf_global.ma_tree_bit_offset)?;
        let ma_config = parse_ma_config(&mut reader, MaTreeLimits::default())
            .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?;
        let descriptor_end = reader.bit_offset();
        let lf_global_end =
            lf_global_packet
                .end()
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "LF-global end",
                })?;
        if descriptor_end > lf_global_end {
            return Err(VarDctPacketError::PacketBoundary {
                cursor: descriptor_end,
                packet_end: lf_global_end,
            }
            .into());
        }
        if ma_config.needs_self_correcting() {
            return Err(UnsupportedVarDctPacketFeature::SelfCorrectingPredictor.into());
        }
        for node in &ma_config.nodes {
            if let MaTreeNodeIr::Decision { property, .. } = *node
                && property > 1
            {
                return Err(UnsupportedVarDctPacketFeature::MaProperty { property }.into());
            }
        }
        let PackedModularMetadata { words } = ma_config
            .pack_gpu_metadata()
            .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?;
        if words.len() <= 9 {
            return Err(BoundedVarDctPacketError::PackedMetadata);
        }
        let entropy_bit_offset = u32::try_from(descriptor_end).map_err(|_| {
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "entropy bit offset",
            }
        })?;
        let blocks_x = profile.width.div_ceil(8);
        let blocks_y = profile.height.div_ceil(8);
        let block_count =
            blocks_x
                .checked_mul(blocks_y)
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "block count",
                })?;
        let coefficient_words = block_count.checked_mul(8 * 8 * 3).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "padded coefficient words",
            },
        )?;
        let correlation_samples = profile
            .width
            .div_ceil(64)
            .checked_mul(profile.height.div_ceil(64))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "correlation sample count",
            })?;
        let decoded_symbol_limit = block_count
            .checked_mul(4)
            .and_then(|samples| {
                task_count
                    .checked_mul(2)
                    .and_then(|tasks| samples.checked_add(tasks))
            })
            .and_then(|samples| {
                correlation_samples
                    .checked_mul(2)
                    .and_then(|correlations| samples.checked_add(correlations))
            })
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "decoded symbol limit",
            })?;
        let lz77_window_words = ma_config
            .entropy
            .lz77_window_words(blocks_x.max(1), decoded_symbol_limit)
            .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?;
        Ok(Self {
            lf_stream_index: profile.lf_quant_stream_index(0)?,
            hf_stream_index: profile.hf_metadata_stream_index(0)?,
            profile,
            transform,
            task_count,
            coefficient_words,
            lf_global: lf_global_packet,
            lf_group,
            hf_global,
            entropy_bit_offset,
            modular_metadata: words,
            lz77_window_words,
        })
    }

    #[must_use]
    pub fn coefficient_words(&self) -> u32 {
        self.coefficient_words
    }

    #[must_use]
    pub fn block_extent(&self) -> [u32; 2] {
        [
            self.profile.width.div_ceil(8),
            self.profile.height.div_ceil(8),
        ]
    }

    /// U32 scratch words retaining LF samples plus the conservative LZ history ring.
    pub fn reconstructed_words(&self) -> Result<u32, BoundedVarDctPacketError> {
        let [blocks_x, blocks_y] = self.block_extent();
        let blocks =
            blocks_x
                .checked_mul(blocks_y)
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "block count",
                })?;
        let correlations = self.correlation_samples()?;
        let hf_samples = self
            .task_count
            .checked_mul(2)
            .and_then(|tasks| blocks.checked_add(tasks))
            .and_then(|samples| {
                correlations
                    .checked_mul(2)
                    .and_then(|cfl| samples.checked_add(cfl))
            });
        let samples = blocks
            .checked_mul(3)
            .zip(hf_samples)
            .map(|(lf, hf)| lf.max(hf))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "VarDCT reconstruction samples",
            })?;
        samples.checked_add(self.lz77_window_words).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "VarDCT reconstruction scratch",
            },
        )
    }

    pub fn packet_control(&self) -> Result<VarDctPacketControl, BoundedVarDctPacketError> {
        let range = |value: u64, field: &'static str| {
            u32::try_from(value).map_err(|_| BoundedVarDctPacketError::ArithmeticOverflow { field })
        };
        let lf_group_end =
            self.lf_group
                .end()
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "LF-group end",
                })?;
        let (lf_start, lf_end, hf_start, hf_end, separate_sections) = if let Some(hf_global) =
            self.hf_global
        {
            let hf_end = hf_global
                .end()
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "HF-global end",
                })?;
            (
                range(self.lf_group.offset, "LF-group start")?,
                range(lf_group_end, "LF-group end")?,
                range(hf_global.offset, "HF-global start")?,
                range(hf_end, "HF-global end")?,
                1,
            )
        } else {
            (
                self.entropy_bit_offset,
                range(lf_group_end, "combined packet end")?,
                0,
                range(lf_group_end, "combined packet end")?,
                0,
            )
        };
        let [blocks_x, blocks_y] = self.block_extent();
        let block_count =
            blocks_x
                .checked_mul(blocks_y)
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "block count",
                })?;
        let correlation_samples = self.correlation_samples()?;
        let strategy_offset = correlation_samples.checked_mul(2).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "strategy offset",
            },
        )?;
        let hf_mul_offset = strategy_offset.checked_add(self.task_count).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF multiplier offset",
            },
        )?;
        let sharpness_offset = hf_mul_offset.checked_add(self.task_count).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "sharpness offset",
            },
        )?;
        let raw_capacity = sharpness_offset.checked_add(block_count).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "raw metadata capacity",
            },
        )?;
        Ok(VarDctPacketControl {
            section_bits: [lf_start, lf_end, hf_start, hf_end],
            geometry: [self.profile.width, self.profile.height, blocks_x, blocks_y],
            offsets: [0, correlation_samples, strategy_offset, hf_mul_offset],
            capacities: [
                self.coefficient_words(),
                raw_capacity,
                block_count
                    .checked_next_power_of_two()
                    .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "first-block field width",
                    })?
                    .trailing_zeros(),
                self.task_count,
            ],
            expected: [
                transform_id(self.transform),
                RAW_HF_MULTIPLIER,
                ZERO_AC_HF_GLOBAL,
                sharpness_offset,
            ],
            quantization: [GLOBAL_SCALE, QUANT_LF, 0, 0],
            streams: [
                self.lf_stream_index,
                self.hf_stream_index,
                separate_sections,
                u32::try_from(self.profile.group_count).map_err(|_| {
                    BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "pass-group count",
                    }
                })?,
            ],
            _reserved1: [0; 4],
        })
    }

    fn correlation_samples(&self) -> Result<u32, BoundedVarDctPacketError> {
        self.profile
            .width
            .div_ceil(64)
            .checked_mul(self.profile.height.div_ceil(64))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "correlation sample count",
            })
    }
}

/// Exact 128-byte uniform consumed by `vardct_packet.wgsl`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VarDctPacketControl {
    pub section_bits: [u32; 4],
    pub geometry: [u32; 4],
    pub offsets: [u32; 4],
    pub capacities: [u32; 4],
    pub expected: [u32; 4],
    pub quantization: [u32; 4],
    pub streams: [u32; 4],
    pub _reserved1: [u32; 4],
}

/// Generic Modular parameter ABI retained by the composable entropy/reconstruction fragments.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VarDctModularParams {
    pub words: [u32; 52],
}

impl Default for VarDctModularParams {
    fn default() -> Self {
        let mut words = [0; 52];
        words[41] = 16;
        words[42] = 10;
        words[43] = 7;
        words[44] = 7;
        words[45] = 7;
        words[48] = 13;
        words[49] = 12;
        words[50] = 12;
        words[51] = 12;
        Self { words }
    }
}

impl VarDctModularParams {
    /// Sets the exact power-of-two LZ ring represented by the packed entropy descriptor.
    pub fn with_lz77_window(mut self, words: u32) -> Self {
        self.words[12] = words.saturating_sub(1);
        self
    }
}

/// Exact 64-byte status written once by the serial packet parser.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuVarDctPacketStatus {
    pub code: u32,
    pub cursor: u32,
    pub expected_end: u32,
    pub lf_decoded: u32,
    pub hf_decoded: u32,
    pub strategy: u32,
    pub hf_mul: u32,
    pub coefficient_words: u32,
    pub detail: u32,
    pub global_scale: u32,
    pub quant_lf: u32,
    pub _reserved: [u32; 5],
}

impl GpuVarDctPacketStatus {
    pub fn validate(
        self,
        expected_strategy: TransformKind,
        expected_lf_samples: u32,
        expected_hf_samples: u32,
    ) -> Result<(), GpuVarDctPacketError> {
        match self.code {
            1 if self.cursor == self.expected_end
                && self.lf_decoded == expected_lf_samples
                && self.hf_decoded == expected_hf_samples
                && self.strategy == transform_id(expected_strategy)
                && self.hf_mul == RAW_HF_MULTIPLIER + 1
                && self.global_scale == GLOBAL_SCALE
                && self.quant_lf == QUANT_LF =>
            {
                Ok(())
            }
            1 => Err(GpuVarDctPacketError::Entropy {
                code: self.code,
                cursor: self.cursor,
                end: self.expected_end,
            }),
            20 => Err(GpuVarDctPacketError::LfHeader),
            21 => Err(GpuVarDctPacketError::FirstBlock),
            22 => Err(GpuVarDctPacketError::HfHeader),
            23 => Err(GpuVarDctPacketError::Correlation { value: self.detail }),
            24 => Err(GpuVarDctPacketError::Strategy {
                actual: self.detail,
                expected: transform_id(expected_strategy),
            }),
            25 => Err(GpuVarDctPacketError::HfMultiplier {
                actual: self.detail,
            }),
            26 => Err(GpuVarDctPacketError::Sharpness { value: self.detail }),
            27 => Err(GpuVarDctPacketError::HfGlobal),
            2..=13 => Err(GpuVarDctPacketError::Entropy {
                code: self.code,
                cursor: self.cursor,
                end: self.expected_end,
            }),
            code => Err(GpuVarDctPacketError::Unknown { code }),
        }
    }
}

/// Buffers supplied to [`VarDctPacketPipeline::encode`].
pub struct VarDctPacketBuffers<'a> {
    pub codestream: &'a wgpu::Buffer,
    pub modular_metadata: &'a wgpu::Buffer,
    pub reconstructed_lf: &'a wgpu::Buffer,
    pub raw_hf_metadata: &'a wgpu::Buffer,
    pub coefficients: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
    pub control: &'a wgpu::Buffer,
    pub modular_params: &'a wgpu::Buffer,
}

/// Reusable serial control-plane decoder. Image entropy is decoded in WGSL.
pub struct VarDctPacketPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl VarDctPacketPipeline {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let source = shader_source();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu bounded VarDCT packet frontend"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu bounded VarDCT packet frontend"),
            layout: None,
            module: &module,
            entry_point: Some("decode_vardct_packet"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: VarDctPacketBuffers<'_>,
    ) {
        let layout = self.pipeline.get_bind_group_layout(0);
        let resources = [
            buffers.codestream,
            buffers.modular_metadata,
            buffers.reconstructed_lf,
            buffers.raw_hf_metadata,
            buffers.coefficients,
            buffers.status,
            buffers.control,
            buffers.modular_params,
        ];
        let entries = resources
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu bounded VarDCT packet bindings"),
            layout: &layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu bounded VarDCT packet frontend"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

#[must_use]
pub fn vardct_packet_shader_source() -> String {
    shader_source()
}

fn shader_source() -> String {
    SHADER_TEMPLATE
        .replace(ENTROPY_MARKER, MODULAR_ENTROPY)
        .replace(RECONSTRUCT_MARKER, MODULAR_RECONSTRUCT)
}

fn transform_for_extent(width: u32, height: u32) -> Option<TransformKind> {
    [
        TransformKind::Dct8,
        TransformKind::Dct16x16,
        TransformKind::Dct32x32,
        TransformKind::Dct16x8,
        TransformKind::Dct8x16,
        TransformKind::Dct32x8,
        TransformKind::Dct8x32,
        TransformKind::Dct32x16,
        TransformKind::Dct16x32,
    ]
    .into_iter()
    .find(|transform| {
        let extent = transform.pixel_extent();
        extent.width == width && extent.height == height
    })
}

const fn transform_id(transform: TransformKind) -> u32 {
    let mut index = 0;
    while index < TransformKind::ALL.len() {
        if TransformKind::ALL[index] as u8 == transform as u8 {
            return index as u32;
        }
        index += 1;
    }
    u32::MAX
}

const _: () = {
    assert!(std::mem::size_of::<VarDctPacketControl>() == 128);
    assert!(std::mem::align_of::<VarDctPacketControl>() == 16);
    assert!(std::mem::size_of::<VarDctModularParams>() == 208);
    assert!(std::mem::size_of::<GpuVarDctPacketStatus>() == 64);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_shader_is_portable_wgsl() {
        let module = naga::front::wgsl::parse_str(&shader_source()).unwrap();
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator.validate(&module).unwrap();
    }

    #[test]
    fn tiled_profile_uses_the_standard_dct8_strategy_id() {
        assert_eq!(transform_id(TransformKind::Dct8), 0);
        assert_eq!(transform_for_extent(16, 32), Some(TransformKind::Dct32x16));
    }
}
