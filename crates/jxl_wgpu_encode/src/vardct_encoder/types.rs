//! VarDCT contracts, ABI records, and frame geometry.

use jxl_gpu_bitstream::FiniteF16;
use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, PixelFormat,
    PlaneFormat, PlaneSampling, SampleKind, Swizzle,
};
use jxl_gpu_protocol::Extent2d;

pub use jxl_gpu_protocol::TransformKind as VarDctStrategy;
use jxl_wgpu::ForwardVarDctMemoryPlan;

use super::entropy::{UINT_SYMBOLS, VarDctPrefixCode};
use crate::{EncodeError, UnsupportedFeature};

pub(super) const HF_QUANTIZATION: [f32; 3] = [1.25, 1.0, 1.0];

pub(super) const AC_GROUP_DIM_PIXELS: u32 = 256;
pub(super) const LF_GROUP_DIM_PIXELS: u32 = 2_048;
pub(super) const HEADER_WORDS: u32 = 68;
pub(super) const SECTION_ALIGNMENT_WORDS: u32 = 64;
pub(super) const ARTIFACT_READY: u32 = 0x5644_4354;
pub(super) const SINGLE_TRANSFORM_TOPOLOGY: u32 = 0;
pub(super) const TILED_DCT8_TOPOLOGY: u32 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VarDctColorEncoding {
    #[default]
    SrgbD65,
}

/// Exact LF dequantization and chroma-from-luma metadata serialized in a VarDCT frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VarDctLfMetadata {
    pub(super) lf_dequantization: [FiniteF16; 3],
    pub(super) colour_factor: u32,
    pub(super) base_correlation: [FiniteF16; 2],
    pub(super) lf_factors: [i8; 2],
}

impl VarDctLfMetadata {
    /// Validates an exact JPEG XL LF metadata bundle.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a dequantization multiplier, colour factor, or base
    /// correlation is outside the interoperable JPEG XL range.
    pub fn new(
        lf_dequantization: [FiniteF16; 3],
        colour_factor: u32,
        base_correlation: [FiniteF16; 2],
        lf_factors: [i8; 2],
    ) -> Result<Self, EncodeError> {
        for (channel, value) in ["X", "Y", "B"]
            .into_iter()
            .zip(lf_dequantization.map(FiniteF16::to_f32))
        {
            if value / 128.0 < 1.0e-8 {
                return Err(EncodeError::VarDctLfDequantization { channel, value });
            }
        }
        if !(2..=65_793).contains(&colour_factor) {
            return Err(EncodeError::VarDctColourFactor {
                value: colour_factor,
            });
        }
        for (channel, value) in ["X", "B"]
            .into_iter()
            .zip(base_correlation.map(FiniteF16::to_f32))
        {
            if value.abs() > 4.0 {
                return Err(EncodeError::VarDctBaseCorrelation { channel, value });
            }
        }
        Ok(Self {
            lf_dequantization,
            colour_factor,
            base_correlation,
            lf_factors,
        })
    }

    #[must_use]
    pub const fn lf_dequantization(self) -> [FiniteF16; 3] {
        self.lf_dequantization
    }

    #[must_use]
    pub const fn colour_factor(self) -> u32 {
        self.colour_factor
    }

    #[must_use]
    pub const fn base_correlation(self) -> [FiniteF16; 2] {
        self.base_correlation
    }

    #[must_use]
    pub const fn lf_factors(self) -> [i8; 2] {
        self.lf_factors
    }

    pub(super) fn has_default_dequantization(self) -> bool {
        self.lf_dequantization == Self::default().lf_dequantization
    }

    pub(super) fn has_default_correlation(self) -> bool {
        let default = Self::default();
        self.colour_factor == default.colour_factor
            && self.base_correlation == default.base_correlation
            && self.lf_factors == default.lf_factors
    }

    pub(super) fn forward_quantization(self) -> ([f32; 3], [f32; 2]) {
        let inverse_dequantization = self
            .lf_dequantization
            .map(|value| 1.0 / (512.0 * value.to_f32()));
        let inverse_colour_factor = 1.0 / self.colour_factor as f32;
        let base = self.base_correlation.map(FiniteF16::to_f32);
        let correlation = [
            base[0] + f32::from(self.lf_factors[0]) * inverse_colour_factor,
            base[1] + f32::from(self.lf_factors[1]) * inverse_colour_factor,
        ];
        (inverse_dequantization, correlation)
    }

    pub(super) fn hf_correlation(self) -> [f32; 2] {
        self.base_correlation.map(FiniteF16::to_f32)
    }
}

impl Default for VarDctLfMetadata {
    fn default() -> Self {
        Self {
            lf_dequantization: [0x2800, 0x3400, 0x3800]
                .map(|bits| FiniteF16::from_bits(bits).expect("default LF F16 is finite")),
            colour_factor: 84,
            base_correlation: [0x0000, 0x3c00]
                .map(|bits| FiniteF16::from_bits(bits).expect("default correlation F16 is finite")),
            lf_factors: [0, 0],
        }
    }
}

impl VarDctColorEncoding {
    /// Canonical three-byte pitch-linear input format. Layouts may add an
    /// arbitrary validated byte offset and row padding.
    #[must_use]
    pub fn pixel_format(self) -> PixelFormat {
        match self {
            Self::SrgbD65 => PixelFormat {
                model: ColorModel::Rgb,
                color_spec: ColorSpecification::Default,
                chroma_subsampling: ChromaSubsampling::None,
                sample_kind: SampleKind::Unsigned,
                byte_order: ByteOrder::Native,
                swizzle: Swizzle::XYZ1,
                planes: vec![PlaneFormat::separate_words(
                    PlaneSampling::FULL,
                    1,
                    &[Channel::X, Channel::Y, Channel::Z],
                    8,
                )],
            },
        }
    }
}

/// Standard block and pass-group grid selected by the tiled DCT8 profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TiledVarDctGrid {
    pub width: u32,
    pub height: u32,
    pub block_columns: u32,
    pub block_rows: u32,
    pub ac_group_columns: u32,
    pub ac_group_rows: u32,
    pub lf_group_columns: u32,
    pub lf_group_rows: u32,
}

impl TiledVarDctGrid {
    /// Pixel dimension of one standard AC/pass group.
    pub const AC_GROUP_DIMENSION: u32 = AC_GROUP_DIM_PIXELS;
    /// Pixel dimension of one standard LF/DC group.
    pub const LF_GROUP_DIMENSION: u32 = LF_GROUP_DIM_PIXELS;
    /// Largest source axis exercised by the checked GPU profile.
    pub const MAX_DIMENSION: u32 = 16_384;

    /// Derives the exact block, LF-group, and AC-group grids without allocating GPU data.
    pub fn new(width: u32, height: u32) -> Result<Self, EncodeError> {
        if width == 0 || height == 0 {
            return Err(EncodeError::InvalidSource(
                "tiled VarDCT dimensions must be nonzero",
            ));
        }
        if width > Self::MAX_DIMENSION || height > Self::MAX_DIMENSION {
            return Err(UnsupportedFeature::TiledVarDctDimensions {
                width,
                height,
                max_dimension: Self::MAX_DIMENSION,
            }
            .into());
        }
        let grid = Self {
            width,
            height,
            block_columns: width.div_ceil(8),
            block_rows: height.div_ceil(8),
            ac_group_columns: width.div_ceil(Self::AC_GROUP_DIMENSION),
            ac_group_rows: height.div_ceil(Self::AC_GROUP_DIMENSION),
            lf_group_columns: width.div_ceil(Self::LF_GROUP_DIMENSION),
            lf_group_rows: height.div_ceil(Self::LF_GROUP_DIMENSION),
        };
        Ok(grid)
    }

    pub fn block_count(self) -> Result<u32, EncodeError> {
        self.block_columns
            .checked_mul(self.block_rows)
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT block count overflow",
            ))
    }

    pub fn ac_group_count(self) -> Result<u32, EncodeError> {
        self.ac_group_columns.checked_mul(self.ac_group_rows).ok_or(
            EncodeError::InvalidConfiguration("VarDCT AC group count overflow"),
        )
    }

    pub fn lf_group_count(self) -> Result<u32, EncodeError> {
        self.lf_group_columns.checked_mul(self.lf_group_rows).ok_or(
            EncodeError::InvalidConfiguration("VarDCT LF group count overflow"),
        )
    }

    /// One fused packet for a single AC group; otherwise DC global, every DC
    /// group, AC global, then one pass packet per AC group.
    pub fn toc_entries(self) -> Result<u32, EncodeError> {
        if self.ac_group_count()? == 1 {
            return Ok(1);
        }
        self.lf_group_count()?
            .checked_add(self.ac_group_count()?)
            .and_then(|groups| groups.checked_add(2))
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT TOC entry count overflow",
            ))
    }
}

/// GPU artifact implementation selected for a VarDCT memory plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarDctKernelLayout {
    /// A validated image-wide map, batched by strategy with shared resident arenas.
    StrategyMap,
    /// One complete transform with GPU-resident coefficients and LF.
    SingleTransform,
    /// Runtime-sized artifact where every 8x8 block is an independent DCT8
    /// transform and the frame may contain multiple 2,048-pixel LF groups and
    /// 256-pixel AC groups.
    TiledDct8,
}

/// Explicit allocations retained by one in-flight VarDCT submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctMemoryPlan {
    pub kernel_layout: VarDctKernelLayout,
    /// Source bytes made addressable by the storage binding. The caller owns
    /// this allocation, so it is not charged to `owned_bytes_per_job`.
    pub source_binding_bytes: u64,
    pub parameter_storage_bytes: u64,
    pub artifact_storage_bytes: u64,
    pub readback_bytes: u64,
    /// Tiled DCT8's X/Y/B order table; general transforms include orders in `transform`.
    pub coefficient_order_bytes: u64,
    /// Resident forward-transform allocations, retained until submission completion.
    pub transform: Option<VarDctTransformMemoryPlan>,
    pub owned_bytes_per_job: u64,
    pub addressed_bytes_per_job: u64,
}

/// Exact resident storage for one general forward transform and its quantizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctTransformMemoryPlan {
    pub forward: ForwardVarDctMemoryPlan,
    pub xyb_bytes: u64,
    pub coefficient_bytes: u64,
    pub lf_bytes: u64,
    pub quantized_bytes: u64,
    pub quantization_metadata_bytes: u64,
    pub task_metadata_bytes: u64,
    pub total_bytes: u64,
}

impl VarDctTransformMemoryPlan {
    #[must_use]
    pub const fn new(strategy: VarDctStrategy) -> Self {
        let forward = ForwardVarDctMemoryPlan::new(strategy);
        let xyb_bytes = forward.coefficient_bytes;
        let coefficient_bytes = forward.coefficient_bytes;
        let lf_bytes = forward.lf_bytes;
        let quantized_bytes = forward.coefficient_bytes;
        let quantization_metadata_bytes = coefficient_bytes / 3 * 6;
        Self {
            forward,
            xyb_bytes,
            coefficient_bytes,
            lf_bytes,
            quantized_bytes,
            quantization_metadata_bytes,
            task_metadata_bytes: std::mem::size_of::<super::strategy_map::TransformTask>() as u64,
            total_bytes: forward.transient_bytes
                + xyb_bytes
                + coefficient_bytes
                + lf_bytes
                + quantized_bytes
                + quantization_metadata_bytes
                + std::mem::size_of::<super::strategy_map::TransformTask>() as u64,
        }
    }
}

impl VarDctMemoryPlan {
    pub(super) const fn new(
        source_binding_bytes: u64,
        artifact_storage_bytes: u64,
        kernel_layout: VarDctKernelLayout,
    ) -> Self {
        let parameter_storage_bytes = std::mem::size_of::<VarDctKernelParams>() as u64;
        let readback_bytes = artifact_storage_bytes;
        let coefficient_order_bytes = if matches!(kernel_layout, VarDctKernelLayout::TiledDct8) {
            64 * 3 * 4
        } else {
            0
        };
        let owned_bytes_per_job = parameter_storage_bytes
            + artifact_storage_bytes
            + readback_bytes
            + coefficient_order_bytes;
        Self {
            kernel_layout,
            source_binding_bytes,
            parameter_storage_bytes,
            artifact_storage_bytes,
            readback_bytes,
            coefficient_order_bytes,
            transform: None,
            owned_bytes_per_job,
            addressed_bytes_per_job: source_binding_bytes + owned_bytes_per_job,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GpuPrefixEntry {
    pub(super) bits: u32,
    pub(super) bit_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct VarDctKernelParams {
    pub(super) row_stride: u32,
    pub(super) byte_offset: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) blocks_x: u32,
    pub(super) blocks_y: u32,
    pub(super) strategy: u32,
    pub(super) global_scale: u32,
    pub(super) quant_lf: u32,
    pub(super) hf_multiplier: u32,
    pub(super) raw_prefix: [GpuPrefixEntry; UINT_SYMBOLS],
    pub(super) strategy_offset: u32,
    pub(super) dc_offset: u32,
    pub(super) token_offset: u32,
    pub(super) extra_offset: u32,
    pub(super) fragment_offset: u32,
    pub(super) fragment_word_capacity: u32,
    pub(super) artifact_words: u32,
    pub(super) topology: u32,
    pub(super) fragment_descriptor_offset: u32,
    pub(super) fragment_descriptor_len: u32,
    pub(super) lf_groups_x: u32,
    pub(super) lf_groups_y: u32,
    pub(super) lf_quantization: [f32; 3],
    pub(super) lf_correlation: [f32; 2],
    pub(super) hf_prefix: [GpuPrefixEntry; UINT_SYMBOLS],
    pub(super) hf_correlation: [f32; 2],
    pub(super) hf_quantization: [f32; 3],
    pub(super) ac_descriptor_offset: u32,
    pub(super) ac_descriptor_len: u32,
    pub(super) ac_fragment_offset: u32,
    pub(super) ac_words_per_block: u32,
    pub(super) ac_fragment_words: u32,
    pub(super) workgroups_x: u32,
    pub(super) padding: [u32; 22],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct VarDctArtifactHeader {
    pub(super) status: u32,
    pub(super) block_count: u32,
    pub(super) dc_sample_count: u32,
    pub(super) strategy: u32,
    pub(super) ac_payload: u32,
    pub(super) strategy_offset: u32,
    pub(super) strategy_len: u32,
    pub(super) dc_offset: u32,
    pub(super) dc_len: u32,
    pub(super) token_offset: u32,
    pub(super) token_len: u32,
    pub(super) extra_offset: u32,
    pub(super) extra_len: u32,
    pub(super) fragment_offset: u32,
    pub(super) fragment_word_capacity: u32,
    pub(super) dc_fragment_bit_len: u32,
    pub(super) artifact_words: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) blocks_x: u32,
    pub(super) blocks_y: u32,
    pub(super) topology: u32,
    pub(super) raw_histogram: [u32; UINT_SYMBOLS],
    pub(super) fragment_descriptor_offset: u32,
    pub(super) fragment_descriptor_len: u32,
    pub(super) lf_groups_x: u32,
    pub(super) lf_groups_y: u32,
    pub(super) lf_group_count: u32,
    pub(super) ac_descriptor_offset: u32,
    pub(super) ac_descriptor_len: u32,
    pub(super) ac_fragment_offset: u32,
    pub(super) ac_words_per_block: u32,
    pub(super) ac_fragment_words: u32,
    pub(super) padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct DcFragmentDescriptor {
    pub(super) bit_offset: u32,
    pub(super) bit_len: u32,
}

const _: () = {
    assert!(std::mem::size_of::<GpuPrefixEntry>() == 8);
    assert!(std::mem::align_of::<GpuPrefixEntry>() == 4);
    assert!(std::mem::size_of::<VarDctKernelParams>() == 768);
    assert!(std::mem::align_of::<VarDctKernelParams>() == 4);
    assert!(std::mem::size_of::<VarDctArtifactHeader>() == 272);
    assert!(std::mem::align_of::<VarDctArtifactHeader>() == 4);
    assert!(std::mem::size_of::<DcFragmentDescriptor>() == 8);
    assert!(std::mem::align_of::<DcFragmentDescriptor>() == 4);
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ArtifactLayout {
    pub(super) fragment_descriptor_offset: u32,
    pub(super) fragment_descriptor_len: u32,
    pub(super) strategy_offset: u32,
    pub(super) strategy_len: u32,
    pub(super) dc_offset: u32,
    pub(super) dc_len: u32,
    pub(super) token_offset: u32,
    pub(super) token_len: u32,
    pub(super) extra_offset: u32,
    pub(super) extra_len: u32,
    pub(super) fragment_offset: u32,
    pub(super) fragment_word_capacity: u32,
    pub(super) fragment_max_bits: u32,
    pub(super) ac_descriptor_offset: u32,
    pub(super) ac_descriptor_len: u32,
    pub(super) ac_fragment_offset: u32,
    pub(super) ac_words_per_block: u32,
    pub(super) ac_fragment_words: u32,
    pub(super) artifact_words: u32,
}

impl ArtifactLayout {
    pub(super) fn for_strategy_map(
        frame: VarDctFrameLayout,
        code: &VarDctPrefixCode,
        transforms: u32,
        ac_words: u32,
    ) -> Result<Self, EncodeError> {
        let mut layout = Self::for_block_grid(
            frame.blocks_x,
            frame.blocks_y,
            frame.lf_group_count()?,
            code,
        )?;
        layout.ac_descriptor_offset = layout.artifact_words;
        layout.ac_descriptor_len = transforms;
        layout.ac_fragment_offset =
            align_words(layout.ac_descriptor_offset.checked_add(transforms).ok_or(
                EncodeError::InvalidConfiguration("VarDCT AC descriptor overflow"),
            )?)?;
        layout.ac_fragment_words = ac_words;
        layout.artifact_words =
            align_words(layout.ac_fragment_offset.checked_add(ac_words).ok_or(
                EncodeError::InvalidConfiguration("VarDCT AC arena overflow"),
            )?)?;
        Ok(layout)
    }
    pub(super) fn new(
        strategy: VarDctStrategy,
        code: &VarDctPrefixCode,
    ) -> Result<Self, EncodeError> {
        let Extent2d {
            width: blocks_x,
            height: blocks_y,
        } = strategy.lf_extent();
        let layout = Self::for_block_grid(blocks_x, blocks_y, 1, code)?;
        let nonzero_count = blocks_x * blocks_y * 63;
        layout.with_ac(
            nonzero_count,
            1,
            &super::entropy::HfEntropyPlan::single_cluster_prefix()?,
        )
    }

    fn for_block_grid(
        blocks_x: u32,
        blocks_y: u32,
        lf_group_count: u32,
        code: &VarDctPrefixCode,
    ) -> Result<Self, EncodeError> {
        let strategy_len =
            blocks_x
                .checked_mul(blocks_y)
                .ok_or(EncodeError::InvalidConfiguration(
                    "VarDCT block count overflow",
                ))?;
        let dc_len = strategy_len
            .checked_mul(3)
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT DC sample count overflow",
            ))?;
        let max_bits_per_sample = code
            .raw_entries()
            .into_iter()
            .enumerate()
            .map(|(token, entry)| {
                let extra_bits = u32::try_from(token.saturating_sub(1))
                    .expect("the fixed entropy alphabet fits u32");
                u32::from(entry.bit_len) + extra_bits
            })
            .max()
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT entropy alphabet must not be empty",
            ))?;
        let fragment_max_bits =
            dc_len
                .checked_mul(max_bits_per_sample)
                .ok_or(EncodeError::InvalidConfiguration(
                    "VarDCT entropy fragment capacity overflow",
                ))?;
        let fragment_word_capacity =
            fragment_max_bits
                .checked_add(31)
                .ok_or(EncodeError::InvalidConfiguration(
                    "VarDCT entropy fragment word count overflow",
                ))?
                / 32;

        let fragment_descriptor_offset = HEADER_WORDS;
        let fragment_descriptor_len =
            lf_group_count
                .checked_mul(2)
                .ok_or(EncodeError::InvalidConfiguration(
                    "VarDCT fragment descriptor size overflow",
                ))?;
        let strategy_offset = align_words(
            fragment_descriptor_offset
                .checked_add(fragment_descriptor_len)
                .ok_or(EncodeError::InvalidConfiguration(
                    "VarDCT fragment descriptor section overflow",
                ))?,
        )?;
        let dc_offset = align_words(strategy_offset.checked_add(strategy_len).ok_or(
            EncodeError::InvalidConfiguration("VarDCT strategy section overflow"),
        )?)?;
        let token_offset = align_words(dc_offset.checked_add(dc_len).ok_or(
            EncodeError::InvalidConfiguration("VarDCT DC section overflow"),
        )?)?;
        let extra_offset = align_words(token_offset.checked_add(dc_len).ok_or(
            EncodeError::InvalidConfiguration("VarDCT token section overflow"),
        )?)?;
        let fragment_offset = align_words(extra_offset.checked_add(dc_len).ok_or(
            EncodeError::InvalidConfiguration("VarDCT extra-bit section overflow"),
        )?)?;
        let artifact_words =
            align_words(fragment_offset.checked_add(fragment_word_capacity).ok_or(
                EncodeError::InvalidConfiguration("VarDCT artifact size overflow"),
            )?)?;
        Ok(Self {
            fragment_descriptor_offset,
            fragment_descriptor_len,
            strategy_offset,
            strategy_len,
            dc_offset,
            dc_len,
            token_offset,
            token_len: dc_len,
            extra_offset,
            extra_len: dc_len,
            fragment_offset,
            fragment_word_capacity,
            fragment_max_bits,
            ac_descriptor_offset: 0,
            ac_descriptor_len: 0,
            ac_fragment_offset: 0,
            ac_words_per_block: 0,
            ac_fragment_words: 0,
            artifact_words,
        })
    }

    pub(super) fn for_tiled_grid(
        frame: VarDctFrameLayout,
        code: &VarDctPrefixCode,
        hf_entropy: &super::entropy::HfEntropyPlan,
    ) -> Result<Self, EncodeError> {
        let layout = Self::for_block_grid(
            frame.blocks_x,
            frame.blocks_y,
            frame.lf_group_count()?,
            code,
        )?;
        layout.with_ac(63, frame.blocks_x * frame.blocks_y, hf_entropy)
    }

    fn with_ac(
        mut self,
        maximum_nonzero: u32,
        transforms: u32,
        hf_entropy: &super::entropy::HfEntropyPlan,
    ) -> Result<Self, EncodeError> {
        // Each GPU transform owns a word-aligned fragment. The entropy
        // model maps all contexts to one distribution, so these independently
        // packed Y/X/B fragments can be concatenated in AC-group raster order.
        let entries = hf_entropy.gpu_entries();
        let token_bits: [u32; UINT_SYMBOLS] =
            std::array::from_fn(|token| entries[token].bit_len + token.saturating_sub(1) as u32);
        let max_count_token = 32 - maximum_nonzero.leading_zeros();
        let max_count_bits = token_bits[..=max_count_token as usize]
            .iter()
            .copied()
            .max()
            .ok_or(EncodeError::InvalidConfiguration("empty HF count alphabet"))?;
        let max_coefficient_bits =
            token_bits
                .iter()
                .copied()
                .max()
                .ok_or(EncodeError::InvalidConfiguration(
                    "empty HF coefficient alphabet",
                ))?;
        let block_bits = max_coefficient_bits
            .checked_mul(maximum_nonzero)
            .and_then(|bits| bits.checked_add(max_count_bits))
            .and_then(|bits| bits.checked_mul(3))
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT AC block capacity overflow",
            ))?;
        self.ac_words_per_block = block_bits.div_ceil(32);
        self.ac_descriptor_offset = self.artifact_words;
        self.ac_descriptor_len = transforms;
        self.ac_fragment_offset = align_words(
            self.ac_descriptor_offset
                .checked_add(self.ac_descriptor_len)
                .ok_or(EncodeError::InvalidConfiguration(
                    "VarDCT AC descriptor overflow",
                ))?,
        )?;
        self.ac_fragment_words = self.ac_words_per_block.checked_mul(transforms).ok_or(
            EncodeError::InvalidConfiguration("VarDCT AC fragment capacity overflow"),
        )?;
        self.artifact_words = align_words(
            self.ac_fragment_offset
                .checked_add(self.ac_fragment_words)
                .ok_or(EncodeError::InvalidConfiguration(
                    "VarDCT AC artifact overflow",
                ))?,
        )?;
        Ok(self)
    }

    pub(super) const fn artifact_bytes(self) -> u64 {
        self.artifact_words as u64 * std::mem::size_of::<u32>() as u64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VarDctTopology {
    SingleTransform(VarDctStrategy),
    TiledDct8,
    StrategyMap,
}

impl VarDctTopology {
    pub(super) const fn strategy_id(self) -> u32 {
        match self {
            Self::SingleTransform(strategy) => strategy.codestream_id() as u32,
            Self::TiledDct8 => VarDctStrategy::Dct8.codestream_id() as u32,
            Self::StrategyMap => u32::MAX,
        }
    }

    pub(super) const fn artifact_id(self) -> u32 {
        match self {
            Self::SingleTransform(_) => SINGLE_TRANSFORM_TOPOLOGY,
            Self::TiledDct8 => TILED_DCT8_TOPOLOGY,
            Self::StrategyMap => 2,
        }
    }

    pub(super) const fn kernel_layout(self) -> VarDctKernelLayout {
        match self {
            Self::SingleTransform(_) => VarDctKernelLayout::SingleTransform,
            Self::TiledDct8 => VarDctKernelLayout::TiledDct8,
            Self::StrategyMap => VarDctKernelLayout::StrategyMap,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct VarDctFrameLayout {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) blocks_x: u32,
    pub(super) blocks_y: u32,
    pub(super) ac_groups_x: u32,
    pub(super) ac_groups_y: u32,
    pub(super) lf_groups_x: u32,
    pub(super) lf_groups_y: u32,
    pub(super) topology: VarDctTopology,
}

impl VarDctFrameLayout {
    pub(super) fn single(strategy: VarDctStrategy) -> Self {
        let Extent2d { width, height } = strategy.pixel_extent();
        let Extent2d {
            width: blocks_x,
            height: blocks_y,
        } = strategy.lf_extent();
        Self {
            width,
            height,
            blocks_x,
            blocks_y,
            ac_groups_x: 1,
            ac_groups_y: 1,
            lf_groups_x: 1,
            lf_groups_y: 1,
            topology: VarDctTopology::SingleTransform(strategy),
        }
    }

    pub(super) fn tiled_dct8(width: u32, height: u32) -> Result<Self, EncodeError> {
        let grid = TiledVarDctGrid::new(width, height)?;
        Ok(Self {
            width,
            height,
            blocks_x: grid.block_columns,
            blocks_y: grid.block_rows,
            ac_groups_x: grid.ac_group_columns,
            ac_groups_y: grid.ac_group_rows,
            lf_groups_x: grid.lf_group_columns,
            lf_groups_y: grid.lf_group_rows,
            topology: VarDctTopology::TiledDct8,
        })
    }

    pub(super) fn ac_group_count(self) -> Result<u32, EncodeError> {
        self.ac_groups_x
            .checked_mul(self.ac_groups_y)
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT AC group count overflow",
            ))
    }

    pub(super) fn lf_group_count(self) -> Result<u32, EncodeError> {
        self.lf_groups_x
            .checked_mul(self.lf_groups_y)
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT LF group count overflow",
            ))
    }

    pub(super) fn lf_group_blocks(self, group: u32) -> Result<LfGroupBlocks, EncodeError> {
        let count = self.lf_group_count()?;
        if group >= count {
            return Err(EncodeError::InvalidConfiguration(
                "VarDCT LF group index is out of range",
            ));
        }
        let group_x = group % self.lf_groups_x;
        let group_y = group / self.lf_groups_x;
        let origin_x = group_x * (LF_GROUP_DIM_PIXELS / 8);
        let origin_y = group_y * (LF_GROUP_DIM_PIXELS / 8);
        let width = (self.blocks_x - origin_x).min(LF_GROUP_DIM_PIXELS / 8);
        let height = (self.blocks_y - origin_y).min(LF_GROUP_DIM_PIXELS / 8);
        Ok(LfGroupBlocks {
            origin_x,
            origin_y,
            width,
            height,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LfGroupBlocks {
    pub(super) origin_x: u32,
    pub(super) origin_y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl LfGroupBlocks {
    pub(super) fn block_count(self) -> Result<u32, EncodeError> {
        self.width
            .checked_mul(self.height)
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT LF group block count overflow",
            ))
    }
}

pub(super) fn align_words(words: u32) -> Result<u32, EncodeError> {
    let adjustment = SECTION_ALIGNMENT_WORDS - 1;
    words
        .checked_add(adjustment)
        .map(|value| value / SECTION_ALIGNMENT_WORDS * SECTION_ALIGNMENT_WORDS)
        .ok_or(EncodeError::InvalidConfiguration(
            "VarDCT artifact alignment overflow",
        ))
}

#[derive(Clone, Copy)]
pub(super) struct VarDctArtifactData<'a> {
    pub(super) transform_plan: Option<&'a super::strategy_map::TransformPlan>,
    pub(super) strategy: u32,
    pub(super) dc_fragment_words: &'a [u32],
    pub(super) dc_fragment_bit_len: u32,
    pub(super) dc_fragment_descriptors: &'a [DcFragmentDescriptor],
    pub(super) ac: super::ac::AcFragments<'a>,
}

impl VarDctArtifactData<'_> {
    pub(super) const fn has_ac_payload(self) -> bool {
        !matches!(self.ac, super::ac::AcFragments::Empty)
    }

    pub(super) fn dc_fragment_descriptor(
        self,
        group: u32,
    ) -> Result<DcFragmentDescriptor, EncodeError> {
        if self.dc_fragment_descriptors.is_empty() && group == 0 {
            return Ok(DcFragmentDescriptor {
                bit_offset: 0,
                bit_len: self.dc_fragment_bit_len,
            });
        }
        self.dc_fragment_descriptors
            .get(usize::try_from(group).map_err(|_| {
                EncodeError::InvalidConfiguration("VarDCT LF group index does not fit usize")
            })?)
            .copied()
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT LF group fragment descriptor is missing",
            ))
    }
}
