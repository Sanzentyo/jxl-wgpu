//! VarDCT contracts, ABI records, and frame geometry.

use jxl_gpu_bitstream::FiniteF16;
use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, PixelFormat,
    PlaneFormat, PlaneSampling, SampleKind, Swizzle,
};

use crate::prefix::{PrefixCode, RAW_SYMBOLS};
use crate::{EncodeError, UnsupportedFeature};

pub(super) const GLOBAL_SCALE: u32 = 8_813;
pub(super) const QUANT_LF: u32 = 10;
pub(super) const HF_MUL: i32 = 6;
pub(super) const MAX_BLOCKS: usize = 16;
pub(super) const MAX_COEFFICIENTS: usize = 32 * 32;
pub(super) const MAX_DC_SAMPLES: usize = 3 * MAX_BLOCKS;
pub(super) const MAX_DC_FRAGMENT_WORDS: usize = 64;
pub(super) const MAX_AC_FRAGMENT_WORDS: usize = 256;
pub(super) const DCT8_COEFFICIENTS: usize = 8 * 8;
pub(super) const MAX_HF_QUANTIZED_MAGNITUDE: i32 = 131_071;
pub(super) const HF_QUANTIZATION: [f32; 3] = [1.25, 1.0, 1.0];
pub(super) const DCT8_NATURAL_ORDER: [usize; DCT8_COEFFICIENTS] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

pub(super) const AC_GROUP_DIM_PIXELS: u32 = 256;
pub(super) const LF_GROUP_DIM_PIXELS: u32 = 2_048;
pub(super) const SCALABLE_HEADER_WORDS: u32 = 64;
pub(super) const SCALABLE_SECTION_ALIGNMENT_WORDS: u32 = 64;
pub(super) const SCALABLE_ARTIFACT_READY: u32 = 0x5644_4354;
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
        if grid.ac_group_count()? == 1 {
            return Err(UnsupportedFeature::TiledVarDctSingleAcGroup {
                width,
                height,
                group_dimension: Self::AC_GROUP_DIMENSION,
            }
            .into());
        }
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

    /// TOC entries for the deliberately non-fused tiled profile: DC global,
    /// every DC group, AC global, then one pass packet per AC group.
    pub fn toc_entries(self) -> Result<u32, EncodeError> {
        self.lf_group_count()?
            .checked_add(self.ac_group_count()?)
            .and_then(|groups| groups.checked_add(2))
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT TOC entry count overflow",
            ))
    }
}

/// Typed JPEG XL VarDCT strategy identifier.
///
/// The enum covers the complete standard strategy alphabet. Use
/// [`Self::EXECUTABLE`] to enumerate the strategies implemented by the
/// current GPU kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VarDctStrategy {
    #[default]
    Dct8 = 0,
    Hornuss,
    Dct2x2,
    Dct4x4,
    Dct16x16,
    Dct32x32,
    Dct16x8,
    Dct8x16,
    Dct32x8,
    Dct8x32,
    Dct32x16,
    Dct16x32,
    Dct4x8,
    Dct8x4,
    Afv0,
    Afv1,
    Afv2,
    Afv3,
    Dct64x64,
    Dct64x32,
    Dct32x64,
    Dct128x128,
    Dct128x64,
    Dct64x128,
    Dct256x256,
    Dct256x128,
    Dct128x256,
}

impl VarDctStrategy {
    /// Every JPEG XL VarDCT strategy in its standard codestream order.
    pub const ALL: [Self; 27] = [
        Self::Dct8,
        Self::Hornuss,
        Self::Dct2x2,
        Self::Dct4x4,
        Self::Dct16x16,
        Self::Dct32x32,
        Self::Dct16x8,
        Self::Dct8x16,
        Self::Dct32x8,
        Self::Dct8x32,
        Self::Dct32x16,
        Self::Dct16x32,
        Self::Dct4x8,
        Self::Dct8x4,
        Self::Afv0,
        Self::Afv1,
        Self::Afv2,
        Self::Afv3,
        Self::Dct64x64,
        Self::Dct64x32,
        Self::Dct32x64,
        Self::Dct128x128,
        Self::Dct128x64,
        Self::Dct64x128,
        Self::Dct256x256,
        Self::Dct256x128,
        Self::Dct128x256,
    ];

    /// Strategies implemented end-to-end by this encoder.
    pub const EXECUTABLE: [Self; 27] = Self::ALL;

    #[must_use]
    pub const fn codestream_id(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub const fn block_extent(self) -> (u16, u16) {
        use VarDctStrategy::*;
        match self {
            Dct8 | Hornuss | Dct2x2 | Dct4x4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3 => {
                (8, 8)
            }
            Dct16x16 => (16, 16),
            Dct32x32 => (32, 32),
            Dct16x8 => (8, 16),
            Dct8x16 => (16, 8),
            Dct32x8 => (8, 32),
            Dct8x32 => (32, 8),
            Dct32x16 => (16, 32),
            Dct16x32 => (32, 16),
            Dct64x64 => (64, 64),
            Dct64x32 => (32, 64),
            Dct32x64 => (64, 32),
            Dct128x128 => (128, 128),
            Dct128x64 => (64, 128),
            Dct64x128 => (128, 64),
            Dct256x256 => (256, 256),
            Dct256x128 => (128, 256),
            Dct128x256 => (256, 128),
        }
    }

    /// Whether this strategy has a GPU transform and standard emitter in this
    /// frontend.
    #[must_use]
    pub const fn is_executable(self) -> bool {
        true
    }

    pub(super) const fn uses_scalable_kernel(self) -> bool {
        !matches!(
            self,
            Self::Dct8
                | Self::Hornuss
                | Self::Dct2x2
                | Self::Dct4x4
                | Self::Dct16x8
                | Self::Dct8x16
                | Self::Dct16x16
                | Self::Dct32x8
                | Self::Dct8x32
                | Self::Dct32x32
                | Self::Dct32x16
                | Self::Dct16x32
                | Self::Dct4x8
                | Self::Dct8x4
                | Self::Afv0
                | Self::Afv1
                | Self::Afv2
                | Self::Afv3
        )
    }

    pub(super) const fn block_grid(self) -> (u32, u32) {
        let (width, height) = self.block_extent();
        (width as u32 / 8, height as u32 / 8)
    }
}

/// GPU artifact implementation selected for a VarDCT memory plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarDctKernelLayout {
    /// Fixed 26.25 KiB coefficient and entropy artifact used through 32x32.
    Bounded,
    /// Runtime-sized artifact and 8x8-block reduction used above 32x32.
    Scalable,
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
    pub owned_bytes_per_job: u64,
    pub addressed_bytes_per_job: u64,
}

impl VarDctMemoryPlan {
    pub(super) const fn fixed(source_binding_bytes: u64) -> Self {
        let parameter_storage_bytes = std::mem::size_of::<VarDctKernelParams>() as u64;
        let artifact_storage_bytes = std::mem::size_of::<VarDctKernelArtifact>() as u64;
        let readback_bytes = artifact_storage_bytes;
        let owned_bytes_per_job = parameter_storage_bytes + artifact_storage_bytes + readback_bytes;
        Self {
            kernel_layout: VarDctKernelLayout::Bounded,
            source_binding_bytes,
            parameter_storage_bytes,
            artifact_storage_bytes,
            readback_bytes,
            owned_bytes_per_job,
            addressed_bytes_per_job: source_binding_bytes + owned_bytes_per_job,
        }
    }

    pub(super) const fn scalable(
        source_binding_bytes: u64,
        artifact_storage_bytes: u64,
        kernel_layout: VarDctKernelLayout,
    ) -> Self {
        let parameter_storage_bytes = std::mem::size_of::<ScalableVarDctKernelParams>() as u64;
        let readback_bytes = artifact_storage_bytes;
        let owned_bytes_per_job = parameter_storage_bytes + artifact_storage_bytes + readback_bytes;
        Self {
            kernel_layout,
            source_binding_bytes,
            parameter_storage_bytes,
            artifact_storage_bytes,
            readback_bytes,
            owned_bytes_per_job,
            addressed_bytes_per_job: source_binding_bytes + owned_bytes_per_job,
        }
    }
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
    pub(super) dc_prefix: [GpuPrefixEntry; RAW_SYMBOLS],
    pub(super) hf_prefix: [GpuPrefixEntry; RAW_SYMBOLS],
    pub(super) lf_quantization: [f32; 3],
    pub(super) lf_correlation: [f32; 2],
    pub(super) hf_correlation: [f32; 2],
    pub(super) hf_quantization: [f32; 3],
    pub(super) padding: [u32; 33],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GpuPrefixEntry {
    pub(super) bits: u32,
    pub(super) bit_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct VarDctKernelArtifact {
    pub(super) strategy_map: [u32; MAX_BLOCKS],
    pub(super) quantized_dc_yxb: [i32; MAX_DC_SAMPLES],
    pub(super) dc_raw_tokens: [u32; MAX_DC_SAMPLES],
    pub(super) dc_extra_bits: [u32; MAX_DC_SAMPLES],
    pub(super) dc_fragment_words: [u32; MAX_DC_FRAGMENT_WORDS],
    pub(super) dc_fragment_bit_len: u32,
    pub(super) dc_sample_count: u32,
    pub(super) block_count: u32,
    pub(super) strategy: u32,
    pub(super) raw_histogram: [u32; RAW_SYMBOLS],
    pub(super) dc_padding: [u32; 9],
    pub(super) ac_fragment_words: [u32; MAX_AC_FRAGMENT_WORDS],
    pub(super) ac_fragment_bit_len: u32,
    pub(super) ac_token_count: u32,
    pub(super) ac_histogram: [u32; RAW_SYMBOLS],
    pub(super) ac_padding: [u32; 43],
    pub(super) forward_xyb_bits: [u32; 3 * MAX_COEFFICIENTS],
    pub(super) quantized_xyb: [i32; 3 * MAX_COEFFICIENTS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ScalableVarDctKernelParams {
    pub(super) row_stride: u32,
    pub(super) byte_offset: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) blocks_x: u32,
    pub(super) blocks_y: u32,
    pub(super) strategy: u32,
    pub(super) global_scale: u32,
    pub(super) quant_lf: u32,
    pub(super) raw_prefix: [GpuPrefixEntry; RAW_SYMBOLS],
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
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ScalableVarDctArtifactHeader {
    pub(super) status: u32,
    pub(super) block_count: u32,
    pub(super) dc_sample_count: u32,
    pub(super) strategy: u32,
    pub(super) ac_all_zero: u32,
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
    pub(super) raw_histogram: [u32; RAW_SYMBOLS],
    pub(super) fragment_descriptor_offset: u32,
    pub(super) fragment_descriptor_len: u32,
    pub(super) lf_groups_x: u32,
    pub(super) lf_groups_y: u32,
    pub(super) lf_group_count: u32,
    pub(super) padding: [u32; 18],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ScalableDcFragmentDescriptor {
    pub(super) bit_offset: u32,
    pub(super) bit_len: u32,
}

const _: () = {
    assert!(std::mem::size_of::<GpuPrefixEntry>() == 8);
    assert!(std::mem::align_of::<GpuPrefixEntry>() == 4);
    assert!(std::mem::size_of::<VarDctKernelParams>() == 512);
    assert!(std::mem::align_of::<VarDctKernelParams>() == 4);
    assert!(std::mem::size_of::<VarDctKernelArtifact>() == 26_880);
    assert!(std::mem::align_of::<VarDctKernelArtifact>() == 4);
    assert!(std::mem::size_of::<ScalableVarDctKernelParams>() == 256);
    assert!(std::mem::align_of::<ScalableVarDctKernelParams>() == 4);
    assert!(std::mem::size_of::<ScalableVarDctArtifactHeader>() == 256);
    assert!(std::mem::align_of::<ScalableVarDctArtifactHeader>() == 4);
    assert!(std::mem::size_of::<ScalableDcFragmentDescriptor>() == 8);
    assert!(std::mem::align_of::<ScalableDcFragmentDescriptor>() == 4);
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ScalableArtifactLayout {
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
    pub(super) artifact_words: u32,
}

impl ScalableArtifactLayout {
    pub(super) fn new(strategy: VarDctStrategy, code: &PrefixCode) -> Result<Self, EncodeError> {
        let (blocks_x, blocks_y) = strategy.block_grid();
        Self::for_block_grid(blocks_x, blocks_y, 1, code)
    }

    pub(super) fn for_block_grid(
        blocks_x: u32,
        blocks_y: u32,
        lf_group_count: u32,
        code: &PrefixCode,
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

        let fragment_descriptor_offset = SCALABLE_HEADER_WORDS;
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
            artifact_words,
        })
    }

    pub(super) const fn artifact_bytes(self) -> u64 {
        self.artifact_words as u64 * std::mem::size_of::<u32>() as u64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VarDctTopology {
    SingleTransform(VarDctStrategy),
    TiledDct8,
}

impl VarDctTopology {
    pub(super) const fn strategy(self) -> VarDctStrategy {
        match self {
            Self::SingleTransform(strategy) => strategy,
            Self::TiledDct8 => VarDctStrategy::Dct8,
        }
    }

    pub(super) const fn artifact_id(self) -> u32 {
        match self {
            Self::SingleTransform(_) => SINGLE_TRANSFORM_TOPOLOGY,
            Self::TiledDct8 => TILED_DCT8_TOPOLOGY,
        }
    }

    pub(super) const fn uses_scalable_kernel(self) -> bool {
        match self {
            Self::SingleTransform(strategy) => strategy.uses_scalable_kernel(),
            Self::TiledDct8 => true,
        }
    }

    pub(super) const fn kernel_layout(self) -> VarDctKernelLayout {
        match self {
            Self::SingleTransform(_) => VarDctKernelLayout::Scalable,
            Self::TiledDct8 => VarDctKernelLayout::TiledDct8,
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
        let (width, height) = strategy.block_extent();
        let (blocks_x, blocks_y) = strategy.block_grid();
        Self {
            width: u32::from(width),
            height: u32::from(height),
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
        if matches!(self.topology, VarDctTopology::SingleTransform(_)) {
            return Ok(LfGroupBlocks {
                origin_x: 0,
                origin_y: 0,
                width: self.blocks_x,
                height: self.blocks_y,
                first_block_count: 1,
            });
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
            first_block_count: width * height,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LfGroupBlocks {
    pub(super) origin_x: u32,
    pub(super) origin_y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) first_block_count: u32,
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
    let adjustment = SCALABLE_SECTION_ALIGNMENT_WORDS - 1;
    words
        .checked_add(adjustment)
        .map(|value| value / SCALABLE_SECTION_ALIGNMENT_WORDS * SCALABLE_SECTION_ALIGNMENT_WORDS)
        .ok_or(EncodeError::InvalidConfiguration(
            "VarDCT artifact alignment overflow",
        ))
}

#[derive(Clone, Copy)]
pub(super) struct VarDctArtifactData<'a> {
    pub(super) strategy: u32,
    pub(super) dc_fragment_words: &'a [u32],
    pub(super) dc_fragment_bit_len: u32,
    pub(super) dc_fragment_descriptors: &'a [ScalableDcFragmentDescriptor],
    pub(super) ac_fragment_words: &'a [u32],
    pub(super) ac_fragment_bit_len: u32,
}

impl VarDctArtifactData<'_> {
    pub(super) const fn has_ac_payload(self) -> bool {
        self.ac_fragment_bit_len != 0
    }

    pub(super) fn dc_fragment_descriptor(
        self,
        group: u32,
    ) -> Result<ScalableDcFragmentDescriptor, EncodeError> {
        if self.dc_fragment_descriptors.is_empty() && group == 0 {
            return Ok(ScalableDcFragmentDescriptor {
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
