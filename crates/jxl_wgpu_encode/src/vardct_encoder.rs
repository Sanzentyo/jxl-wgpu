//! Standard VarDCT still-image encoder frontend.
//!
//! The frontend encodes one strategy whose footprint is also the
//! image extent. Its control-plane syntax is kept separate from the lossless
//! Modular encoder so neither profile becomes a compatibility layer for the
//! other.

use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use jxl_gpu_bitstream::{BitWriter, PrefixCodeEntry};
use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, PixelFormat,
    PlaneFormat, PlaneSampling, SampleKind, Swizzle,
};
use jxl_wgpu::{KernelVariant, MemoryPermit};

use crate::prefix::{LZ77_SYMBOLS, PrefixCode, RAW_SYMBOLS};
use crate::{
    AnimationHeader, BackendError, BitFragment, BufferImageSource, Determinism, EncodeError,
    EncodeProfile, EncoderCapabilities, FrameEncodeRequest, FrameGroupLayout, FrameIndex,
    FrameOptions, FramePacketSet, FrameSubmission, GpuEncodeBackend, GpuEncodeJob, GpuEncoder,
    GpuFrameArtifacts, GpuFrameSource, GroupPacket, GroupPacketKind, KernelStage,
    PerceptualDistance, ProfileCapability, ProgressivePlan, UnsupportedFeature, WgpuContext,
    assemble_frame,
};

const GLOBAL_SCALE: u32 = 8_813;
const QUANT_LF: u32 = 10;
const HF_MUL: i32 = 6;
const MAX_BLOCKS: usize = 16;
const MAX_COEFFICIENTS: usize = 32 * 32;
const MAX_DC_SAMPLES: usize = 3 * MAX_BLOCKS;
const MAX_DC_FRAGMENT_WORDS: usize = 64;
const SHADER: &str = include_str!("vardct_encoder.wgsl");
const LARGE_SHADER: &str = include_str!("vardct_large_encoder.wgsl");
const PROFILE_DISTANCE: f32 = 25.0;
const BOUNDED_KERNEL_KEY: &str = "vardct_encode_bounded";
const SCALABLE_QUANTIZE_KERNEL_KEY: &str = "vardct_encode_quantize";
const BOUNDED_WORKGROUP_STORAGE_BYTES: u32 = 1_024 * 16;
const LARGE_WORKGROUP_STORAGE_BYTES: u32 = 64 * 16;
const AC_GROUP_DIM_PIXELS: u32 = 256;
const LF_GROUP_DIM_PIXELS: u32 = 2_048;
const SCALABLE_HEADER_WORDS: u32 = 64;
const SCALABLE_SECTION_ALIGNMENT_WORDS: u32 = 64;
const SCALABLE_ARTIFACT_READY: u32 = 0x5644_4354;
const SINGLE_TRANSFORM_TOPOLOGY: u32 = 0;
const TILED_DCT8_TOPOLOGY: u32 = 1;

/// Presentation/source color contract of the standard VarDCT frontend.
///
/// Samples are interleaved nonlinear sRGB bytes with a D65 white point. The
/// GPU applies the IEC sRGB transfer function and JPEG XL's default opsin
/// absorbance matrix before the forward transform.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VarDctColorEncoding {
    #[default]
    SrgbD65,
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
}

impl TiledVarDctGrid {
    /// Pixel dimension of one standard AC/pass group.
    pub const AC_GROUP_DIMENSION: u32 = AC_GROUP_DIM_PIXELS;
    /// Current one-LF-group profile bound on each source axis.
    pub const MAX_DIMENSION: u32 = LF_GROUP_DIM_PIXELS;

    /// Derives the exact block and AC-group grid without allocating GPU data.
    pub fn new(width: u32, height: u32) -> Result<Self, EncodeError> {
        if width == 0 || height == 0 {
            return Err(EncodeError::InvalidSource(
                "tiled VarDCT dimensions must be nonzero",
            ));
        }
        if width > Self::MAX_DIMENSION || height > Self::MAX_DIMENSION {
            return Err(UnsupportedFeature::TiledVarDctLfGroups {
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

    /// TOC entries for the deliberately non-fused tiled profile: DC global,
    /// one DC group, AC global, then one pass packet per AC group.
    pub fn toc_entries(self) -> Result<u32, EncodeError> {
        self.ac_group_count()?
            .checked_add(3)
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

    const fn uses_scalable_kernel(self) -> bool {
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

    const fn block_grid(self) -> (u32, u32) {
        let (width, height) = self.block_extent();
        (width as u32 / 8, height as u32 / 8)
    }
}

/// GPU artifact implementation selected for a VarDCT memory plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarDctKernelLayout {
    /// Fixed 25 KiB diagnostic artifact used through 32x32.
    Bounded,
    /// Runtime-sized artifact and 8x8-block reduction used above 32x32.
    Scalable,
    /// Runtime-sized artifact where every 8x8 block is an independent DCT8
    /// transform and the frame may contain multiple 256-pixel AC groups.
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
    const fn fixed(source_binding_bytes: u64) -> Self {
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

    const fn scalable(
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
struct VarDctKernelParams {
    row_stride: u32,
    byte_offset: u32,
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    strategy: u32,
    global_scale: u32,
    quant_lf: u32,
    raw_prefix: [GpuPrefixEntry; RAW_SYMBOLS],
    padding: [u32; 17],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPrefixEntry {
    bits: u32,
    bit_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VarDctKernelArtifact {
    strategy_map: [u32; MAX_BLOCKS],
    quantized_dc_yxb: [i32; MAX_DC_SAMPLES],
    dc_raw_tokens: [u32; MAX_DC_SAMPLES],
    dc_extra_bits: [u32; MAX_DC_SAMPLES],
    dc_fragment_words: [u32; MAX_DC_FRAGMENT_WORDS],
    dc_fragment_bit_len: u32,
    dc_sample_count: u32,
    block_count: u32,
    strategy: u32,
    raw_histogram: [u32; RAW_SYMBOLS],
    padding: [u32; 9],
    forward_xyb_bits: [u32; 3 * MAX_COEFFICIENTS],
    quantized_xyb: [i32; 3 * MAX_COEFFICIENTS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalableVarDctKernelParams {
    row_stride: u32,
    byte_offset: u32,
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    strategy: u32,
    global_scale: u32,
    quant_lf: u32,
    raw_prefix: [GpuPrefixEntry; RAW_SYMBOLS],
    strategy_offset: u32,
    dc_offset: u32,
    token_offset: u32,
    extra_offset: u32,
    fragment_offset: u32,
    fragment_word_capacity: u32,
    artifact_words: u32,
    topology: u32,
    padding: [u32; 9],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalableVarDctArtifactHeader {
    status: u32,
    block_count: u32,
    dc_sample_count: u32,
    strategy: u32,
    ac_all_zero: u32,
    strategy_offset: u32,
    strategy_len: u32,
    dc_offset: u32,
    dc_len: u32,
    token_offset: u32,
    token_len: u32,
    extra_offset: u32,
    extra_len: u32,
    fragment_offset: u32,
    fragment_word_capacity: u32,
    dc_fragment_bit_len: u32,
    artifact_words: u32,
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    topology: u32,
    raw_histogram: [u32; RAW_SYMBOLS],
    padding: [u32; 23],
}

const _: () = {
    assert!(std::mem::size_of::<GpuPrefixEntry>() == 8);
    assert!(std::mem::align_of::<GpuPrefixEntry>() == 4);
    assert!(std::mem::size_of::<VarDctKernelParams>() == 256);
    assert!(std::mem::align_of::<VarDctKernelParams>() == 4);
    assert!(std::mem::size_of::<VarDctKernelArtifact>() == 25_600);
    assert!(std::mem::align_of::<VarDctKernelArtifact>() == 4);
    assert!(std::mem::size_of::<ScalableVarDctKernelParams>() == 256);
    assert!(std::mem::align_of::<ScalableVarDctKernelParams>() == 4);
    assert!(std::mem::size_of::<ScalableVarDctArtifactHeader>() == 256);
    assert!(std::mem::align_of::<ScalableVarDctArtifactHeader>() == 4);
};

fn fixed_prefix_code() -> Result<PrefixCode, EncodeError> {
    PrefixCode::from_aggregated_counts(&[0; RAW_SYMBOLS], &[0; LZ77_SYMBOLS], RAW_SYMBOLS - 1, true)
}

fn prefix_entries(code: &PrefixCode) -> [GpuPrefixEntry; RAW_SYMBOLS] {
    code.raw_entries()
        .map(|PrefixCodeEntry { bit_len, bits }| GpuPrefixEntry {
            bits: u32::from(bits),
            bit_len: u32::from(bit_len),
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScalableArtifactLayout {
    strategy_offset: u32,
    strategy_len: u32,
    dc_offset: u32,
    dc_len: u32,
    token_offset: u32,
    token_len: u32,
    extra_offset: u32,
    extra_len: u32,
    fragment_offset: u32,
    fragment_word_capacity: u32,
    fragment_max_bits: u32,
    artifact_words: u32,
}

impl ScalableArtifactLayout {
    fn new(strategy: VarDctStrategy, code: &PrefixCode) -> Result<Self, EncodeError> {
        let (blocks_x, blocks_y) = strategy.block_grid();
        Self::for_block_grid(blocks_x, blocks_y, code)
    }

    fn for_block_grid(
        blocks_x: u32,
        blocks_y: u32,
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

        let strategy_offset = SCALABLE_HEADER_WORDS;
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

    const fn artifact_bytes(self) -> u64 {
        self.artifact_words as u64 * std::mem::size_of::<u32>() as u64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VarDctTopology {
    SingleTransform(VarDctStrategy),
    TiledDct8,
}

impl VarDctTopology {
    const fn strategy(self) -> VarDctStrategy {
        match self {
            Self::SingleTransform(strategy) => strategy,
            Self::TiledDct8 => VarDctStrategy::Dct8,
        }
    }

    const fn artifact_id(self) -> u32 {
        match self {
            Self::SingleTransform(_) => SINGLE_TRANSFORM_TOPOLOGY,
            Self::TiledDct8 => TILED_DCT8_TOPOLOGY,
        }
    }

    const fn uses_scalable_kernel(self) -> bool {
        match self {
            Self::SingleTransform(strategy) => strategy.uses_scalable_kernel(),
            Self::TiledDct8 => true,
        }
    }

    const fn kernel_layout(self) -> VarDctKernelLayout {
        match self {
            Self::SingleTransform(_) => VarDctKernelLayout::Scalable,
            Self::TiledDct8 => VarDctKernelLayout::TiledDct8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VarDctFrameLayout {
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    ac_groups_x: u32,
    ac_groups_y: u32,
    topology: VarDctTopology,
}

impl VarDctFrameLayout {
    fn single(strategy: VarDctStrategy) -> Self {
        let (width, height) = strategy.block_extent();
        let (blocks_x, blocks_y) = strategy.block_grid();
        Self {
            width: u32::from(width),
            height: u32::from(height),
            blocks_x,
            blocks_y,
            ac_groups_x: 1,
            ac_groups_y: 1,
            topology: VarDctTopology::SingleTransform(strategy),
        }
    }

    fn tiled_dct8(width: u32, height: u32) -> Result<Self, EncodeError> {
        let grid = TiledVarDctGrid::new(width, height)?;
        Ok(Self {
            width,
            height,
            blocks_x: grid.block_columns,
            blocks_y: grid.block_rows,
            ac_groups_x: grid.ac_group_columns,
            ac_groups_y: grid.ac_group_rows,
            topology: VarDctTopology::TiledDct8,
        })
    }

    fn block_count(self) -> Result<u32, EncodeError> {
        self.blocks_x
            .checked_mul(self.blocks_y)
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT block count overflow",
            ))
    }

    fn ac_group_count(self) -> Result<u32, EncodeError> {
        self.ac_groups_x
            .checked_mul(self.ac_groups_y)
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT AC group count overflow",
            ))
    }

    const fn first_block_count(self) -> u32 {
        match self.topology {
            VarDctTopology::SingleTransform(_) => 1,
            VarDctTopology::TiledDct8 => self.blocks_x * self.blocks_y,
        }
    }
}

fn align_words(words: u32) -> Result<u32, EncodeError> {
    let adjustment = SCALABLE_SECTION_ALIGNMENT_WORDS - 1;
    words
        .checked_add(adjustment)
        .map(|value| value / SCALABLE_SECTION_ALIGNMENT_WORDS * SCALABLE_SECTION_ALIGNMENT_WORDS)
        .ok_or(EncodeError::InvalidConfiguration(
            "VarDCT artifact alignment overflow",
        ))
}

fn write_size(output: &mut BitWriter, size: u32, ratio: bool) -> Result<(), EncodeError> {
    if !(1..(1 << 30)).contains(&size) {
        return Err(EncodeError::InvalidConfiguration(
            "VarDCT dimensions must be in 1..2^30",
        ));
    }
    let value = size - 1;
    let (selector, bits) = if value < 1 << 9 {
        (0, 9)
    } else if value < 1 << 13 {
        (1, 13)
    } else if value < 1 << 18 {
        (2, 18)
    } else {
        (3, 30)
    };
    output.write_bits(selector, 2)?;
    output.write_bits(u64::from(value), bits)?;
    if ratio {
        output.write_bits(0, 3)?;
    }
    Ok(())
}

fn image_header(width: u32, height: u32) -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0x0aff, 16)?;
    output.write_bits(0, 1)?; // dimensions are not encoded as multiples of eight
    write_size(&mut output, height, true)?;
    write_size(&mut output, width, false)?;
    output.write_bits(1, 1)?; // all-default image metadata: 8-bit, XYB, sRGB presentation
    output.write_bits(1, 1)?; // default opsin inverse matrix and upsampling weights
    output.align_to_byte()?;
    Ok(BitFragment::byte_aligned(output.into_bytes())?)
}

fn frame_header() -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0, 1)?; // non-default so restoration can be disabled
    output.write_bits(0, 2)?; // regular frame
    output.write_bits(0, 1)?; // VarDCT
    output.write_bits(0, 2)?; // no frame flags
    output.write_bits(0, 2)?; // no upsampling
    output.write_bits(3, 3)?; // default X quant-matrix scale
    output.write_bits(2, 3)?; // default B quant-matrix scale
    output.write_bits(0, 2)?; // one pass
    output.write_bits(0, 1)?; // full-canvas frame
    output.write_bits(0, 2)?; // replace blending
    output.write_bits(1, 1)?; // final frame
    output.write_bits(0, 2)?; // empty frame name
    output.write_bits(0, 1)?; // non-default restoration filter
    output.write_bits(0, 1)?; // no Gaborish
    output.write_bits(0, 2)?; // no EPF
    output.write_bits(0, 2)?; // no restoration extensions
    output.write_bits(0, 2)?; // no frame extensions
    let bit_len = output.bit_len();
    Ok(BitFragment::new(output.into_bytes(), bit_len)?)
}

fn write_u32(
    output: &mut BitWriter,
    value: u32,
    alternatives: [(u32, u8); 4],
) -> Result<(), EncodeError> {
    let Some((selector, offset, bits)) =
        alternatives
            .into_iter()
            .enumerate()
            .find_map(|(selector, (offset, bits))| {
                let encoded = value.checked_sub(offset)?;
                (u64::from(encoded) < (1u64 << bits)).then_some((selector, offset, bits))
            })
    else {
        return Err(EncodeError::InvalidConfiguration(
            "VarDCT integer is outside the JPEG XL U32 representation",
        ));
    };
    output.write_bits(selector as u64, 2)?;
    output.write_bits(u64::from(value - offset), bits)?;
    Ok(())
}

fn write_global_ma_config(
    output: &mut BitWriter,
    codes: &[PrefixCode; 4],
) -> Result<(), EncodeError> {
    // A fixed four-cluster MA tree. All four distributions are identical so
    // stream/channel routing cannot change the GPU token bit representation.
    output.write_bits(1, 1)?; // global MA tree present
    output.write_bits(0, 1)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 4)?;
    output.write_bits(0b100011, 6)?;
    output.write_bits(1, 2)?;
    output.write_bits(3, 2)?;
    for symbol in 0..4 {
        output.write_bits(symbol, 2)?;
    }
    output.write_bits(0, 1)?;

    const TREE_INDICES: [usize; 26] = [
        1, 2, 1, 4, 1, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0,
    ];
    const SYMBOL_BITS: [u64; 6] = [0b00, 0b10, 0b001, 0b101, 0b0011, 0b0111];
    const SYMBOL_NBITS: [u8; 6] = [2, 2, 3, 3, 4, 4];
    for index in TREE_INDICES {
        output.write_bits(SYMBOL_BITS[index], SYMBOL_NBITS[index])?;
    }

    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0b1010, 4)?;
    output.write_bits(4, 4)?;
    output.write_bits(0, 3)?;
    output.write_bits(0, 3)?;
    output.write_bits(1, 1)?;
    output.write_bits(3, 2)?;
    for context in [4, 3, 2, 1, 0] {
        output.write_bits(context, 3)?;
    }
    output.write_bits(1, 1)?;
    output.write_bits(0, 4)?;
    for _ in 0..4 {
        output.write_bits(0, 4)?;
    }
    output.write_bits(1, 5)?;
    for _ in 0..4 {
        output.write_bits(1, 1)?;
        output.write_bits(8, 4)?;
        output.write_bits(0, 8)?;
    }
    output.write_bits(1, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    for code in codes {
        code.write_tree(output)?;
    }
    Ok(())
}

fn write_lf_global(output: &mut BitWriter, code: &PrefixCode) -> Result<(), EncodeError> {
    output.write_bits(1, 1)?; // default LF dequantization
    write_u32(
        output,
        GLOBAL_SCALE,
        [(1, 11), (2_049, 11), (4_097, 12), (8_193, 16)],
    )?;
    write_u32(output, QUANT_LF, [(16, 0), (1, 5), (1, 8), (1, 16)])?;
    output.write_bits(1, 1)?; // default HF block contexts
    output.write_bits(1, 1)?; // default LF channel correlation
    write_global_ma_config(
        output,
        &[code.clone(), code.clone(), code.clone(), code.clone()],
    )
}

fn write_local_modular_header(output: &mut BitWriter) -> Result<(), EncodeError> {
    output.write_bits(1, 1)?; // use the LF-global MA tree
    output.write_bits(1, 1)?; // default weighted-predictor header
    output.write_bits(0, 2)?; // zero transforms
    Ok(())
}

fn write_unsigned_token(
    output: &mut BitWriter,
    code: &PrefixCode,
    value: u32,
) -> Result<(), EncodeError> {
    if value == 0 {
        return code.write_raw(output, 0, 0, 0);
    }
    let nbits = 31 - value.leading_zeros();
    let token = nbits + 1;
    code.write_raw(output, token, nbits, value - (1 << nbits))
}

fn pack_signed_control(value: i32) -> u32 {
    if value < 0 {
        value.unsigned_abs() * 2 - 1
    } else {
        value as u32 * 2
    }
}

#[derive(Clone, Copy)]
struct VarDctArtifactData<'a> {
    block_count: u32,
    strategy: u32,
    dc_fragment_words: &'a [u32],
    dc_fragment_bit_len: u32,
}

fn append_gpu_dc_fragment(
    output: &mut BitWriter,
    artifact: VarDctArtifactData<'_>,
) -> Result<(), EncodeError> {
    let bit_len = usize::try_from(artifact.dc_fragment_bit_len)
        .map_err(|_| EncodeError::Backend("GPU DC fragment length overflow".into()))?;
    if bit_len > artifact.dc_fragment_words.len() * 32 {
        return Err(EncodeError::Backend(
            "GPU DC fragment exceeds its fixed artifact allocation".into(),
        ));
    }
    for bit_index in 0..bit_len {
        let word = artifact.dc_fragment_words[bit_index / 32];
        output.write_bits(u64::from((word >> (bit_index % 32)) & 1), 1)?;
    }
    Ok(())
}

fn write_lf_group(
    output: &mut BitWriter,
    code: &PrefixCode,
    artifact: VarDctArtifactData<'_>,
    frame: VarDctFrameLayout,
) -> Result<(), EncodeError> {
    output.write_bits(0, 2)?; // no extra LF precision
    write_local_modular_header(output)?;
    append_gpu_dc_fragment(output, artifact)?;

    // GPU-selected regular strategies, no chroma-from-luma correction, fixed
    // HF multiplier, and zero EPF sharpness. Source-dependent DC entropy was
    // already packed by the GPU; these values describe its control map.
    let first_block_bits = artifact.block_count.next_power_of_two().trailing_zeros() as u8;
    output.write_bits(
        u64::from(frame.first_block_count().checked_sub(1).ok_or(
            EncodeError::InvalidConfiguration("VarDCT frame has no first transform block"),
        )?),
        first_block_bits,
    )?;
    write_local_modular_header(output)?;
    let correlation_samples = frame.blocks_x.div_ceil(8) * frame.blocks_y.div_ceil(8);
    // The two chroma-from-luma maps are tiled on the 8x8-block grid. They are
    // one sample each through DCT64, then scale to 2x2 and 4x4 for the
    // DCT128/DCT256 families.
    for _ in 0..2 * correlation_samples {
        write_unsigned_token(output, code, 0)?;
    }
    for _ in 0..frame.first_block_count() {
        write_unsigned_token(output, code, pack_signed_control(artifact.strategy as i32))?;
    }
    let first_quant_residual = (HF_MUL - 1) - artifact.strategy as i32;
    write_unsigned_token(output, code, pack_signed_control(first_quant_residual))?;
    for _ in 1..frame.first_block_count() {
        write_unsigned_token(output, code, 0)?;
    }
    for _ in 0..artifact.block_count {
        write_unsigned_token(output, code, 0)?;
    }
    Ok(())
}

fn write_hf_global(output: &mut BitWriter, ac_groups: u32) -> Result<(), EncodeError> {
    // Default dequant matrices, natural coefficient order, and a single-token
    // HF decoder whose only symbol is zero. All AC coefficients are zero in
    // this LF-first strategy profile, so the pass group has no
    // payload bits. The historical one-group bundle is 0x2495/18 bits. Its
    // first bit precedes a ceil(log2(ac_groups))-wide histogram selector, so
    // split it explicitly when a frame has multiple pass groups.
    output.write_bits(1, 1)?;
    let histogram_bits = ac_groups.next_power_of_two().trailing_zeros() as u8;
    output.write_bits(0, histogram_bits)?;
    output.write_bits(0x124a, 17)?;
    Ok(())
}

fn build_frame_packet(
    artifact: VarDctArtifactData<'_>,
    code: &PrefixCode,
    frame: VarDctFrameLayout,
) -> Result<FramePacketSet, EncodeError> {
    let ac_groups = frame.ac_group_count()?;
    if ac_groups == 1 {
        let mut group = BitWriter::new();
        write_lf_global(&mut group, code)?;
        write_lf_group(&mut group, code, artifact, frame)?;
        write_hf_global(&mut group, ac_groups)?;
        group.align_to_byte()?;
        return Ok(FramePacketSet::new(
            frame_header()?,
            FrameGroupLayout::new(1, 1, 1)?,
            [GroupPacket::new(
                GroupPacketKind::Single,
                group.into_bytes(),
            )],
        )?);
    }

    let mut dc_global = BitWriter::new();
    write_lf_global(&mut dc_global, code)?;
    dc_global.align_to_byte()?;
    let mut dc_group = BitWriter::new();
    write_lf_group(&mut dc_group, code, artifact, frame)?;
    dc_group.align_to_byte()?;
    let mut ac_global = BitWriter::new();
    write_hf_global(&mut ac_global, ac_groups)?;
    ac_global.align_to_byte()?;

    let mut packets = Vec::with_capacity(
        usize::try_from(ac_groups)
            .map_err(|_| EncodeError::InvalidConfiguration("VarDCT AC group count overflow"))?
            + 3,
    );
    packets.push(GroupPacket::new(
        GroupPacketKind::DcGlobal,
        dc_global.into_bytes(),
    ));
    packets.push(GroupPacket::new(
        GroupPacketKind::DcGroup(0),
        dc_group.into_bytes(),
    ));
    packets.push(GroupPacket::new(
        GroupPacketKind::AcGlobal,
        ac_global.into_bytes(),
    ));
    packets
        .extend((0..ac_groups).map(|group| {
            GroupPacket::new(GroupPacketKind::AcGroup { pass: 0, group }, Vec::new())
        }));
    Ok(FramePacketSet::new(
        frame_header()?,
        FrameGroupLayout::new(1, ac_groups, 1)?,
        packets,
    )?)
}

#[derive(Clone, Copy, Debug)]
struct VarDctDispatchPlan {
    source_binding_offset: u64,
    source_binding_size: NonZeroU64,
    kernel: VarDctKernelPlan,
    memory: VarDctMemoryPlan,
    frame: VarDctFrameLayout,
}

#[derive(Clone, Copy, Debug)]
enum VarDctKernelPlan {
    Bounded(VarDctKernelParams),
    Scalable {
        params: ScalableVarDctKernelParams,
        layout: ScalableArtifactLayout,
    },
}

enum VarDctPipelines {
    Bounded(Arc<wgpu::ComputePipeline>),
    Scalable {
        quantize: Arc<wgpu::ComputePipeline>,
        serialize: Arc<wgpu::ComputePipeline>,
    },
}

/// GPU backend for one standard VarDCT still-image strategy.
///
/// The source extent must equal the selected transform extent. The backend
/// emits a standards-compliant VarDCT frame and does not route pixels or
/// coefficients through a CPU codec.
pub struct VarDctBackend {
    pipelines: VarDctPipelines,
    workgroup_variant: KernelVariant,
    code: PrefixCode,
    topology: VarDctTopology,
    capabilities: EncoderCapabilities,
    max_storage_binding_size: u64,
    max_buffer_size: u64,
    max_compute_workgroups_per_dimension: u32,
    storage_offset_alignment: u64,
}

impl VarDctBackend {
    /// Creates a standard VarDCT strategy backend and its compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns an encoder error if the fixed standard entropy tree cannot be
    /// represented by the JPEG XL prefix-code writer.
    pub fn new(context: &WgpuContext, strategy: VarDctStrategy) -> Result<Self, EncodeError> {
        Self::new_with_topology(context, VarDctTopology::SingleTransform(strategy))
    }

    /// Creates the bounded tiled-DCT8 profile used by [`TiledVarDctEncoder`].
    /// Every padded 8x8 block is an independent regular transform. The source
    /// extent selects the checked block and AC-group grid at submission time.
    pub fn new_tiled_dct8(context: &WgpuContext) -> Result<Self, EncodeError> {
        Self::new_with_topology(context, VarDctTopology::TiledDct8)
    }

    fn new_with_topology(
        context: &WgpuContext,
        topology: VarDctTopology,
    ) -> Result<Self, EncodeError> {
        let code = fixed_prefix_code()?;
        let limits = context.device().limits();
        let (kernel_key, default_variant, workgroup_storage_bytes) =
            if topology.uses_scalable_kernel() {
                (
                    SCALABLE_QUANTIZE_KERNEL_KEY,
                    KernelVariant::Lanes64,
                    LARGE_WORKGROUP_STORAGE_BYTES,
                )
            } else {
                (
                    BOUNDED_KERNEL_KEY,
                    KernelVariant::Lanes256,
                    BOUNDED_WORKGROUP_STORAGE_BYTES,
                )
            };
        let workgroup_variant = context
            .kernel_policy()
            .variant_for(kernel_key, default_variant)?;
        workgroup_variant.validate_for(kernel_key, &limits, workgroup_storage_bytes)?;
        let (workgroup_x, _) = workgroup_variant.workgroup_size();
        let workgroup_constants = [("wg_x", f64::from(workgroup_x))];
        let pipelines = if topology.uses_scalable_kernel() {
            validate_scalable_device_limits(&limits)?;
            let module = context
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("jxl-wgpu scalable VarDCT kernel"),
                    source: wgpu::ShaderSource::Wgsl(LARGE_SHADER.into()),
                });
            VarDctPipelines::Scalable {
                quantize: Arc::new(context.device().create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("jxl-wgpu scalable VarDCT block quantization"),
                        layout: None,
                        module: &module,
                        entry_point: Some("quantize_blocks"),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &workgroup_constants,
                            ..Default::default()
                        },
                        cache: None,
                    },
                )),
                serialize: Arc::new(context.device().create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("jxl-wgpu scalable VarDCT control serialization"),
                        layout: None,
                        module: &module,
                        entry_point: Some("serialize_control"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    },
                )),
            }
        } else {
            let module = context
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("jxl-wgpu VarDCT forward-transform kernel"),
                    source: wgpu::ShaderSource::Wgsl(SHADER.into()),
                });
            VarDctPipelines::Bounded(Arc::new(context.device().create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("jxl-wgpu VarDCT strategy pipeline"),
                    layout: None,
                    module: &module,
                    entry_point: Some("encode"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &workgroup_constants,
                        ..Default::default()
                    },
                    cache: None,
                },
            )))
        };
        let distance = profile_distance();
        Ok(Self {
            pipelines,
            workgroup_variant,
            code,
            topology,
            capabilities: EncoderCapabilities {
                profiles: vec![ProfileCapability::VarDct {
                    min_distance: distance,
                    max_distance: distance,
                }],
                max_progressive_passes: 1,
                animation: false,
                determinism: Determinism::SameDevice,
                implemented_stages: vec![
                    KernelStage::InputNormalization,
                    KernelStage::ColorTransform,
                    KernelStage::ForwardTransform,
                    KernelStage::Quantization,
                    KernelStage::CoefficientTokenization,
                    KernelStage::HistogramReduction,
                ],
            },
            max_storage_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            storage_offset_alignment: u64::from(limits.min_storage_buffer_offset_alignment),
        })
    }

    /// Selected linear workgroup for the parallel forward/quantization pass.
    ///
    /// The scalable control serializer remains a separate fixed scalar pass because its DC
    /// prediction and bit-offset state are sequential.
    #[must_use]
    pub const fn workgroup_variant(&self) -> KernelVariant {
        self.workgroup_variant
    }

    /// Computes the exact memory admission and source binding before a job is
    /// submitted.
    pub fn memory_plan(&self, source: &BufferImageSource) -> Result<VarDctMemoryPlan, EncodeError> {
        Ok(self.dispatch_plan(source)?.memory)
    }

    fn dispatch_plan(&self, source: &BufferImageSource) -> Result<VarDctDispatchPlan, EncodeError> {
        let extent = source.layout.extent;
        let frame = match self.topology {
            VarDctTopology::SingleTransform(strategy) => {
                let frame = VarDctFrameLayout::single(strategy);
                if extent.width != frame.width || extent.height != frame.height {
                    return Err(EncodeError::InvalidSource(
                        "the VarDCT source extent must equal the selected transform extent",
                    ));
                }
                frame
            }
            VarDctTopology::TiledDct8 => {
                VarDctFrameLayout::tiled_dct8(extent.width, extent.height)?
            }
        };
        if source.layout.format != VarDctColorEncoding::SrgbD65.pixel_format()
            || source.layout.planes.len() != 1
            || !source.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        let plane = source
            .layout
            .plane(0)
            .ok_or(EncodeError::InvalidSource("missing VarDCT RGB plane"))?;
        let row_bytes = u64::from(extent.width) * 3;
        if plane.row_bytes != row_bytes || plane.row_stride < row_bytes {
            return Err(EncodeError::InvalidSource(
                "the VarDCT RGB plane has an invalid row layout",
            ));
        }
        let row_stride = u32::try_from(plane.row_stride)
            .map_err(|_| EncodeError::InvalidSource("VarDCT row stride exceeds WGSL u32"))?;
        let sample_end = plane
            .row_stride
            .checked_mul(u64::from(extent.height - 1))
            .and_then(|rows| plane.offset.checked_add(rows))
            .and_then(|offset| offset.checked_add(row_bytes))
            .ok_or(EncodeError::InvalidSource(
                "VarDCT source address arithmetic overflow",
            ))?;
        let binding_end = align_up(sample_end, 4).ok_or(EncodeError::InvalidSource(
            "VarDCT source binding size overflow",
        ))?;
        if binding_end > source.buffer.size() {
            return Err(EncodeError::InvalidSource(
                "VarDCT source binding does not contain the final sample word",
            ));
        }
        let alignment = self.storage_offset_alignment.max(4);
        let source_binding_offset = plane.offset - plane.offset % alignment;
        let source_binding_bytes =
            binding_end
                .checked_sub(source_binding_offset)
                .ok_or(EncodeError::InvalidSource(
                    "VarDCT source binding range underflow",
                ))?;
        if source_binding_bytes > self.max_storage_binding_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                required: source_binding_bytes,
                available: self.max_storage_binding_size,
            }
            .into());
        }
        let source_binding_size = NonZeroU64::new(source_binding_bytes).ok_or(
            EncodeError::InvalidSource("VarDCT source binding must not be empty"),
        )?;
        let relative_offset =
            plane
                .offset
                .checked_sub(source_binding_offset)
                .ok_or(EncodeError::InvalidSource(
                    "VarDCT source address arithmetic underflow",
                ))?;
        let shader_last_byte = sample_end
            .checked_sub(source_binding_offset)
            .and_then(|end| end.checked_sub(1))
            .ok_or(EncodeError::InvalidSource(
                "VarDCT source address arithmetic underflow",
            ))?;
        u32::try_from(shader_last_byte).map_err(|_| {
            EncodeError::InvalidSource("VarDCT source address exceeds the WGSL u32 space")
        })?;
        let byte_offset = u32::try_from(relative_offset).map_err(|_| {
            EncodeError::InvalidSource("VarDCT source offset exceeds the WGSL u32 space")
        })?;
        let blocks_x = frame.blocks_x;
        let blocks_y = frame.blocks_y;
        let common_strategy = u32::from(frame.topology.strategy().codestream_id());
        let (kernel, memory) = if frame.topology.uses_scalable_kernel() {
            let layout = match frame.topology {
                VarDctTopology::SingleTransform(strategy) => {
                    ScalableArtifactLayout::new(strategy, &self.code)?
                }
                VarDctTopology::TiledDct8 => {
                    ScalableArtifactLayout::for_block_grid(blocks_x, blocks_y, &self.code)?
                }
            };
            let block_count = frame.block_count()?;
            if block_count > self.max_compute_workgroups_per_dimension {
                return Err(UnsupportedFeature::DeviceLimit {
                    name: "max_compute_workgroups_per_dimension",
                    required: u64::from(block_count),
                    available: u64::from(self.max_compute_workgroups_per_dimension),
                }
                .into());
            }
            let artifact_bytes = layout.artifact_bytes();
            if artifact_bytes > self.max_storage_binding_size {
                return Err(UnsupportedFeature::DeviceLimit {
                    name: "max_storage_buffer_binding_size",
                    required: artifact_bytes,
                    available: self.max_storage_binding_size,
                }
                .into());
            }
            if artifact_bytes > self.max_buffer_size {
                return Err(UnsupportedFeature::DeviceLimit {
                    name: "max_buffer_size",
                    required: artifact_bytes,
                    available: self.max_buffer_size,
                }
                .into());
            }
            (
                VarDctKernelPlan::Scalable {
                    params: ScalableVarDctKernelParams {
                        row_stride,
                        byte_offset,
                        width: extent.width,
                        height: extent.height,
                        blocks_x,
                        blocks_y,
                        strategy: common_strategy,
                        global_scale: GLOBAL_SCALE,
                        quant_lf: QUANT_LF,
                        raw_prefix: prefix_entries(&self.code),
                        strategy_offset: layout.strategy_offset,
                        dc_offset: layout.dc_offset,
                        token_offset: layout.token_offset,
                        extra_offset: layout.extra_offset,
                        fragment_offset: layout.fragment_offset,
                        fragment_word_capacity: layout.fragment_word_capacity,
                        artifact_words: layout.artifact_words,
                        topology: frame.topology.artifact_id(),
                        padding: [0; 9],
                    },
                    layout,
                },
                VarDctMemoryPlan::scalable(
                    source_binding_bytes,
                    artifact_bytes,
                    frame.topology.kernel_layout(),
                ),
            )
        } else {
            (
                VarDctKernelPlan::Bounded(VarDctKernelParams {
                    row_stride,
                    byte_offset,
                    width: extent.width,
                    height: extent.height,
                    blocks_x,
                    blocks_y,
                    strategy: common_strategy,
                    global_scale: GLOBAL_SCALE,
                    quant_lf: QUANT_LF,
                    raw_prefix: prefix_entries(&self.code),
                    padding: [0; 17],
                }),
                VarDctMemoryPlan::fixed(source_binding_bytes),
            )
        };
        Ok(VarDctDispatchPlan {
            source_binding_offset,
            source_binding_size,
            kernel,
            memory,
            frame,
        })
    }
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let adjustment = alignment.checked_sub(1)?;
    value
        .checked_add(adjustment)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

fn validate_scalable_device_limits(limits: &wgpu::Limits) -> Result<(), EncodeError> {
    let checks = [(
        "max_storage_buffers_per_shader_stage",
        3,
        u64::from(limits.max_storage_buffers_per_shader_stage),
    )];
    if let Some((name, required, available)) = checks
        .into_iter()
        .find(|(_, required, available)| required > available)
    {
        return Err(UnsupportedFeature::DeviceLimit {
            name,
            required,
            available,
        }
        .into());
    }
    Ok(())
}

fn profile_distance() -> PerceptualDistance {
    PerceptualDistance::new(PROFILE_DISTANCE)
        .expect("the fixed VarDCT distance is within the public validated range")
}

fn validate_vardct_request(
    request: &FrameEncodeRequest,
    frame: VarDctFrameLayout,
) -> Result<(), EncodeError> {
    if request.frame_index != FrameIndex::new(0)
        || !request.is_last
        || request.animation != AnimationHeader::Still
        || request.canvas_width != frame.width
        || request.canvas_height != frame.height
        || request.options != FrameOptions::default()
        || request.progressive != ProgressivePlan::single()
    {
        return Err(EncodeError::InvalidConfiguration(
            "the VarDCT profile requires one full-canvas final transform-sized still frame",
        ));
    }
    if request.profile
        != (EncodeProfile::VarDct {
            distance: profile_distance(),
        })
    {
        return Err(EncodeError::InvalidConfiguration(
            "the requested VarDCT distance does not match the fixed LF-first profile",
        ));
    }
    Ok(())
}

impl GpuEncodeBackend for VarDctBackend {
    type Job = VarDctJob;

    fn capabilities(&self) -> &EncoderCapabilities {
        &self.capabilities
    }

    fn supports_input(&self, source: &GpuFrameSource) -> bool {
        let GpuFrameSource::Buffer(source) = source else {
            return false;
        };
        self.dispatch_plan(source).is_ok()
    }

    fn submit(
        &self,
        context: &WgpuContext,
        source: GpuFrameSource,
        request: &FrameEncodeRequest,
    ) -> Result<Self::Job, EncodeError> {
        let GpuFrameSource::Buffer(source) = source else {
            return Err(UnsupportedFeature::InputFormat.into());
        };
        let plan = self.dispatch_plan(&source)?;
        validate_vardct_request(request, plan.frame)?;
        let memory_permit = context
            .memory_budget()
            .try_reserve(plan.memory.owned_bytes_per_job)?;

        let parameters = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu VarDCT parameters"),
            size: plan.memory.parameter_storage_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let artifact = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu VarDCT artifact"),
            size: plan.memory.artifact_storage_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let readback = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu VarDCT readback"),
            size: plan.memory.readback_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        context.queue().write_buffer(
            &parameters,
            0,
            match &plan.kernel {
                VarDctKernelPlan::Bounded(params) => bytemuck::bytes_of(params),
                VarDctKernelPlan::Scalable { params, .. } => bytemuck::bytes_of(params),
            },
        );

        let source_binding = wgpu::BufferBinding {
            buffer: &source.buffer,
            offset: plan.source_binding_offset,
            size: Some(plan.source_binding_size),
        };
        let params_binding_size = NonZeroU64::new(plan.memory.parameter_storage_bytes)
            .expect("the VarDCT parameter ABI is non-empty");
        let artifact_binding_size = NonZeroU64::new(plan.memory.artifact_storage_bytes)
            .expect("the VarDCT artifact ABI is non-empty");
        let create_bind_group = |pipeline: &wgpu::ComputePipeline, label| {
            context
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(source_binding.clone()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &parameters,
                                offset: 0,
                                size: Some(params_binding_size),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &artifact,
                                offset: 0,
                                size: Some(artifact_binding_size),
                            }),
                        },
                    ],
                })
        };
        let mut commands =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu VarDCT encode"),
                });
        commands.clear_buffer(&artifact, 0, None);
        let job_layout = match (&self.pipelines, plan.kernel) {
            (VarDctPipelines::Bounded(pipeline), VarDctKernelPlan::Bounded(_)) => {
                let bind_group = create_bind_group(pipeline, "jxl-wgpu VarDCT bindings");
                let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("jxl-wgpu VarDCT forward transform and tokenization"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
                VarDctJobLayout::Bounded
            }
            (
                VarDctPipelines::Scalable {
                    quantize,
                    serialize,
                },
                VarDctKernelPlan::Scalable { params, layout },
            ) => {
                let quantize_bind_group =
                    create_bind_group(quantize, "jxl-wgpu scalable VarDCT quantization bindings");
                {
                    let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("jxl-wgpu scalable VarDCT 8x8 DC quantization"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(quantize);
                    pass.set_bind_group(0, &quantize_bind_group, &[]);
                    pass.dispatch_workgroups(params.blocks_x * params.blocks_y, 1, 1);
                }
                // A separate WebGPU pass is the explicit global storage
                // visibility boundary for all block workgroups before the
                // single deterministic prediction/serialization invocation.
                // The control entry point intentionally has no source binding;
                // automatic pipeline layouts therefore retain only bindings
                // 1 and 2 for this pass.
                let serialize_bind_group =
                    context
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("jxl-wgpu scalable VarDCT serialization bindings"),
                            layout: &serialize.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &parameters,
                                        offset: 0,
                                        size: Some(params_binding_size),
                                    }),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &artifact,
                                        offset: 0,
                                        size: Some(artifact_binding_size),
                                    }),
                                },
                            ],
                        });
                {
                    let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("jxl-wgpu scalable VarDCT control and entropy serialization"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(serialize);
                    pass.set_bind_group(0, &serialize_bind_group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                VarDctJobLayout::Scalable(layout)
            }
            _ => {
                return Err(BackendError::Invariant(
                    "VarDCT strategy selected incompatible GPU pipelines",
                )
                .into());
            }
        };
        commands.copy_buffer_to_buffer(
            &artifact,
            0,
            &readback,
            0,
            plan.memory.artifact_storage_bytes,
        );

        let completion = Arc::new(VarDctMapCompletion::default());
        let callback_completion = Arc::clone(&completion);
        let readback_for_map = Arc::clone(&readback);
        let lifetime = Arc::new(VarDctJobLifetime {
            _parameters: parameters,
            _artifact: artifact,
            readback,
            _memory_permit: memory_permit,
            mapped: AtomicBool::new(false),
        });
        let callback_lifetime = Arc::clone(&lifetime);
        commands.map_buffer_on_submit(
            &readback_for_map,
            wgpu::MapMode::Read,
            0..plan.memory.readback_bytes,
            move |result| {
                if result.is_ok() {
                    callback_lifetime.mapped.store(true, Ordering::Release);
                }
                callback_completion.complete(result.map_err(BackendError::ArtifactMapping));
                drop(callback_lifetime);
            },
        );
        let poll_permit = context.submission_poller().try_reserve()?;
        let submission_index = context.queue().submit([commands.finish()]);
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll_permit.register(submission_index, move |error| {
            poll_completion.complete(Err(BackendError::PollWorker(error)));
        }) {
            completion.complete(Err(BackendError::PollRegistration(error)));
        }

        Ok(VarDctJob {
            lifetime: Some(lifetime),
            completion,
            code: self.code.clone(),
            frame_layout: plan.frame,
            artifact_layout: job_layout,
            frame_index: request.frame_index,
            is_last: request.is_last,
        })
    }
}

#[derive(Default)]
struct VarDctMapCompletion {
    state: Mutex<VarDctMapState>,
    condition: Condvar,
}

#[derive(Default)]
struct VarDctMapState {
    result: Option<Result<(), BackendError>>,
    waker: Option<Waker>,
}

impl VarDctMapCompletion {
    fn complete(&self, result: Result<(), BackendError>) {
        let waker = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        self.condition.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn poll(&self, cx: &Context<'_>) -> Option<Result<(), BackendError>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.result.is_none() {
            state.waker = Some(cx.waker().clone());
        }
        state.result.take()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) -> Result<(), BackendError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.result.is_none() {
            state = self
                .condition
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state
            .result
            .take()
            .expect("VarDCT map completion was checked as present")
    }
}

struct VarDctJobLifetime {
    _parameters: Arc<wgpu::Buffer>,
    _artifact: Arc<wgpu::Buffer>,
    readback: Arc<wgpu::Buffer>,
    _memory_permit: MemoryPermit,
    mapped: AtomicBool,
}

impl Drop for VarDctJobLifetime {
    fn drop(&mut self) {
        if self.mapped.swap(false, Ordering::AcqRel) {
            self.readback.unmap();
        }
    }
}

/// Runtime-neutral completion for one standard VarDCT GPU submission.
#[derive(Clone, Copy, Debug)]
enum VarDctJobLayout {
    Bounded,
    Scalable(ScalableArtifactLayout),
}

pub struct VarDctJob {
    lifetime: Option<Arc<VarDctJobLifetime>>,
    completion: Arc<VarDctMapCompletion>,
    code: PrefixCode,
    frame_layout: VarDctFrameLayout,
    artifact_layout: VarDctJobLayout,
    frame_index: FrameIndex,
    is_last: bool,
}

impl VarDctJob {
    fn finish(
        &mut self,
        mapping: Result<(), BackendError>,
    ) -> Result<GpuFrameArtifacts, EncodeError> {
        let lifetime = self.lifetime.take().ok_or(BackendError::Invariant(
            "VarDCT GPU job was already consumed",
        ))?;
        mapping?;
        let mapped = match lifetime.readback.slice(..).get_mapped_range() {
            Ok(mapped) => mapped,
            Err(error) => {
                lifetime.readback.unmap();
                lifetime.mapped.store(false, Ordering::Release);
                return Err(BackendError::ArtifactRange(error).into());
            }
        };
        let result = (|| {
            let artifact = match self.artifact_layout {
                VarDctJobLayout::Bounded => {
                    let artifact = bytemuck::try_from_bytes::<VarDctKernelArtifact>(&mapped)
                        .map_err(|_| {
                            BackendError::InvalidArtifact("VarDCT ABI size or alignment")
                        })?;
                    validate_artifact(artifact, &self.code, self.frame_layout)?
                }
                VarDctJobLayout::Scalable(layout) => {
                    validate_scalable_artifact(&mapped, layout, &self.code, self.frame_layout)?
                }
            };
            Ok(GpuFrameArtifacts {
                frame_index: self.frame_index,
                is_last: self.is_last,
                packets: build_frame_packet(artifact, &self.code, self.frame_layout)?,
                acceleration: None,
            })
        })();
        drop(mapped);
        lifetime.readback.unmap();
        lifetime.mapped.store(false, Ordering::Release);
        drop(lifetime);
        result
    }
}

impl GpuEncodeJob for VarDctJob {
    fn poll_complete(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<GpuFrameArtifacts, EncodeError>> {
        match self.completion.poll(cx) {
            Some(result) => Poll::Ready(self.finish(result)),
            None => Poll::Pending,
        }
    }

    fn wait(self) -> Result<GpuFrameArtifacts, EncodeError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut job = self;
            let result = job.completion.wait();
            job.finish(result)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(BackendError::Invariant(
                "blocking GPU waits are unavailable on browser WebGPU; await the submission",
            )
            .into())
        }
    }
}

fn validate_artifact<'a>(
    artifact: &'a VarDctKernelArtifact,
    code: &PrefixCode,
    frame: VarDctFrameLayout,
) -> Result<VarDctArtifactData<'a>, BackendError> {
    let VarDctTopology::SingleTransform(strategy) = frame.topology else {
        return Err(BackendError::InvalidArtifact(
            "the fixed VarDCT artifact cannot represent a tiled frame",
        ));
    };
    let (blocks_x, blocks_y) = strategy.block_grid();
    let block_count = usize::try_from(blocks_x * blocks_y)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT block count does not fit usize"))?;
    let expected_strategy = u32::from(strategy.codestream_id());
    if artifact.strategy != expected_strategy
        || artifact.block_count != block_count as u32
        || artifact.dc_sample_count != (3 * block_count) as u32
    {
        return Err(BackendError::InvalidArtifact(
            "VarDCT strategy or live-count header mismatch",
        ));
    }
    for block in 0..MAX_BLOCKS {
        let expected = if block < block_count {
            expected_strategy | u32::from(block == 0) << 8
        } else {
            0
        };
        if artifact.strategy_map[block] != expected {
            return Err(BackendError::InvalidArtifact(
                "VarDCT GPU strategy map is malformed",
            ));
        }
    }

    let coefficient_count =
        usize::from(strategy.block_extent().0) * usize::from(strategy.block_extent().1);
    let xyb_channels = [1usize, 0, 2];
    for (dc_channel, &xyb_channel) in xyb_channels.iter().enumerate() {
        let dc_base = dc_channel * MAX_BLOCKS;
        let coefficient_base = xyb_channel * MAX_COEFFICIENTS;
        for block in 0..block_count {
            if artifact.quantized_dc_yxb[dc_base + block]
                != artifact.quantized_xyb[coefficient_base + block]
            {
                return Err(BackendError::InvalidArtifact(
                    "VarDCT DC channel ordering mismatch",
                ));
            }
        }
        if artifact.quantized_dc_yxb[dc_base + block_count..dc_base + MAX_BLOCKS]
            .iter()
            .any(|&value| value != 0)
            || artifact.quantized_xyb
                [coefficient_base + block_count..coefficient_base + MAX_COEFFICIENTS]
                .iter()
                .any(|&value| value != 0)
        {
            return Err(BackendError::InvalidArtifact(
                "the VarDCT profile produced a nonzero AC or padding token",
            ));
        }
    }
    if artifact
        .forward_xyb_bits
        .chunks_exact(MAX_COEFFICIENTS)
        .flat_map(|channel| &channel[..coefficient_count])
        .any(|&bits| !f32::from_bits(bits).is_finite())
    {
        return Err(BackendError::InvalidArtifact(
            "VarDCT forward transform produced a non-finite coefficient",
        ));
    }

    let entries = code.raw_entries();
    let mut expected_histogram = [0u32; RAW_SYMBOLS];
    let mut bit_offset = 0u32;
    for channel in 0..3 {
        let base = channel * MAX_BLOCKS;
        for block in 0..block_count {
            let block_x = block % blocks_x as usize;
            let block_y = block / blocks_x as usize;
            let left = if block_x > 0 {
                artifact.quantized_dc_yxb[base + block - 1]
            } else if block_y > 0 {
                artifact.quantized_dc_yxb[base + block - blocks_x as usize]
            } else {
                0
            };
            let top = if block_y > 0 {
                artifact.quantized_dc_yxb[base + block - blocks_x as usize]
            } else {
                left
            };
            let top_left = if block_x > 0 && block_y > 0 {
                artifact.quantized_dc_yxb[base + block - blocks_x as usize - 1]
            } else {
                left
            };
            let residual =
                gradient_residual_i32(artifact.quantized_dc_yxb[base + block], top, left, top_left);
            let (token, extra_bit_count, extra) = signed_token(residual)?;
            let slot = base + block;
            if artifact.dc_raw_tokens[slot] != token || artifact.dc_extra_bits[slot] != extra {
                return Err(BackendError::InvalidArtifact(
                    "VarDCT DC token does not match its predicted residual",
                ));
            }
            let token_index = usize::try_from(token).map_err(|_| {
                BackendError::InvalidArtifact("VarDCT DC token index does not fit usize")
            })?;
            let entry = entries
                .get(token_index)
                .ok_or(BackendError::InvalidArtifact(
                    "VarDCT DC token exceeds the fixed entropy alphabet",
                ))?;
            if read_fragment_bits(artifact, bit_offset, u32::from(entry.bit_len))?
                != u32::from(entry.bits)
            {
                return Err(BackendError::InvalidArtifact(
                    "VarDCT GPU prefix fragment does not match its token",
                ));
            }
            bit_offset += u32::from(entry.bit_len);
            if read_fragment_bits(artifact, bit_offset, extra_bit_count)? != extra {
                return Err(BackendError::InvalidArtifact(
                    "VarDCT GPU extra-bit fragment does not match its token",
                ));
            }
            bit_offset += extra_bit_count;
            expected_histogram[token_index] += 1;
        }
        if artifact.dc_raw_tokens[base + block_count..base + MAX_BLOCKS]
            .iter()
            .chain(&artifact.dc_extra_bits[base + block_count..base + MAX_BLOCKS])
            .any(|&value| value != 0)
        {
            return Err(BackendError::InvalidArtifact(
                "VarDCT DC token padding is nonzero",
            ));
        }
    }
    if bit_offset != artifact.dc_fragment_bit_len || artifact.raw_histogram != expected_histogram {
        return Err(BackendError::InvalidArtifact(
            "VarDCT GPU entropy fragment length or histogram mismatch",
        ));
    }
    Ok(fixed_artifact_data(artifact))
}

fn fixed_artifact_data(artifact: &VarDctKernelArtifact) -> VarDctArtifactData<'_> {
    VarDctArtifactData {
        block_count: artifact.block_count,
        strategy: artifact.strategy,
        dc_fragment_words: &artifact.dc_fragment_words,
        dc_fragment_bit_len: artifact.dc_fragment_bit_len,
    }
}

fn validate_scalable_artifact<'a>(
    mapped: &'a [u8],
    layout: ScalableArtifactLayout,
    code: &PrefixCode,
    frame: VarDctFrameLayout,
) -> Result<VarDctArtifactData<'a>, BackendError> {
    let expected_bytes = usize::try_from(layout.artifact_bytes()).map_err(|_| {
        BackendError::InvalidArtifact("scalable VarDCT artifact size does not fit usize")
    })?;
    if mapped.len() != expected_bytes {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT mapped artifact has the wrong byte length",
        ));
    }
    let words = bytemuck::try_cast_slice::<u8, u32>(mapped).map_err(|_| {
        BackendError::InvalidArtifact("scalable VarDCT artifact word ABI alignment")
    })?;
    let header_bytes = mapped
        .get(..std::mem::size_of::<ScalableVarDctArtifactHeader>())
        .ok_or(BackendError::InvalidArtifact(
            "scalable VarDCT artifact header is truncated",
        ))?;
    let header = bytemuck::try_from_bytes::<ScalableVarDctArtifactHeader>(header_bytes)
        .map_err(|_| BackendError::InvalidArtifact("scalable VarDCT header ABI alignment"))?;
    let blocks_x = frame.blocks_x;
    let blocks_y = frame.blocks_y;
    let block_count = blocks_x
        .checked_mul(blocks_y)
        .ok_or(BackendError::InvalidArtifact(
            "scalable VarDCT block count overflow",
        ))?;
    let dc_sample_count = block_count
        .checked_mul(3)
        .ok_or(BackendError::InvalidArtifact(
            "scalable VarDCT sample count overflow",
        ))?;
    let strategy = frame.topology.strategy();
    if header.status != SCALABLE_ARTIFACT_READY
        || header.block_count != block_count
        || header.dc_sample_count != dc_sample_count
        || header.strategy != u32::from(strategy.codestream_id())
        || header.ac_all_zero != 1
        || header.strategy_offset != layout.strategy_offset
        || header.strategy_len != layout.strategy_len
        || header.dc_offset != layout.dc_offset
        || header.dc_len != layout.dc_len
        || header.token_offset != layout.token_offset
        || header.token_len != layout.token_len
        || header.extra_offset != layout.extra_offset
        || header.extra_len != layout.extra_len
        || header.fragment_offset != layout.fragment_offset
        || header.fragment_word_capacity != layout.fragment_word_capacity
        || header.artifact_words != layout.artifact_words
        || header.width != frame.width
        || header.height != frame.height
        || header.blocks_x != blocks_x
        || header.blocks_y != blocks_y
        || header.topology != frame.topology.artifact_id()
    {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT status, live counts, orientation, or layout metadata mismatch",
        ));
    }
    if header.padding.iter().any(|&word| word != 0) {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT header padding is nonzero",
        ));
    }
    if header.dc_fragment_bit_len > layout.fragment_max_bits
        || header.dc_fragment_bit_len
            > layout
                .fragment_word_capacity
                .checked_mul(32)
                .ok_or(BackendError::InvalidArtifact(
                    "scalable VarDCT fragment capacity overflow",
                ))?
    {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT entropy fragment exceeds its checked capacity",
        ));
    }

    let strategy_map = artifact_words(words, layout.strategy_offset, layout.strategy_len)?;
    let quantized_dc = artifact_words(words, layout.dc_offset, layout.dc_len)?;
    let raw_tokens = artifact_words(words, layout.token_offset, layout.token_len)?;
    let extra_bits = artifact_words(words, layout.extra_offset, layout.extra_len)?;
    let fragment_words =
        artifact_words(words, layout.fragment_offset, layout.fragment_word_capacity)?;
    validate_zero_gap(words, SCALABLE_HEADER_WORDS, layout.strategy_offset)?;
    validate_zero_gap(
        words,
        layout.strategy_offset + layout.strategy_len,
        layout.dc_offset,
    )?;
    validate_zero_gap(words, layout.dc_offset + layout.dc_len, layout.token_offset)?;
    validate_zero_gap(
        words,
        layout.token_offset + layout.token_len,
        layout.extra_offset,
    )?;
    validate_zero_gap(
        words,
        layout.extra_offset + layout.extra_len,
        layout.fragment_offset,
    )?;
    validate_zero_gap(
        words,
        layout.fragment_offset + layout.fragment_word_capacity,
        layout.artifact_words,
    )?;

    let expected_strategy = u32::from(strategy.codestream_id());
    for (block, &value) in strategy_map.iter().enumerate() {
        let is_first = match frame.topology {
            VarDctTopology::SingleTransform(_) => block == 0,
            VarDctTopology::TiledDct8 => true,
        };
        let expected = expected_strategy | u32::from(is_first) << 8;
        if value != expected {
            return Err(BackendError::InvalidArtifact(
                "scalable VarDCT GPU strategy map is malformed",
            ));
        }
    }

    let block_count_usize = usize::try_from(block_count)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT block count does not fit usize"))?;
    let entries = code.raw_entries();
    let mut expected_histogram = [0u32; RAW_SYMBOLS];
    let mut bit_offset = 0u32;
    for channel in 0..3usize {
        let base = channel * block_count_usize;
        for block in 0..block_count_usize {
            let block_x = block % blocks_x as usize;
            let block_y = block / blocks_x as usize;
            let left = if block_x > 0 {
                quantized_dc[base + block - 1] as i32
            } else if block_y > 0 {
                quantized_dc[base + block - blocks_x as usize] as i32
            } else {
                0
            };
            let top = if block_y > 0 {
                quantized_dc[base + block - blocks_x as usize] as i32
            } else {
                left
            };
            let top_left = if block_x > 0 && block_y > 0 {
                quantized_dc[base + block - blocks_x as usize - 1] as i32
            } else {
                left
            };
            let actual = quantized_dc[base + block] as i32;
            let residual = gradient_residual_i32(actual, top, left, top_left);
            let (token, extra_bit_count, extra) = signed_token(residual)?;
            let slot = base + block;
            if raw_tokens[slot] != token || extra_bits[slot] != extra {
                return Err(BackendError::InvalidArtifact(
                    "scalable VarDCT DC token does not match its predicted residual",
                ));
            }
            let token_index = usize::try_from(token).map_err(|_| {
                BackendError::InvalidArtifact("VarDCT DC token index does not fit usize")
            })?;
            let entry = entries
                .get(token_index)
                .ok_or(BackendError::InvalidArtifact(
                    "VarDCT DC token exceeds the fixed entropy alphabet",
                ))?;
            if read_fragment_slice(
                fragment_words,
                header.dc_fragment_bit_len,
                bit_offset,
                u32::from(entry.bit_len),
            )? != u32::from(entry.bits)
            {
                return Err(BackendError::InvalidArtifact(
                    "scalable VarDCT GPU prefix fragment does not match its token",
                ));
            }
            bit_offset += u32::from(entry.bit_len);
            if read_fragment_slice(
                fragment_words,
                header.dc_fragment_bit_len,
                bit_offset,
                extra_bit_count,
            )? != extra
            {
                return Err(BackendError::InvalidArtifact(
                    "scalable VarDCT GPU extra-bit fragment does not match its token",
                ));
            }
            bit_offset += extra_bit_count;
            expected_histogram[token_index] += 1;
        }
    }
    if bit_offset != header.dc_fragment_bit_len || header.raw_histogram != expected_histogram {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT entropy fragment length or histogram mismatch",
        ));
    }
    validate_fragment_padding(fragment_words, header.dc_fragment_bit_len)?;
    Ok(VarDctArtifactData {
        block_count,
        strategy: expected_strategy,
        dc_fragment_words: fragment_words,
        dc_fragment_bit_len: header.dc_fragment_bit_len,
    })
}

fn artifact_words(words: &[u32], offset: u32, len: u32) -> Result<&[u32], BackendError> {
    let start = usize::try_from(offset)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT artifact offset does not fit usize"))?;
    let len = usize::try_from(len)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT artifact length does not fit usize"))?;
    let end = start.checked_add(len).ok_or(BackendError::InvalidArtifact(
        "VarDCT artifact range overflow",
    ))?;
    words.get(start..end).ok_or(BackendError::InvalidArtifact(
        "VarDCT artifact range is out of bounds",
    ))
}

fn validate_zero_gap(words: &[u32], start: u32, end: u32) -> Result<(), BackendError> {
    if artifact_words(
        words,
        start,
        end.checked_sub(start).ok_or(BackendError::InvalidArtifact(
            "VarDCT artifact section order is invalid",
        ))?,
    )?
    .iter()
    .any(|&word| word != 0)
    {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT artifact alignment padding is nonzero",
        ));
    }
    Ok(())
}

fn validate_fragment_padding(words: &[u32], bit_len: u32) -> Result<(), BackendError> {
    let used_words = bit_len
        .checked_add(31)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT fragment word count overflow",
        ))?
        / 32;
    let used_words = usize::try_from(used_words)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT fragment size does not fit usize"))?;
    if let Some(&last_word) = used_words.checked_sub(1).and_then(|index| words.get(index)) {
        let live_bits = bit_len % 32;
        if live_bits != 0 && last_word & !((1u32 << live_bits) - 1) != 0 {
            return Err(BackendError::InvalidArtifact(
                "scalable VarDCT fragment has nonzero high padding bits",
            ));
        }
    }
    if words
        .get(used_words..)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT fragment used-word count is out of bounds",
        ))?
        .iter()
        .any(|&word| word != 0)
    {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT fragment word padding is nonzero",
        ));
    }
    Ok(())
}

fn read_fragment_slice(
    words: &[u32],
    bit_len: u32,
    start: u32,
    count: u32,
) -> Result<u32, BackendError> {
    let end = start
        .checked_add(count)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT GPU fragment address overflow",
        ))?;
    let capacity = u32::try_from(words.len())
        .ok()
        .and_then(|len| len.checked_mul(32))
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT GPU fragment capacity overflow",
        ))?;
    if end > bit_len || end > capacity {
        return Err(BackendError::InvalidArtifact(
            "VarDCT GPU fragment is truncated",
        ));
    }
    let mut value = 0u32;
    for index in 0..count {
        let bit = start + index;
        value |= ((words[(bit / 32) as usize] >> (bit % 32)) & 1) << index;
    }
    Ok(value)
}

fn clamped_gradient_i32(top: i32, left: i32, top_left: i32) -> i32 {
    top.wrapping_add(left)
        .wrapping_sub(top_left)
        .clamp(top.min(left), top.max(left))
}

fn gradient_residual_i32(actual: i32, top: i32, left: i32, top_left: i32) -> i32 {
    actual.wrapping_sub(clamped_gradient_i32(top, left, top_left))
}

fn signed_token(value: i32) -> Result<(u32, u32, u32), BackendError> {
    let packed = if value >= 0 {
        u64::from(value as u32) * 2
    } else {
        u64::try_from(-i64::from(value)).expect("the negated i32 value fits u64") * 2 - 1
    };
    let packed = u32::try_from(packed).map_err(|_| {
        BackendError::InvalidArtifact("VarDCT signed coefficient exceeds the token alphabet")
    })?;
    if packed == 0 {
        return Ok((0, 0, 0));
    }
    let extra_bit_count = 31 - packed.leading_zeros();
    let token = extra_bit_count + 1;
    if token as usize >= RAW_SYMBOLS {
        return Err(BackendError::InvalidArtifact(
            "VarDCT DC token exceeds the fixed entropy alphabet",
        ));
    }
    Ok((token, extra_bit_count, packed - (1 << extra_bit_count)))
}

fn read_fragment_bits(
    artifact: &VarDctKernelArtifact,
    start: u32,
    count: u32,
) -> Result<u32, BackendError> {
    let end = start
        .checked_add(count)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT GPU fragment address overflow",
        ))?;
    if end > artifact.dc_fragment_bit_len
        || end > u32::try_from(MAX_DC_FRAGMENT_WORDS * 32).expect("fixed artifact fits u32")
    {
        return Err(BackendError::InvalidArtifact(
            "VarDCT GPU fragment is truncated",
        ));
    }
    let mut value = 0u32;
    for index in 0..count {
        let bit = start + index;
        let word = artifact.dc_fragment_words[(bit / 32) as usize];
        value |= ((word >> (bit % 32)) & 1) << index;
    }
    Ok(value)
}

/// GPU-only convenience encoder for one standard VarDCT transform.
pub struct VarDctEncoder {
    encoder: GpuEncoder<VarDctBackend>,
    strategy: VarDctStrategy,
}

impl VarDctEncoder {
    /// Creates the profile backend.
    ///
    /// # Errors
    ///
    /// Returns an encoder error if the fixed standard entropy tree cannot be
    /// constructed or the selected device cannot execute the strategy's
    /// checked storage/workgroup/dispatch requirements.
    pub fn new(context: WgpuContext, strategy: VarDctStrategy) -> Result<Self, EncodeError> {
        let backend = VarDctBackend::new(&context, strategy)?;
        Ok(Self {
            encoder: GpuEncoder::new(context, backend),
            strategy,
        })
    }

    #[must_use]
    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.encoder.capabilities()
    }

    #[must_use]
    pub const fn strategy(&self) -> VarDctStrategy {
        self.strategy
    }

    /// Workgroup selected for this encoder's parallel VarDCT pass.
    #[must_use]
    pub fn workgroup_variant(&self) -> KernelVariant {
        self.encoder.backend().workgroup_variant()
    }

    #[must_use]
    pub const fn color_encoding(&self) -> VarDctColorEncoding {
        VarDctColorEncoding::SrgbD65
    }

    #[must_use]
    pub fn distance(&self) -> PerceptualDistance {
        profile_distance()
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> jxl_wgpu::MemoryBudgetSnapshot {
        self.encoder.memory_stats()
    }

    pub fn memory_plan(&self, source: &BufferImageSource) -> Result<VarDctMemoryPlan, EncodeError> {
        self.encoder.backend().memory_plan(source)
    }

    pub fn submit(&self, source: BufferImageSource) -> Result<VarDctSubmission, EncodeError> {
        self.submit_inner(source, false)
    }

    pub fn submit_container(
        &self,
        source: BufferImageSource,
    ) -> Result<VarDctSubmission, EncodeError> {
        self.submit_inner(source, true)
    }

    pub fn encode(&self, source: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit(source)?.wait()
    }

    pub fn encode_container(&self, source: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit_container(source)?.wait()
    }

    fn submit_inner(
        &self,
        source: BufferImageSource,
        container: bool,
    ) -> Result<VarDctSubmission, EncodeError> {
        self.memory_plan(&source)?;
        let (width, height) = self.strategy.block_extent();
        let width = u32::from(width);
        let height = u32::from(height);
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::VarDct {
                distance: profile_distance(),
            },
            progressive: ProgressivePlan::single(),
            minimum_determinism: Determinism::SameDevice,
            animation: AnimationHeader::Still,
            canvas_width: width,
            canvas_height: height,
            options: FrameOptions::default(),
        };
        let frame = self
            .encoder
            .submit_frame(GpuFrameSource::Buffer(source), request)?;
        Ok(VarDctSubmission {
            frame: Some(frame),
            codestream_header: image_header(width, height)?,
            container,
        })
    }
}

/// GPU-only JPEG XL VarDCT encoder for a rectangular grid of independent
/// regular DCT8 transforms.
///
/// The current executable subset accepts RGB8 dimensions through 2048 pixels
/// on each axis when at least one axis exceeds 256 pixels, including partial
/// 8x8 edge blocks. This guarantees an explicit multi-section TOC: one LF/DC
/// group and at least two 256x256 AC/pass groups. AC coefficients are
/// deliberately zero, so decoded quality is the profile's LF-only contract
/// rather than a general distance-25 guarantee.
pub struct TiledVarDctEncoder {
    encoder: GpuEncoder<VarDctBackend>,
}

impl TiledVarDctEncoder {
    /// Creates the tiled DCT8 backend.
    ///
    /// # Errors
    ///
    /// Returns an encoder error if the fixed entropy tree cannot be built or
    /// the device cannot execute the checked scalable kernel ABI.
    pub fn new(context: WgpuContext) -> Result<Self, EncodeError> {
        let backend = VarDctBackend::new_tiled_dct8(&context)?;
        Ok(Self {
            encoder: GpuEncoder::new(context, backend),
        })
    }

    #[must_use]
    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.encoder.capabilities()
    }

    /// Workgroup selected for block quantization. Control serialization remains scalar.
    #[must_use]
    pub fn workgroup_variant(&self) -> KernelVariant {
        self.encoder.backend().workgroup_variant()
    }

    #[must_use]
    pub const fn color_encoding(&self) -> VarDctColorEncoding {
        VarDctColorEncoding::SrgbD65
    }

    #[must_use]
    pub fn distance(&self) -> PerceptualDistance {
        profile_distance()
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> jxl_wgpu::MemoryBudgetSnapshot {
        self.encoder.memory_stats()
    }

    pub fn memory_plan(&self, source: &BufferImageSource) -> Result<VarDctMemoryPlan, EncodeError> {
        self.encoder.backend().memory_plan(source)
    }

    pub fn grid(&self, source: &BufferImageSource) -> Result<TiledVarDctGrid, EncodeError> {
        self.memory_plan(source)?;
        TiledVarDctGrid::new(source.layout.extent.width, source.layout.extent.height)
    }

    pub fn submit(&self, source: BufferImageSource) -> Result<VarDctSubmission, EncodeError> {
        self.submit_inner(source, false)
    }

    pub fn submit_container(
        &self,
        source: BufferImageSource,
    ) -> Result<VarDctSubmission, EncodeError> {
        self.submit_inner(source, true)
    }

    pub fn encode(&self, source: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit(source)?.wait()
    }

    pub fn encode_container(&self, source: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit_container(source)?.wait()
    }

    fn submit_inner(
        &self,
        source: BufferImageSource,
        container: bool,
    ) -> Result<VarDctSubmission, EncodeError> {
        let frame =
            VarDctFrameLayout::tiled_dct8(source.layout.extent.width, source.layout.extent.height)?;
        self.memory_plan(&source)?;
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::VarDct {
                distance: profile_distance(),
            },
            progressive: ProgressivePlan::single(),
            minimum_determinism: Determinism::SameDevice,
            animation: AnimationHeader::Still,
            canvas_width: frame.width,
            canvas_height: frame.height,
            options: FrameOptions::default(),
        };
        let frame_submission = self
            .encoder
            .submit_frame(GpuFrameSource::Buffer(source), request)?;
        Ok(VarDctSubmission {
            frame: Some(frame_submission),
            codestream_header: image_header(frame.width, frame.height)?,
            container,
        })
    }
}

/// Executor-independent future for a complete standard VarDCT codestream.
pub struct VarDctSubmission {
    frame: Option<FrameSubmission<VarDctJob>>,
    codestream_header: BitFragment,
    container: bool,
}

impl VarDctSubmission {
    pub fn wait(mut self) -> Result<Vec<u8>, EncodeError> {
        let frame = self
            .frame
            .take()
            .expect("a VarDCT submission can only complete once")
            .wait()?;
        self.assemble(frame)
    }

    fn assemble(&self, frame: GpuFrameArtifacts) -> Result<Vec<u8>, EncodeError> {
        let encoded_frame = assemble_frame(frame.packets)?;
        let mut codestream = self.codestream_header.bytes().to_vec();
        codestream.extend_from_slice(encoded_frame.bytes());
        if self.container {
            Ok(jxl_gpu_bitstream::write_container(&codestream)?)
        } else {
            Ok(codestream)
        }
    }
}

impl Future for VarDctSubmission {
    type Output = Result<Vec<u8>, EncodeError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let submission = self.get_mut();
        let frame = submission
            .frame
            .as_mut()
            .expect("a VarDCT submission must not be polled after completion");
        match Pin::new(frame).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                submission.frame.take();
                Poll::Ready(result.and_then(|frame| submission.assemble(frame)))
            }
        }
    }
}

#[cfg(test)]
fn cpu_test_artifact(q_yxb: [i32; 3], code: &PrefixCode) -> VarDctKernelArtifact {
    let mut fragment = BitWriter::new();
    let mut histogram = [0u32; RAW_SYMBOLS];
    let mut quantized_dc_yxb = [0i32; MAX_DC_SAMPLES];
    let mut raw_tokens = [0u32; MAX_DC_SAMPLES];
    let mut extra_bits = [0u32; MAX_DC_SAMPLES];
    for (channel, value) in q_yxb.into_iter().enumerate() {
        let index = channel * MAX_BLOCKS;
        quantized_dc_yxb[index] = value;
        let packed = if value >= 0 {
            (value as u32) << 1
        } else {
            ((-i64::from(value)) as u32) * 2 - 1
        };
        let nbits = if packed == 0 {
            0
        } else {
            31 - packed.leading_zeros()
        };
        let token = u32::from(packed != 0) + nbits;
        let extra = packed.saturating_sub(1u32 << nbits);
        code.write_raw(&mut fragment, token, nbits, extra).unwrap();
        histogram[token as usize] += 1;
        raw_tokens[index] = token;
        extra_bits[index] = extra;
    }
    let bit_len = fragment.bit_len() as u32;
    let bytes = fragment.into_bytes();
    let mut words = [0u32; MAX_DC_FRAGMENT_WORDS];
    for (index, byte) in bytes.into_iter().enumerate() {
        words[index / 4] |= u32::from(byte) << ((index % 4) * 8);
    }
    let mut strategy_map = [0u32; MAX_BLOCKS];
    strategy_map[0] = 1 << 8;
    let mut quantized_xyb = [0; 3 * MAX_COEFFICIENTS];
    quantized_xyb[MAX_COEFFICIENTS] = q_yxb[0];
    quantized_xyb[0] = q_yxb[1];
    quantized_xyb[2 * MAX_COEFFICIENTS] = q_yxb[2];
    VarDctKernelArtifact {
        strategy_map,
        quantized_dc_yxb,
        dc_raw_tokens: raw_tokens,
        dc_extra_bits: extra_bits,
        dc_fragment_words: words,
        dc_fragment_bit_len: bit_len,
        dc_sample_count: 3,
        block_count: 1,
        strategy: 0,
        raw_histogram: histogram,
        padding: [0; 9],
        forward_xyb_bits: [0; 3 * MAX_COEFFICIENTS],
        quantized_xyb,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use jxl::api::{
        JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
    };
    use jxl_gpu_formats::{ImageLayout, PitchLinearPlaneLayout};
    use jxl_gpu_protocol::Extent2d;
    use jxl_wgpu::{
        AdapterFingerprint, AutotuneProfile, KernelPolicy, TunedKernel, WgpuBackend,
        WgpuBackendConfig,
    };
    use wgpu::util::DeviceExt;

    use super::*;
    use crate::assemble_frame;

    fn decode_rgb8_sized(codestream: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut input = codestream;
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = loop {
            match decoder.process(&mut input, None).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };
        assert_eq!(decoder.basic_info().size, (width, height));
        decoder.set_pixel_format(JxlPixelFormat::rgb8(0));
        let mut frame = loop {
            match decoder.process(&mut input, None).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };
        let mut pixels = vec![0u8; width * height * 3];
        let mut buffers = [JxlOutputBuffer::new(&mut pixels, height, width * 3)];
        loop {
            match frame.process(&mut input, &mut buffers, None).unwrap() {
                ProcessingResult::Complete { .. } => break,
                ProcessingResult::NeedsMoreInput { fallback, .. } => frame = fallback,
            }
        }
        pixels
    }

    fn decode_rgb8(codestream: &[u8]) -> Vec<u8> {
        decode_rgb8_sized(codestream, 8, 8)
    }

    fn test_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>, wgpu::AdapterInfo)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jxl-wgpu VarDCT encoder test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        Some((Arc::new(device), Arc::new(queue), info))
    }

    fn test_context() -> Option<WgpuContext> {
        let (device, queue, _) = test_device()?;
        WgpuContext::new(device, queue).ok()
    }

    fn test_context_with_variants(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        info: &wgpu::AdapterInfo,
        variants: &[(&str, KernelVariant)],
    ) -> Option<WgpuContext> {
        let mut profile = AutotuneProfile::new(AdapterFingerprint::from_adapter_info(info));
        for &(kernel, variant) in variants {
            profile.record(TunedKernel::from_samples(kernel, variant, &[1])?);
        }
        let backend = WgpuBackend::from_device(
            device.as_ref().clone(),
            queue.as_ref().clone(),
            info.clone(),
            WgpuBackendConfig {
                enable_timestamps: false,
                kernel_policy: KernelPolicy::Profile(profile),
                ..WgpuBackendConfig::default()
            },
        )
        .ok()?;
        Some(WgpuContext::from_backend(&backend))
    }

    fn padded_rgb_source(context: &WgpuContext, pixels: &[[u8; 3]; 64]) -> BufferImageSource {
        padded_rgb_source_sized(context, 8, 8, pixels)
    }

    fn padded_rgb_source_sized(
        context: &WgpuContext,
        width: usize,
        height: usize,
        pixels: &[[u8; 3]],
    ) -> BufferImageSource {
        const OFFSET: u64 = 5;
        let row_bytes = (width * 3) as u64;
        let row_stride = row_bytes + 5;
        let extent = Extent2d::new(width as u32, height as u32);
        let allocation_size =
            align_up(OFFSET + row_stride * (height as u64 - 1) + row_bytes, 4).unwrap();
        let mut allocation = vec![0xa5; allocation_size as usize];
        for y in 0..height {
            let start = usize::try_from(OFFSET + row_stride * y as u64).unwrap();
            for x in 0..width {
                allocation[start + x * 3..start + x * 3 + 3]
                    .copy_from_slice(&pixels[y * width + x]);
            }
        }
        let layout = ImageLayout::from_planes(
            extent,
            VarDctColorEncoding::SrgbD65.pixel_format(),
            vec![PitchLinearPlaneLayout {
                plane_index: 0,
                offset: OFFSET,
                row_stride,
                sample_extent: extent,
                row_bytes,
            }],
        )
        .unwrap();
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu padded VarDCT RGB fixture"),
                contents: &allocation,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        BufferImageSource::new(buffer, layout).unwrap()
    }

    fn psnr(reference: &[[u8; 3]], actual: &[u8]) -> f64 {
        let squared_error = reference
            .iter()
            .flatten()
            .zip(actual)
            .map(|(&expected, &observed)| {
                let difference = f64::from(expected) - f64::from(observed);
                difference * difference
            })
            .sum::<f64>();
        if squared_error == 0.0 {
            return f64::INFINITY;
        }
        let mse = squared_error / actual.len() as f64;
        10.0 * (255.0 * 255.0 / mse).log10()
    }

    fn max_abs_error(left: &[u8], right: &[u8]) -> u8 {
        left.iter()
            .zip(right)
            .map(|(&left, &right)| left.abs_diff(right))
            .max()
            .unwrap_or(0)
    }

    fn oracle_directory() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("the test clock is after the Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "jxl-wgpu-vardct-oracle-{}-{nonce}",
            std::process::id()
        ))
    }

    fn ppm_bytes(pixels: &[[u8; 3]], width: usize, height: usize) -> Vec<u8> {
        let mut output = format!("P6\n{width} {height}\n255\n").into_bytes();
        output.extend(pixels.iter().flatten().copied());
        output
    }

    fn next_ppm_token<'a>(bytes: &'a [u8], cursor: &mut usize) -> &'a [u8] {
        loop {
            while bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
                *cursor += 1;
            }
            if bytes.get(*cursor) != Some(&b'#') {
                break;
            }
            while bytes.get(*cursor).is_some_and(|&byte| byte != b'\n') {
                *cursor += 1;
            }
        }
        let start = *cursor;
        while bytes
            .get(*cursor)
            .is_some_and(|byte| !byte.is_ascii_whitespace())
        {
            *cursor += 1;
        }
        &bytes[start..*cursor]
    }

    fn read_ppm_rgb8(path: &Path, width: usize, height: usize) -> Vec<u8> {
        let bytes = fs::read(path).unwrap();
        let mut cursor = 0usize;
        assert_eq!(next_ppm_token(&bytes, &mut cursor), b"P6");
        assert_eq!(
            next_ppm_token(&bytes, &mut cursor),
            width.to_string().as_bytes(),
        );
        assert_eq!(
            next_ppm_token(&bytes, &mut cursor),
            height.to_string().as_bytes(),
        );
        assert_eq!(next_ppm_token(&bytes, &mut cursor), b"255");
        assert!(bytes[cursor].is_ascii_whitespace());
        if bytes[cursor] == b'\r' && bytes.get(cursor + 1) == Some(&b'\n') {
            cursor += 2;
        } else {
            cursor += 1;
        }
        assert_eq!(bytes.len() - cursor, width * height * 3);
        bytes[cursor..].to_vec()
    }

    #[test]
    fn fixed_control_plane_decodes_as_standard_black_vardct() {
        let code = fixed_prefix_code().unwrap();
        assert!(prefix_entries(&code).iter().all(|entry| entry.bit_len > 0));
        let artifact = cpu_test_artifact([0, 0, 0], &code);
        let frame = assemble_frame(
            build_frame_packet(
                fixed_artifact_data(&artifact),
                &code,
                VarDctFrameLayout::single(VarDctStrategy::Dct8),
            )
            .unwrap(),
        )
        .unwrap();
        let mut codestream = image_header(8, 8).unwrap().bytes().to_vec();
        codestream.extend_from_slice(frame.bytes());
        let decoded = decode_rgb8(&codestream);
        assert_eq!(decoded, vec![0; 8 * 8 * 3]);
    }

    #[test]
    fn fixed_control_plane_accepts_nonzero_quantized_xyb_dc() {
        let code = fixed_prefix_code().unwrap();
        // libjxl's DCT8 oracle quantizes a solid red block close to these
        // Y/X/(B-Y) values with this profile's global DC scale.
        let artifact = cpu_test_artifact([332, 153, -6], &code);
        let frame = assemble_frame(
            build_frame_packet(
                fixed_artifact_data(&artifact),
                &code,
                VarDctFrameLayout::single(VarDctStrategy::Dct8),
            )
            .unwrap(),
        )
        .unwrap();
        let mut codestream = image_header(8, 8).unwrap().bytes().to_vec();
        codestream.extend_from_slice(frame.bytes());
        let decoded = decode_rgb8(&codestream);
        for pixel in decoded.chunks_exact(3) {
            assert!(pixel[0] > 240, "red={}", pixel[0]);
            assert!(pixel[1] < 16, "green={}", pixel[1]);
            assert!(pixel[2] < 16, "blue={}", pixel[2]);
        }
    }

    #[test]
    fn abi_records_are_pod_and_word_aligned() {
        fn assert_pod<T: bytemuck::Pod>() {}
        assert_pod::<GpuPrefixEntry>();
        assert_pod::<VarDctKernelParams>();
        assert_pod::<VarDctKernelArtifact>();
        assert_pod::<ScalableVarDctKernelParams>();
        assert_pod::<ScalableVarDctArtifactHeader>();
        assert_eq!(std::mem::size_of::<VarDctKernelParams>(), 256);
        assert_eq!(std::mem::size_of::<VarDctKernelArtifact>(), 25_600);
        assert_eq!(std::mem::align_of::<VarDctKernelArtifact>(), 4);
        assert_eq!(std::mem::size_of::<ScalableVarDctKernelParams>(), 256);
        assert_eq!(std::mem::size_of::<ScalableVarDctArtifactHeader>(), 256);
    }

    #[test]
    fn naga_validates_vardct_shaders() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("VarDCT WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("VarDCT WGSL validates");

        let module =
            naga::front::wgsl::parse_str(LARGE_SHADER).expect("scalable VarDCT WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("scalable VarDCT WGSL validates");
    }

    #[test]
    fn strategy_ir_uses_exact_standard_codestream_order() {
        for (id, strategy) in VarDctStrategy::ALL.into_iter().enumerate() {
            assert_eq!(usize::from(strategy.codestream_id()), id);
        }
        assert_eq!(VarDctStrategy::Dct16x8.block_extent(), (8, 16));
        assert_eq!(VarDctStrategy::Dct8x16.block_extent(), (16, 8));
        assert_eq!(VarDctStrategy::Dct256x128.block_extent(), (128, 256));
        assert_eq!(VarDctStrategy::Dct256x128.block_grid(), (16, 32));
        assert_eq!(VarDctStrategy::Dct128x256.block_extent(), (256, 128));
        assert_eq!(VarDctStrategy::Dct128x256.block_grid(), (32, 16));
        assert_eq!(VarDctStrategy::EXECUTABLE, VarDctStrategy::ALL);
        assert!(
            VarDctStrategy::EXECUTABLE
                .into_iter()
                .all(VarDctStrategy::is_executable)
        );
        assert!(VarDctStrategy::Hornuss.is_executable());
        assert!(VarDctStrategy::Dct64x64.is_executable());
    }

    #[test]
    fn artifact_gradient_validation_matches_wgsl_wrapping_without_panicking() {
        fn wgsl_gradient(top: i32, left: i32, top_left: i32) -> i32 {
            let wrapped = i32::from_ne_bytes(
                u32::from_ne_bytes(top.to_ne_bytes())
                    .wrapping_add(u32::from_ne_bytes(left.to_ne_bytes()))
                    .wrapping_sub(u32::from_ne_bytes(top_left.to_ne_bytes()))
                    .to_ne_bytes(),
            );
            wrapped.clamp(top.min(left), top.max(left))
        }

        for (actual, top, left, top_left) in [
            (i32::MAX, i32::MAX, i32::MAX, i32::MIN),
            (i32::MIN, i32::MIN, i32::MIN, i32::MAX),
            (0, i32::MIN, i32::MAX, 0),
            (i32::MAX, i32::MIN, 1, i32::MAX),
            (i32::MIN, -1, i32::MAX, i32::MIN),
        ] {
            let expected = wgsl_gradient(top, left, top_left);
            assert_eq!(clamped_gradient_i32(top, left, top_left), expected);
            let residual = gradient_residual_i32(actual, top, left, top_left);
            assert_eq!(residual, actual.wrapping_sub(expected));
            assert!(std::panic::catch_unwind(|| signed_token(residual)).is_ok());
        }
    }

    #[test]
    fn scalable_layout_is_checked_and_preserves_large_orientation() {
        let code = fixed_prefix_code().unwrap();
        let portrait = ScalableArtifactLayout::new(VarDctStrategy::Dct256x128, &code).unwrap();
        let landscape = ScalableArtifactLayout::new(VarDctStrategy::Dct128x256, &code).unwrap();
        let largest = ScalableArtifactLayout::new(VarDctStrategy::Dct256x256, &code).unwrap();
        assert_eq!(portrait.strategy_len, 16 * 32);
        assert_eq!(landscape.strategy_len, 32 * 16);
        assert_eq!(portrait.dc_len, 3 * 16 * 32);
        assert_eq!(portrait, landscape);
        assert_eq!(portrait.strategy_offset, SCALABLE_HEADER_WORDS);
        assert_eq!(
            portrait.strategy_offset % SCALABLE_SECTION_ALIGNMENT_WORDS,
            0
        );
        assert_eq!(portrait.dc_offset % SCALABLE_SECTION_ALIGNMENT_WORDS, 0);
        assert_eq!(portrait.token_offset % SCALABLE_SECTION_ALIGNMENT_WORDS, 0);
        assert_eq!(portrait.extra_offset % SCALABLE_SECTION_ALIGNMENT_WORDS, 0);
        assert_eq!(
            portrait.fragment_offset % SCALABLE_SECTION_ALIGNMENT_WORDS,
            0
        );
        assert_eq!(
            portrait.artifact_words % SCALABLE_SECTION_ALIGNMENT_WORDS,
            0
        );
        assert!(portrait.fragment_max_bits > 0);
        assert_eq!(portrait.artifact_bytes(), 25_600);
        assert_eq!(largest.strategy_len, 1_024);
        assert_eq!(largest.dc_len, 3_072);
        assert_eq!(largest.fragment_max_bits, 76_800);
        assert_eq!(largest.artifact_bytes(), 50_944);
    }

    #[test]
    fn gpu_profile_encodes_exact_black_from_padded_rgb() {
        let Some(context) = test_context() else {
            return;
        };
        let encoder = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct8).unwrap();
        let source = padded_rgb_source(&context, &[[0, 0, 0]; 64]);
        let plan = encoder.memory_plan(&source).unwrap();
        assert_eq!(plan.kernel_layout, VarDctKernelLayout::Bounded);
        assert_eq!(plan.source_binding_bytes, 232);
        assert_eq!(plan.parameter_storage_bytes, 256);
        assert_eq!(plan.artifact_storage_bytes, 25_600);
        assert_eq!(plan.readback_bytes, 25_600);
        assert_eq!(plan.owned_bytes_per_job, 51_456);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);

        let codestream = encoder.encode(source).unwrap();
        assert_eq!(decode_rgb8(&codestream), vec![0; 8 * 8 * 3]);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }

    #[test]
    fn every_linear_workgroup_produces_identical_bounded_and_scalable_codestreams() {
        let Some((device, queue, info)) = test_device() else {
            return;
        };
        let default_context = WgpuContext::new(Arc::clone(&device), Arc::clone(&queue)).unwrap();

        let mut bounded_pixels = [[0u8; 3]; 64];
        for y in 0..8usize {
            for x in 0..8usize {
                bounded_pixels[y * 8 + x] = [
                    (x * 29 + y * 5) as u8,
                    (y * 31 + x * 3) as u8,
                    ((x + y) * 17) as u8,
                ];
            }
        }
        let default_bounded_encoder =
            VarDctEncoder::new(default_context.clone(), VarDctStrategy::Dct8).unwrap();
        let default_bounded_source = padded_rgb_source(&default_context, &bounded_pixels);
        assert_eq!(
            default_bounded_encoder
                .memory_plan(&default_bounded_source)
                .unwrap()
                .kernel_layout,
            VarDctKernelLayout::Bounded,
        );
        let default_bounded = default_bounded_encoder
            .encode(default_bounded_source)
            .unwrap();

        let scalable_width = 32usize;
        let scalable_height = 64usize;
        let scalable_pixels = (0..scalable_height)
            .flat_map(|y| {
                (0..scalable_width).map(move |x| {
                    [
                        (x * 255 / (scalable_width - 1)) as u8,
                        (y * 255 / (scalable_height - 1)) as u8,
                        ((x * 11 + y * 7) & 0xff) as u8,
                    ]
                })
            })
            .collect::<Vec<_>>();
        let default_scalable_encoder =
            VarDctEncoder::new(default_context.clone(), VarDctStrategy::Dct64x32).unwrap();
        let default_scalable_source = padded_rgb_source_sized(
            &default_context,
            scalable_width,
            scalable_height,
            &scalable_pixels,
        );
        assert_eq!(
            default_scalable_encoder
                .memory_plan(&default_scalable_source)
                .unwrap()
                .kernel_layout,
            VarDctKernelLayout::Scalable,
        );
        let default_scalable = default_scalable_encoder
            .encode(default_scalable_source)
            .unwrap();

        for variant in [
            KernelVariant::Scalar,
            KernelVariant::Lanes32,
            KernelVariant::Lanes64,
            KernelVariant::Lanes128,
            KernelVariant::Lanes256,
        ] {
            let context = test_context_with_variants(
                &device,
                &queue,
                &info,
                &[
                    (BOUNDED_KERNEL_KEY, variant),
                    (SCALABLE_QUANTIZE_KERNEL_KEY, variant),
                ],
            )
            .unwrap();

            let bounded = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct8).unwrap();
            assert_eq!(bounded.workgroup_variant(), variant);
            assert_eq!(
                bounded
                    .encode(padded_rgb_source(&context, &bounded_pixels))
                    .unwrap(),
                default_bounded,
                "bounded variant={variant:?}",
            );

            let scalable = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct64x32).unwrap();
            assert_eq!(scalable.workgroup_variant(), variant);
            assert_eq!(
                scalable
                    .encode(padded_rgb_source_sized(
                        &context,
                        scalable_width,
                        scalable_height,
                        &scalable_pixels,
                    ))
                    .unwrap(),
                default_scalable,
                "scalable variant={variant:?}",
            );
        }

        let incompatible = test_context_with_variants(
            &device,
            &queue,
            &info,
            &[(BOUNDED_KERNEL_KEY, KernelVariant::Tile8x8)],
        )
        .unwrap();
        assert!(matches!(
            VarDctEncoder::new(incompatible, VarDctStrategy::Dct8),
            Err(EncodeError::KernelPolicy(jxl_wgpu::Error::Unsupported(_)))
        ));
    }

    #[test]
    fn tiled_dct8_emits_multiple_ac_groups_for_odd_black_extent() {
        let Some(context) = test_context() else {
            return;
        };
        let width = 257usize;
        let height = 17usize;
        let pixels = vec![[0u8; 3]; width * height];
        let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
        let source = padded_rgb_source_sized(&context, width, height, &pixels);
        let async_source = source.clone();
        let plan = encoder.memory_plan(&source).unwrap();
        let grid = encoder.grid(&source).unwrap();
        assert_eq!(plan.kernel_layout, VarDctKernelLayout::TiledDct8);
        assert_eq!(plan.parameter_storage_bytes, 256);
        assert_eq!((grid.block_columns, grid.block_rows), (33, 3));
        assert_eq!(grid.block_count().unwrap(), 99);
        assert_eq!((grid.ac_group_columns, grid.ac_group_rows), (2, 1));
        assert_eq!(grid.ac_group_count().unwrap(), 2);
        assert_eq!(grid.toc_entries().unwrap(), 5);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);

        let codestream = encoder.encode(source).unwrap();
        let inventory =
            jxl_gpu_bitstream::parse(&codestream, jxl_gpu_bitstream::ParseLimits::default())
                .unwrap()
                .codestream_inventory(jxl_gpu_bitstream::InventoryLimits::default())
                .unwrap();
        let frame = &inventory.frames[0];
        assert_eq!(frame.group_count, 2);
        assert_eq!(frame.low_frequency_group_count, 1);
        assert_eq!(frame.sections.len(), 5);
        assert_eq!(
            frame
                .sections
                .iter()
                .map(|section| section.kind)
                .collect::<Vec<_>>(),
            vec![
                jxl_gpu_bitstream::FrameSectionKind::LowFrequencyGlobal,
                jxl_gpu_bitstream::FrameSectionKind::LowFrequencyGroup { group_index: 0 },
                jxl_gpu_bitstream::FrameSectionKind::HighFrequencyGlobal,
                jxl_gpu_bitstream::FrameSectionKind::PassGroup {
                    pass_index: 0,
                    group_index: 0,
                },
                jxl_gpu_bitstream::FrameSectionKind::PassGroup {
                    pass_index: 0,
                    group_index: 1,
                },
            ]
        );
        assert_eq!(
            decode_rgb8_sized(&codestream, width, height),
            vec![0; width * height * 3]
        );
        let async_codestream = pollster::block_on(encoder.submit(async_source).unwrap()).unwrap();
        assert_eq!(async_codestream, codestream);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }

    #[test]
    fn tiled_dct8_preserves_asymmetric_solid_and_lf_gradient() {
        let Some(context) = test_context() else {
            return;
        };
        let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
        for (width, height, solid) in [
            (513usize, 259usize, [255u8, 0, 0]),
            (768usize, 513usize, [0u8, 255, 0]),
        ] {
            let solid_pixels = vec![solid; width * height];
            let solid_stream = encoder
                .encode(padded_rgb_source_sized(
                    &context,
                    width,
                    height,
                    &solid_pixels,
                ))
                .unwrap();
            let decoded_solid = decode_rgb8_sized(&solid_stream, width, height);
            let solid_quality = psnr(&solid_pixels, &decoded_solid);
            assert!(
                solid_quality > 30.0,
                "{width}x{height} solid PSNR={solid_quality}",
            );

            let gradient = (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        [
                            (x * 255 / (width - 1)) as u8,
                            (y * 255 / (height - 1)) as u8,
                            ((x + y) * 255 / (width + height - 2)) as u8,
                        ]
                    })
                })
                .collect::<Vec<_>>();
            let gradient_stream = encoder
                .encode(padded_rgb_source_sized(&context, width, height, &gradient))
                .unwrap();
            let decoded_gradient = decode_rgb8_sized(&gradient_stream, width, height);
            let gradient_quality = psnr(&gradient, &decoded_gradient);
            assert!(
                gradient_quality > 9.0,
                "{width}x{height} LF gradient PSNR={gradient_quality}",
            );
        }
    }

    #[test]
    fn tiled_dct8_reports_lf_group_boundary_as_a_typed_error() {
        let Some(context) = test_context() else {
            return;
        };
        let width = 2_049usize;
        let height = 1usize;
        let pixels = vec![[0u8; 3]; width * height];
        let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
        let source = padded_rgb_source_sized(&context, width, height, &pixels);
        assert!(matches!(
            encoder.memory_plan(&source),
            Err(EncodeError::Unsupported(
                UnsupportedFeature::TiledVarDctLfGroups {
                    width: 2_049,
                    height: 1,
                    max_dimension: 2_048,
                }
            ))
        ));
    }

    #[test]
    fn tiled_dct8_reports_fused_single_group_ambiguity_as_a_typed_error() {
        assert!(matches!(
            TiledVarDctGrid::new(17, 9),
            Err(EncodeError::Unsupported(
                UnsupportedFeature::TiledVarDctSingleAcGroup {
                    width: 17,
                    height: 9,
                    group_dimension: 256,
                }
            ))
        ));
    }

    #[test]
    fn abandoned_tiled_job_holds_and_releases_its_exact_budget() {
        let Some(base_context) = test_context() else {
            return;
        };
        let width = 513usize;
        let height = 259usize;
        let pixels = vec![[0u8; 3]; width * height];
        let provisional = TiledVarDctEncoder::new(base_context.clone()).unwrap();
        let provisional_source = padded_rgb_source_sized(&base_context, width, height, &pixels);
        let plan = provisional.memory_plan(&provisional_source).unwrap();
        assert_eq!(plan.kernel_layout, VarDctKernelLayout::TiledDct8);
        assert_eq!(
            plan.owned_bytes_per_job,
            256 + 2 * plan.artifact_storage_bytes
        );

        let limited_context = WgpuContext::with_memory_budget(
            Arc::new(base_context.device().clone()),
            Arc::new(base_context.queue().clone()),
            NonZeroU64::new(plan.owned_bytes_per_job).unwrap(),
        )
        .unwrap();
        let encoder = TiledVarDctEncoder::new(limited_context.clone()).unwrap();
        let source = padded_rgb_source_sized(&limited_context, width, height, &pixels);
        let abandoned = encoder.submit(source.clone()).unwrap();
        assert_eq!(
            encoder.in_flight_memory_stats().reserved_bytes,
            plan.owned_bytes_per_job
        );
        assert!(matches!(
            encoder.submit(source),
            Err(EncodeError::MemoryBackpressure(_))
        ));
        drop(abandoned);

        let fence_commands =
            limited_context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("abandoned tiled VarDCT completion fence"),
                });
        let fence = limited_context.queue().submit([fence_commands.finish()]);
        limited_context
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(fence),
                timeout: None,
            })
            .expect("abandoned tiled VarDCT work completes");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while encoder.in_flight_memory_stats().reserved_bytes != 0
            && std::time::Instant::now() < deadline
        {
            limited_context
                .device()
                .poll(wgpu::PollType::Poll)
                .expect("drive abandoned tiled VarDCT map callback");
            std::thread::yield_now();
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }

    #[test]
    fn abandoned_scalable_job_retains_and_releases_its_exact_budget() {
        let Some(base_context) = test_context() else {
            return;
        };
        let strategy = VarDctStrategy::Dct256x256;
        let pixels = vec![[0u8; 3]; 256 * 256];
        let provisional = VarDctEncoder::new(base_context.clone(), strategy).unwrap();
        let provisional_source = padded_rgb_source_sized(&base_context, 256, 256, &pixels);
        let plan = provisional.memory_plan(&provisional_source).unwrap();
        let limited_context = WgpuContext::with_memory_budget(
            Arc::new(base_context.device().clone()),
            Arc::new(base_context.queue().clone()),
            NonZeroU64::new(plan.owned_bytes_per_job).unwrap(),
        )
        .unwrap();
        let encoder = VarDctEncoder::new(limited_context.clone(), strategy).unwrap();
        let source = padded_rgb_source_sized(&limited_context, 256, 256, &pixels);

        let abandoned = encoder.submit(source.clone()).unwrap();
        assert_eq!(
            encoder.in_flight_memory_stats().reserved_bytes,
            plan.owned_bytes_per_job
        );
        assert!(matches!(
            encoder.submit(source),
            Err(EncodeError::MemoryBackpressure(_))
        ));
        drop(abandoned);

        let fence_commands =
            limited_context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("abandoned scalable VarDCT completion fence"),
                });
        let fence = limited_context.queue().submit([fence_commands.finish()]);
        limited_context
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(fence),
                timeout: None,
            })
            .expect("abandoned scalable VarDCT work completes");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while encoder.in_flight_memory_stats().reserved_bytes != 0
            && std::time::Instant::now() < deadline
        {
            limited_context
                .device()
                .poll(wgpu::PollType::Poll)
                .expect("drive abandoned scalable VarDCT map callback");
            std::thread::yield_now();
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }

    #[test]
    fn every_executable_strategy_emits_a_standard_black_codestream() {
        let Some(context) = test_context() else {
            return;
        };
        for strategy in VarDctStrategy::EXECUTABLE {
            let (width, height) = strategy.block_extent();
            let width = usize::from(width);
            let height = usize::from(height);
            let pixels = vec![[0, 0, 0]; width * height];
            let encoder = VarDctEncoder::new(context.clone(), strategy).unwrap();
            let source = padded_rgb_source_sized(&context, width, height, &pixels);
            let plan = encoder.memory_plan(&source).unwrap();
            if strategy.uses_scalable_kernel() {
                let layout =
                    ScalableArtifactLayout::new(strategy, &fixed_prefix_code().unwrap()).unwrap();
                assert_eq!(plan.kernel_layout, VarDctKernelLayout::Scalable);
                assert_eq!(plan.parameter_storage_bytes, 256);
                assert_eq!(plan.artifact_storage_bytes, layout.artifact_bytes());
                assert_eq!(plan.readback_bytes, layout.artifact_bytes());
                assert_eq!(plan.owned_bytes_per_job, 256 + 2 * layout.artifact_bytes());
            } else {
                assert_eq!(plan.kernel_layout, VarDctKernelLayout::Bounded);
            }
            let codestream = encoder.encode(source).unwrap();
            assert_eq!(
                decode_rgb8_sized(&codestream, width, height),
                vec![0; width * height * 3],
                "strategy={strategy:?}",
            );
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        }
    }

    #[test]
    fn every_executable_strategy_preserves_solid_color_and_lf_gradient() {
        let Some(context) = test_context() else {
            return;
        };
        for strategy in VarDctStrategy::EXECUTABLE {
            let (width, height) = strategy.block_extent();
            let width = usize::from(width);
            let height = usize::from(height);
            let encoder = VarDctEncoder::new(context.clone(), strategy).unwrap();

            let red = vec![[255, 0, 0]; width * height];
            let red_stream = encoder
                .encode(padded_rgb_source_sized(&context, width, height, &red))
                .unwrap();
            let decoded_red = decode_rgb8_sized(&red_stream, width, height);
            let red_quality = psnr(&red, &decoded_red);
            assert!(
                red_quality > 30.0,
                "strategy={strategy:?}, PSNR={red_quality}"
            );

            let mut gradient = vec![[0u8; 3]; width * height];
            for y in 0..height {
                for x in 0..width {
                    gradient[y * width + x] = [
                        (x * 255 / (width - 1)) as u8,
                        (y * 255 / (height - 1)) as u8,
                        ((x + y) * 255 / (width + height - 2)) as u8,
                    ];
                }
            }
            let gradient_stream = encoder
                .encode(padded_rgb_source_sized(&context, width, height, &gradient))
                .unwrap();
            let decoded_gradient = decode_rgb8_sized(&gradient_stream, width, height);
            let gradient_quality = psnr(&gradient, &decoded_gradient);
            assert!(
                gradient_quality > 9.0,
                "strategy={strategy:?}, PSNR={gradient_quality}",
            );
        }
    }

    #[test]
    fn gpu_profile_is_same_device_deterministic_and_bounded_quality() {
        let Some(context) = test_context() else {
            return;
        };
        let encoder = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct8).unwrap();
        let mut fixture = [[0u8; 3]; 64];
        for y in 0..8usize {
            for x in 0..8usize {
                fixture[y * 8 + x] = [
                    (x * 31 + y * 3) as u8,
                    (y * 31 + x * 3) as u8,
                    ((x + y) * 16) as u8,
                ];
            }
        }
        let first_source = padded_rgb_source(&context, &fixture);
        let second_source = first_source.clone();
        let first = encoder.encode(first_source).unwrap();
        let second = pollster::block_on(encoder.submit(second_source).unwrap()).unwrap();
        assert_eq!(first, second);
        let decoded = decode_rgb8(&first);
        let quality = psnr(&fixture, &decoded);
        assert!(quality > 9.0, "PSNR={quality}");

        let red = [[255, 0, 0]; 64];
        let red_stream = encoder.encode(padded_rgb_source(&context, &red)).unwrap();
        let decoded_red = decode_rgb8(&red_stream);
        let red_quality = psnr(&red, &decoded_red);
        assert!(red_quality > 30.0, "solid-red PSNR={red_quality}");
        for pixel in decoded_red.chunks_exact(3) {
            assert!(pixel[0] > 248);
            assert!(pixel[1] < 8);
            assert!(pixel[2] < 8);
        }
    }

    #[test]
    fn libjxl_cli_and_rust_oracles_agree_on_gpu_codestream() {
        const CJXL: &str = "/opt/homebrew/bin/cjxl";
        const DJXL: &str = "/opt/homebrew/bin/djxl";
        if !Path::new(CJXL).is_file() || !Path::new(DJXL).is_file() {
            return;
        }
        let Some(context) = test_context() else {
            return;
        };
        let directory = oracle_directory();
        fs::create_dir_all(&directory).unwrap();
        for strategy in VarDctStrategy::EXECUTABLE {
            let (width, height) = strategy.block_extent();
            let width = usize::from(width);
            let height = usize::from(height);
            let mut fixture = vec![[0u8; 3]; width * height];
            for y in 0..height {
                for x in 0..width {
                    fixture[y * width + x] = [
                        (x * 255 / (width - 1)) as u8,
                        (y * 255 / (height - 1)) as u8,
                        ((x + y) * 255 / (width + height - 2)) as u8,
                    ];
                }
            }
            let encoder = VarDctEncoder::new(context.clone(), strategy).unwrap();
            let codestream = encoder
                .encode(padded_rgb_source_sized(&context, width, height, &fixture))
                .unwrap();
            let rust_pixels = decode_rgb8_sized(&codestream, width, height);

            let stem = strategy.codestream_id().to_string();
            let gpu_path = directory.join(format!("gpu-{stem}.jxl"));
            let gpu_ppm_path = directory.join(format!("gpu-{stem}.ppm"));
            let source_path = directory.join(format!("source-{stem}.ppm"));
            let reference_path = directory.join(format!("reference-{stem}.jxl"));
            let reference_ppm_path = directory.join(format!("reference-{stem}.ppm"));
            fs::write(&gpu_path, &codestream).unwrap();
            fs::write(&source_path, ppm_bytes(&fixture, width, height)).unwrap();

            let gpu_decode = Command::new(DJXL)
                .arg(&gpu_path)
                .arg(&gpu_ppm_path)
                .args(["--num_threads=0", "--quiet"])
                .status()
                .unwrap();
            assert!(gpu_decode.success(), "strategy={strategy:?}");
            let libjxl_pixels = read_ppm_rgb8(&gpu_ppm_path, width, height);
            assert!(
                max_abs_error(&libjxl_pixels, &rust_pixels) <= 1,
                "strategy={strategy:?}",
            );

            let reference_encode = Command::new(CJXL)
                .arg(&source_path)
                .arg(&reference_path)
                .args([
                    "-d",
                    "25",
                    "-e",
                    "1",
                    "-m",
                    "0",
                    "--progressive_dc=0",
                    "--resampling=1",
                    "--epf=0",
                    "--gaborish=0",
                    "--container=0",
                    "--num_threads=0",
                    "--quiet",
                ])
                .status()
                .unwrap();
            assert!(reference_encode.success(), "strategy={strategy:?}");
            let reference_codestream = fs::read(&reference_path).unwrap();
            let rust_reference = decode_rgb8_sized(&reference_codestream, width, height);
            let reference_decode = Command::new(DJXL)
                .arg(&reference_path)
                .arg(&reference_ppm_path)
                .args(["--num_threads=0", "--quiet"])
                .status()
                .unwrap();
            assert!(reference_decode.success(), "strategy={strategy:?}");
            let libjxl_reference = read_ppm_rgb8(&reference_ppm_path, width, height);
            assert!(
                max_abs_error(&libjxl_reference, &rust_reference) <= 1,
                "strategy={strategy:?}",
            );
            assert!(psnr(&fixture, &rust_pixels) > 9.0);
            assert!(psnr(&fixture, &rust_reference) > 9.0);
        }

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn tiled_dct8_rust_jxl_djxl_and_cjxl_oracles_cover_group_edges() {
        const CJXL: &str = "/opt/homebrew/bin/cjxl";
        const DJXL: &str = "/opt/homebrew/bin/djxl";
        if !Path::new(CJXL).is_file() || !Path::new(DJXL).is_file() {
            return;
        }
        let Some(context) = test_context() else {
            return;
        };
        let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
        let directory = oracle_directory();
        fs::create_dir_all(&directory).unwrap();

        for (case, width, height) in [
            ("odd-group-edge", 257usize, 17usize),
            ("larger-asymmetric", 768usize, 513usize),
        ] {
            let fixture = (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        [
                            (x * 255 / (width - 1)) as u8,
                            (y * 255 / (height - 1)) as u8,
                            ((x + y) * 255 / (width + height - 2)) as u8,
                        ]
                    })
                })
                .collect::<Vec<_>>();
            let codestream = encoder
                .encode(padded_rgb_source_sized(&context, width, height, &fixture))
                .unwrap();
            let rust_pixels = decode_rgb8_sized(&codestream, width, height);
            assert!(psnr(&fixture, &rust_pixels) > 9.0, "case={case}");

            let source_path = directory.join(format!("source-{case}.ppm"));
            let gpu_path = directory.join(format!("gpu-{case}.jxl"));
            let gpu_ppm_path = directory.join(format!("gpu-{case}.ppm"));
            let reference_path = directory.join(format!("reference-{case}.jxl"));
            let reference_ppm_path = directory.join(format!("reference-{case}.ppm"));
            fs::write(&source_path, ppm_bytes(&fixture, width, height)).unwrap();
            fs::write(&gpu_path, &codestream).unwrap();

            let gpu_decode = Command::new(DJXL)
                .arg(&gpu_path)
                .arg(&gpu_ppm_path)
                .args(["--num_threads=0", "--quiet"])
                .status()
                .unwrap();
            assert!(gpu_decode.success(), "case={case}");
            let libjxl_pixels = read_ppm_rgb8(&gpu_ppm_path, width, height);
            assert!(
                max_abs_error(&libjxl_pixels, &rust_pixels) <= 1,
                "case={case}",
            );

            let reference_encode = Command::new(CJXL)
                .arg(&source_path)
                .arg(&reference_path)
                .args([
                    "-d",
                    "25",
                    "-e",
                    "1",
                    "-m",
                    "0",
                    "--progressive_dc=0",
                    "--resampling=1",
                    "--epf=0",
                    "--gaborish=0",
                    "--container=0",
                    "--num_threads=0",
                    "--quiet",
                ])
                .status()
                .unwrap();
            assert!(reference_encode.success(), "case={case}");
            let reference_codestream = fs::read(&reference_path).unwrap();
            let rust_reference = decode_rgb8_sized(&reference_codestream, width, height);
            let reference_decode = Command::new(DJXL)
                .arg(&reference_path)
                .arg(&reference_ppm_path)
                .args(["--num_threads=0", "--quiet"])
                .status()
                .unwrap();
            assert!(reference_decode.success(), "case={case}");
            let libjxl_reference = read_ppm_rgb8(&reference_ppm_path, width, height);
            assert!(
                max_abs_error(&libjxl_reference, &rust_reference) <= 1,
                "case={case}",
            );
            assert!(psnr(&fixture, &rust_reference) > 9.0, "case={case}");
        }

        fs::remove_dir_all(directory).unwrap();
    }
}
