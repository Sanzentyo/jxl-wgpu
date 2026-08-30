//! Standard VarDCT still-image encoder frontend.
//!
//! The bounded frontend encodes one regular transform whose extent is also the
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
use jxl_wgpu::MemoryPermit;

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
const PROFILE_DISTANCE: f32 = 25.0;

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

/// Typed JPEG XL VarDCT strategy identifier.
///
/// The enum covers the complete standard strategy alphabet. Use
/// [`Self::EXECUTABLE`] to enumerate the regular transforms implemented by the
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

    /// Regular strategies implemented end-to-end by this encoder.
    pub const EXECUTABLE: [Self; 7] = [
        Self::Dct8,
        Self::Dct16x8,
        Self::Dct8x16,
        Self::Dct16x16,
        Self::Dct32x32,
        Self::Dct32x16,
        Self::Dct16x32,
    ];

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
        matches!(
            self,
            Self::Dct8
                | Self::Dct16x8
                | Self::Dct8x16
                | Self::Dct16x16
                | Self::Dct32x32
                | Self::Dct32x16
                | Self::Dct16x32
        )
    }

    const fn block_grid(self) -> (u32, u32) {
        let (width, height) = self.block_extent();
        (width as u32 / 8, height as u32 / 8)
    }
}

/// Explicit bounded allocations retained by one in-flight VarDCT submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctMemoryPlan {
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

const _: () = {
    assert!(std::mem::size_of::<GpuPrefixEntry>() == 8);
    assert!(std::mem::align_of::<GpuPrefixEntry>() == 4);
    assert!(std::mem::size_of::<VarDctKernelParams>() == 256);
    assert!(std::mem::align_of::<VarDctKernelParams>() == 4);
    assert!(std::mem::size_of::<VarDctKernelArtifact>() == 25_600);
    assert!(std::mem::align_of::<VarDctKernelArtifact>() == 4);
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

fn append_gpu_dc_fragment(
    output: &mut BitWriter,
    artifact: &VarDctKernelArtifact,
) -> Result<(), EncodeError> {
    let bit_len = usize::try_from(artifact.dc_fragment_bit_len)
        .map_err(|_| EncodeError::Backend("GPU DC fragment length overflow".into()))?;
    if bit_len > MAX_DC_FRAGMENT_WORDS * 32 {
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
    artifact: &VarDctKernelArtifact,
) -> Result<(), EncodeError> {
    output.write_bits(0, 2)?; // no extra LF precision
    write_local_modular_header(output)?;
    append_gpu_dc_fragment(output, artifact)?;

    // One GPU-selected regular transform, no chroma-from-luma correction,
    // fixed HF multiplier, and zero EPF sharpness. Source-dependent DC entropy
    // was already packed by the GPU; these values describe its control map.
    // HfMetadata first stores `number of first blocks - 1`; this profile has
    // exactly one first block, while the field width grows with the DC grid.
    let first_block_bits = artifact.block_count.next_power_of_two().trailing_zeros() as u8;
    output.write_bits(0, first_block_bits)?;
    write_local_modular_header(output)?;
    write_unsigned_token(output, code, 0)?;
    write_unsigned_token(output, code, 0)?;
    write_unsigned_token(output, code, pack_signed_control(artifact.strategy as i32))?;
    write_unsigned_token(
        output,
        code,
        pack_signed_control((HF_MUL - 1) - artifact.strategy as i32),
    )?;
    for _ in 0..artifact.block_count {
        write_unsigned_token(output, code, 0)?;
    }
    Ok(())
}

fn write_hf_global(output: &mut BitWriter) -> Result<(), EncodeError> {
    // Default dequant matrices, natural coefficient order, and a single-token
    // HF decoder whose only symbol is zero. All AC coefficients are zero in
    // this LF-first regular-transform profile, so the pass group has no
    // payload bits. This is the 18-bit normative bundle encoding of those
    // fields, written LSB-first by BitWriter.
    output.write_bits(0x2495, 18)?;
    Ok(())
}

fn build_frame_packet(
    artifact: &VarDctKernelArtifact,
    code: &PrefixCode,
) -> Result<FramePacketSet, EncodeError> {
    let mut group = BitWriter::new();
    write_lf_global(&mut group, code)?;
    write_lf_group(&mut group, code, artifact)?;
    write_hf_global(&mut group)?;
    group.align_to_byte()?;
    Ok(FramePacketSet::new(
        frame_header()?,
        FrameGroupLayout::new(1, 1, 1)?,
        [GroupPacket::new(
            GroupPacketKind::Single,
            group.into_bytes(),
        )],
    )?)
}

#[derive(Clone, Copy, Debug)]
struct VarDctDispatchPlan {
    source_binding_offset: u64,
    source_binding_size: NonZeroU64,
    params: VarDctKernelParams,
    memory: VarDctMemoryPlan,
}

/// GPU backend for one bounded standard regular-transform still-image profile.
///
/// The source extent must equal the selected transform extent. The backend
/// emits a standards-compliant VarDCT frame and does not route pixels or
/// coefficients through a CPU codec.
pub struct VarDctBackend {
    pipeline: Arc<wgpu::ComputePipeline>,
    code: PrefixCode,
    strategy: VarDctStrategy,
    capabilities: EncoderCapabilities,
    max_storage_binding_size: u64,
    storage_offset_alignment: u64,
}

impl VarDctBackend {
    /// Creates a standard regular-transform backend and its compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns an encoder error if the fixed standard entropy tree cannot be
    /// represented by the JPEG XL prefix-code writer.
    pub fn new(context: &WgpuContext, strategy: VarDctStrategy) -> Result<Self, EncodeError> {
        if !strategy.is_executable() {
            return Err(EncodeError::InvalidConfiguration(
                "the selected VarDCT strategy is not implemented by the GPU kernel",
            ));
        }
        let code = fixed_prefix_code()?;
        let module = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("jxl-wgpu VarDCT forward-transform kernel"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = Arc::new(context.device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu VarDCT regular-transform pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("encode"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));
        let distance = profile_distance();
        let limits = context.device().limits();
        Ok(Self {
            pipeline,
            code,
            strategy,
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
            storage_offset_alignment: u64::from(limits.min_storage_buffer_offset_alignment),
        })
    }

    /// Computes the exact memory admission and source binding before a job is
    /// submitted.
    pub fn memory_plan(&self, source: &BufferImageSource) -> Result<VarDctMemoryPlan, EncodeError> {
        Ok(self.dispatch_plan(source)?.memory)
    }

    fn dispatch_plan(&self, source: &BufferImageSource) -> Result<VarDctDispatchPlan, EncodeError> {
        let extent = source.layout.extent;
        let (expected_width, expected_height) = self.strategy.block_extent();
        if extent.width != u32::from(expected_width) || extent.height != u32::from(expected_height)
        {
            return Err(EncodeError::InvalidSource(
                "the VarDCT source extent must equal the selected transform extent",
            ));
        }
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
        let params = VarDctKernelParams {
            row_stride,
            byte_offset: u32::try_from(relative_offset).map_err(|_| {
                EncodeError::InvalidSource("VarDCT source offset exceeds the WGSL u32 space")
            })?,
            width: extent.width,
            height: extent.height,
            blocks_x: extent.width / 8,
            blocks_y: extent.height / 8,
            strategy: u32::from(self.strategy.codestream_id()),
            global_scale: GLOBAL_SCALE,
            quant_lf: QUANT_LF,
            raw_prefix: prefix_entries(&self.code),
            padding: [0; 17],
        };
        Ok(VarDctDispatchPlan {
            source_binding_offset,
            source_binding_size,
            params,
            memory: VarDctMemoryPlan::fixed(source_binding_bytes),
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

fn profile_distance() -> PerceptualDistance {
    PerceptualDistance::new(PROFILE_DISTANCE)
        .expect("the fixed VarDCT distance is within the public validated range")
}

fn validate_vardct_request(
    request: &FrameEncodeRequest,
    strategy: VarDctStrategy,
) -> Result<(), EncodeError> {
    let (width, height) = strategy.block_extent();
    if request.frame_index != FrameIndex::new(0)
        || !request.is_last
        || request.animation != AnimationHeader::Still
        || request.canvas_width != u32::from(width)
        || request.canvas_height != u32::from(height)
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
        validate_vardct_request(request, self.strategy)?;
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
        context
            .queue()
            .write_buffer(&parameters, 0, bytemuck::bytes_of(&plan.params));

        let source_binding = wgpu::BufferBinding {
            buffer: &source.buffer,
            offset: plan.source_binding_offset,
            size: Some(plan.source_binding_size),
        };
        let params_binding_size = NonZeroU64::new(plan.memory.parameter_storage_bytes)
            .expect("the VarDCT parameter ABI is non-empty");
        let artifact_binding_size = NonZeroU64::new(plan.memory.artifact_storage_bytes)
            .expect("the VarDCT artifact ABI is non-empty");
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu VarDCT bindings"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(source_binding),
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
            });
        let mut commands =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu VarDCT encode"),
                });
        commands.clear_buffer(&artifact, 0, None);
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu VarDCT forward transform and tokenization"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
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
            strategy: self.strategy,
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
pub struct VarDctJob {
    lifetime: Option<Arc<VarDctJobLifetime>>,
    completion: Arc<VarDctMapCompletion>,
    code: PrefixCode,
    strategy: VarDctStrategy,
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
            let artifact = bytemuck::try_from_bytes::<VarDctKernelArtifact>(&mapped)
                .map_err(|_| BackendError::InvalidArtifact("VarDCT ABI size or alignment"))?;
            validate_artifact(artifact, &self.code, self.strategy)?;
            Ok(GpuFrameArtifacts {
                frame_index: self.frame_index,
                is_last: self.is_last,
                packets: build_frame_packet(artifact, &self.code)?,
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

fn validate_artifact(
    artifact: &VarDctKernelArtifact,
    code: &PrefixCode,
    strategy: VarDctStrategy,
) -> Result<(), BackendError> {
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
                artifact.quantized_dc_yxb[base + block] - clamped_gradient_i32(top, left, top_left);
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
    Ok(())
}

fn clamped_gradient_i32(top: i32, left: i32, top_left: i32) -> i32 {
    (top + left - top_left).clamp(top.min(left), top.max(left))
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

/// GPU-only convenience encoder for one bounded standard regular VarDCT
/// transform.
pub struct VarDctEncoder {
    encoder: GpuEncoder<VarDctBackend>,
    strategy: VarDctStrategy,
}

impl VarDctEncoder {
    /// Creates the profile backend.
    ///
    /// # Errors
    ///
    /// Returns an encoder error if `strategy` is not in
    /// [`VarDctStrategy::EXECUTABLE`] or the fixed standard entropy tree cannot
    /// be constructed.
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

    fn test_context() -> Option<WgpuContext> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jxl-wgpu VarDCT encoder test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        WgpuContext::new(Arc::new(device), Arc::new(queue)).ok()
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
        let frame = assemble_frame(build_frame_packet(&artifact, &code).unwrap()).unwrap();
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
        let frame = assemble_frame(build_frame_packet(&artifact, &code).unwrap()).unwrap();
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
        assert_eq!(std::mem::size_of::<VarDctKernelParams>(), 256);
        assert_eq!(std::mem::size_of::<VarDctKernelArtifact>(), 25_600);
        assert_eq!(std::mem::align_of::<VarDctKernelArtifact>(), 4);
    }

    #[test]
    fn naga_validates_regular_transform_shader_and_bounded_abi() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("VarDCT WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("VarDCT WGSL validates");
        assert!(SHADER.contains("@compute @workgroup_size(256)"));
        assert!(SHADER.contains("strategy_map: array<u32, 16>"));
        assert!(SHADER.contains("padding: array<u32, 17>"));
        assert!(SHADER.contains("forward_xyb_bits: array<u32, 3072>"));
    }

    #[test]
    fn strategy_ir_uses_exact_standard_codestream_order() {
        for (id, strategy) in VarDctStrategy::ALL.into_iter().enumerate() {
            assert_eq!(usize::from(strategy.codestream_id()), id);
        }
        assert_eq!(VarDctStrategy::Dct16x8.block_extent(), (8, 16));
        assert_eq!(VarDctStrategy::Dct8x16.block_extent(), (16, 8));
        assert_eq!(VarDctStrategy::Dct256x128.block_extent(), (128, 256));
        assert!(
            VarDctStrategy::EXECUTABLE
                .into_iter()
                .all(VarDctStrategy::is_executable)
        );
        assert!(!VarDctStrategy::Hornuss.is_executable());
        assert!(!VarDctStrategy::Dct64x64.is_executable());
    }

    #[test]
    fn unsupported_strategy_is_rejected_without_a_dct8_fallback() {
        let Some(context) = test_context() else {
            return;
        };
        let result = VarDctEncoder::new(context, VarDctStrategy::Hornuss);
        assert!(matches!(result, Err(EncodeError::InvalidConfiguration(_))));
    }

    #[test]
    fn gpu_profile_encodes_exact_black_from_padded_rgb() {
        let Some(context) = test_context() else {
            return;
        };
        let encoder = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct8).unwrap();
        let source = padded_rgb_source(&context, &[[0, 0, 0]; 64]);
        let plan = encoder.memory_plan(&source).unwrap();
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
    fn every_executable_regular_strategy_emits_a_standard_black_codestream() {
        let Some(context) = test_context() else {
            return;
        };
        for strategy in VarDctStrategy::EXECUTABLE {
            let (width, height) = strategy.block_extent();
            let width = usize::from(width);
            let height = usize::from(height);
            let pixels = vec![[0, 0, 0]; width * height];
            let encoder = VarDctEncoder::new(context.clone(), strategy).unwrap();
            let codestream = encoder
                .encode(padded_rgb_source_sized(&context, width, height, &pixels))
                .unwrap();
            assert_eq!(
                decode_rgb8_sized(&codestream, width, height),
                vec![0; width * height * 3],
                "strategy={strategy:?}",
            );
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        }
    }

    #[test]
    fn every_executable_regular_strategy_preserves_solid_color_and_lf_gradient() {
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
}
