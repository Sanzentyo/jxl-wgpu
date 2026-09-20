//! Validated GPU-resident input to JPEG reconstruction; this is not a JPEG byte stream.

use std::sync::{Arc, atomic::AtomicUsize};

use jxl_gpu_bitstream::jpeg_reconstruction::{
    JpegReconstructionError, JpegReconstructionLimits, JpegReconstructionMetadata,
};
use jxl_gpu_bitstream::{InventoryLimits, ParseLimits, SampleBitDepth};
use jxl_wgpu::GpuBufferLease;

use super::execution::{FrameDecodeSession, VarDctRuntimeStats};
use super::output::FrameOutputMemory;
use super::source::{VarDctPrepareOptions, prepare_jpeg_source};
use super::{VarDctDecodeError, VarDctDecodeMemoryStats, VarDctSubmissionEngine};
use crate::vardct_frontend::{StandardVarDctProfile, VarDctColorTransform};
use crate::{GpuCodestream, GpuSubmissionSession};

pub(super) const QUANTIZATION_WORDS: u32 = 192;
pub(super) const STATUS_BYTES: u64 = 16;

/// Limits apply before coefficient allocation. Metadata and input retain their own bounded parsers.
#[derive(Clone, Copy, Debug)]
pub struct JpegCoefficientLimits {
    pub max_coefficient_words: u64,
    pub parse: ParseLimits,
    pub inventory: InventoryLimits,
    pub metadata: JpegReconstructionLimits,
}

impl Default for JpegCoefficientLimits {
    fn default() -> Self {
        Self {
            max_coefficient_words: 64 << 20,
            parse: ParseLimits {
                max_input_bytes: 16 << 20,
                max_codestream_bytes: 16 << 20,
                max_box_bytes: 16 << 20,
                ..Default::default()
            },
            inventory: InventoryLimits::default(),
            metadata: JpegReconstructionLimits::default(),
        }
    }
}

/// A reconstruction input is authoritative only after every packet and restoration status passes.
#[derive(Debug, thiserror::Error)]
pub enum JpegCoefficientError {
    #[error(transparent)]
    Metadata(#[from] JpegReconstructionError),
    #[error("JPEG reconstruction requires a unique jbrd box")]
    MissingMetadata,
    #[error("unsupported JPEG reconstruction input: {reason}")]
    Unsupported { reason: &'static str },
    #[error("JPEG coefficient binding disagrees with its metadata: {reason}")]
    Binding { reason: &'static str },
    #[error("JPEG coefficient storage needs {required} words, limit {limit}")]
    CoefficientLimit { required: u64, limit: u64 },
    #[error("JPEG reconstruction GPU status {actual:?}, expected {expected:?}")]
    GpuStatus {
        expected: [u32; 4],
        actual: [u32; 4],
    },
}

/// One JPEG component in original component order. Coefficients use natural 8×8 raster order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JpegCoefficientPlane {
    /// Word offset in the retained buffer, including its 192-word quantization prefix.
    pub coefficient_word_offset: u32,
    /// Word offset of this component's 64 quantizers, also in natural raster order.
    pub quantization_word_offset: u32,
    pub blocks_per_row: u32,
    pub block_rows: u32,
    pub real_blocks: [u32; 2],
    pub sampling: [u32; 2],
    pub(super) channel: u32,
}

/// Immutable checked layout for the quantization prefix and padded integer coefficient planes.
#[derive(Debug)]
pub struct JpegCoefficientLayout {
    planes: [JpegCoefficientPlane; 3],
    components: usize,
    coefficient_words: u32,
    extent: [u32; 2],
    pub(super) groups: u32,
    pub(super) rgb: bool,
    pub(super) cfl: bool,
}

impl JpegCoefficientLayout {
    pub(super) fn new(
        profile: &StandardVarDctProfile,
        grayscale: bool,
        groups: u32,
        limit: u64,
    ) -> Result<Self, JpegCoefficientError> {
        if profile.width == 0
            || profile.height == 0
            || profile.width > 65535
            || profile.height > 65535
            || profile.upsampling != 1
            || profile.sample_bit_depth != (SampleBitDepth::Integer { bits_per_sample: 8 })
            || profile.lf_level != 0
            || profile.uses_lf_frame
            || profile.color_transform == VarDctColorTransform::Xyb
        {
            return Err(JpegCoefficientError::Unsupported {
                reason: "8-bit integer non-XYB JPEG frame geometry",
            });
        }
        let order: &[u32] = if grayscale {
            &[1]
        } else if profile.color_transform == VarDctColorTransform::Ycbcr {
            &[1, 0, 2]
        } else {
            &[0, 1, 2]
        };
        let [horizontal, vertical] = profile.jpeg_block_alignment;
        if horizontal > 1 || vertical > 1 || (grayscale && horizontal + vertical != 0) {
            return Err(JpegCoefficientError::Unsupported {
                reason: "JPEG sampling factors",
            });
        }
        let padded = [
            profile.width.div_ceil(8 << horizontal) << horizontal,
            profile.height.div_ceil(8 << vertical) << vertical,
        ];
        let mut planes = [JpegCoefficientPlane::default(); 3];
        let mut words = 0u64;
        for (index, &channel) in order.iter().enumerate() {
            let shift = profile.channel_shifts[channel as usize];
            if shift.horizontal > horizontal || shift.vertical > vertical {
                return Err(JpegCoefficientError::Binding {
                    reason: "component shift exceeds MCU alignment",
                });
            }
            let stride = padded[0] >> shift.horizontal;
            let rows = padded[1] >> shift.vertical;
            let base = words;
            words += u64::from(stride) * u64::from(rows) * 64;
            let limit = limit.min(u64::from(u32::MAX - QUANTIZATION_WORDS));
            if words > limit {
                return Err(JpegCoefficientError::CoefficientLimit {
                    required: words,
                    limit,
                });
            }
            planes[index] = JpegCoefficientPlane {
                coefficient_word_offset: QUANTIZATION_WORDS + base as u32,
                quantization_word_offset: channel * 64,
                blocks_per_row: stride,
                block_rows: rows,
                real_blocks: [
                    profile.width.div_ceil(8 << shift.horizontal),
                    profile.height.div_ceil(8 << shift.vertical),
                ],
                sampling: [
                    1 << (horizontal - shift.horizontal),
                    1 << (vertical - shift.vertical),
                ],
                channel,
            };
        }
        Ok(Self {
            planes,
            components: order.len(),
            coefficient_words: words as u32,
            extent: [profile.width, profile.height],
            groups,
            rgb: profile.color_transform == VarDctColorTransform::Rgb,
            cfl: !grayscale
                && profile
                    .channel_shifts
                    .iter()
                    .all(|shift| !shift.is_subsampled()),
        })
    }

    #[must_use]
    pub fn planes(&self) -> &[JpegCoefficientPlane] {
        &self.planes[..self.components]
    }
    #[must_use]
    pub const fn extent(&self) -> [u32; 2] {
        self.extent
    }
    #[must_use]
    pub const fn coefficient_words(&self) -> u32 {
        self.coefficient_words
    }
    #[must_use]
    pub const fn storage_bytes(&self) -> u64 {
        (QUANTIZATION_WORDS as u64 + self.coefficient_words as u64) * 4
    }

    pub(super) fn memory(&self) -> FrameOutputMemory {
        FrameOutputMemory {
            storage_bytes: self.storage_bytes(),
            uniform_bytes: u64::from(self.groups) * super::execution::jpeg::RESTORE_UNIFORM_BYTES,
            status_bytes: STATUS_BYTES,
        }
    }

    pub(super) fn validate_status(&self, bytes: &[u8]) -> Result<(), VarDctDecodeError> {
        let actual: [u32; 4] =
            bytemuck::try_pod_read_unaligned(bytes).map_err(|_| VarDctDecodeError::StatusAbi {
                status: "JPEG coefficient",
            })?;
        let expected = [QUANTIZATION_WORDS, 0, self.coefficient_words, 0];
        if actual != expected {
            return Err(JpegCoefficientError::GpuStatus { expected, actual }.into());
        }
        Ok(())
    }
}

/// GPU-resident quantizers and signed coefficients, validated without a CPU codec or pixel output.
#[derive(Clone, Debug)]
pub struct GpuJpegCoefficients {
    pub(super) layout: Arc<JpegCoefficientLayout>,
    pub(super) buffer: GpuBufferLease,
}

impl GpuJpegCoefficients {
    #[must_use]
    pub fn layout(&self) -> &JpegCoefficientLayout {
        &self.layout
    }
    /// Retain this lease for the full lifetime of any GPU consumer or explicit readback.
    #[must_use]
    pub const fn buffer(&self) -> &GpuBufferLease {
        &self.buffer
    }
}

/// One complete still JPEG-reconstruction frame, with no image presentation request.
#[derive(Debug)]
pub struct JpegCoefficientSession {
    pub(super) inner: FrameDecodeSession,
    pub(super) layout: Arc<JpegCoefficientLayout>,
    metadata: JpegReconstructionMetadata,
}

impl JpegCoefficientSession {
    #[must_use]
    pub fn memory_stats(&self) -> VarDctDecodeMemoryStats {
        self.inner.memory_stats()
    }
    #[must_use]
    pub fn layout(&self) -> &JpegCoefficientLayout {
        &self.layout
    }
    #[must_use]
    pub const fn metadata(&self) -> &JpegReconstructionMetadata {
        &self.metadata
    }
}

impl GpuSubmissionSession for JpegCoefficientSession {
    type Frame = GpuJpegCoefficients;
    type Pending = super::execution::jpeg::JpegCoefficientPending;
    fn submit_next(&mut self) -> crate::Result<Option<Self::Pending>> {
        Ok(self.inner.submit_next()?.map(|inner| Self::Pending {
            inner,
            layout: Arc::clone(&self.layout),
        }))
    }
}

impl VarDctSubmissionEngine {
    /// Opens GPU-only coefficient reconstruction from a complete transport-validated JXL input.
    ///
    /// This bounded stage produces quantizers and coefficients, not original JPEG bytes. JPEG
    /// scan entropy and marker assembly remain separate. Orientation and ICC color transforms do
    /// not apply to these integer values. Header and metadata admission precede submission;
    /// decoded quantizer, task and coefficient statuses gate authoritative output.
    pub fn open_jpeg_coefficients(
        &self,
        bytes: &[u8],
        limits: JpegCoefficientLimits,
    ) -> crate::Result<JpegCoefficientSession> {
        let parsed = jxl_gpu_bitstream::parse(bytes, limits.parse)?;
        self.open_jpeg_coefficients_parsed(&parsed, limits, None)
            .map(|(session, _)| session)
    }

    pub(super) fn open_jpeg_coefficients_parsed(
        &self,
        parsed: &jxl_gpu_bitstream::ParsedJxl<'_>,
        mut limits: JpegCoefficientLimits,
        host_plan_limit: Option<u64>,
    ) -> crate::Result<(
        JpegCoefficientSession,
        jxl_gpu_bitstream::CodestreamInventory,
    )> {
        if let Some(limit) = host_plan_limit {
            limits.metadata.max_owned_bytes = limits.metadata.max_owned_bytes.min(limit);
        }
        let metadata = parsed
            .jpeg_reconstruction(limits.metadata)
            .map_err(JpegCoefficientError::from)
            .map_err(VarDctDecodeError::from)?
            .ok_or(JpegCoefficientError::MissingMetadata)
            .map_err(VarDctDecodeError::from)?;
        if let Some(limit) = host_plan_limit {
            // Bound retained ICC before allocation. Inventory scratch has independent limits.
            limits.inventory.max_decoded_icc_bytes = limits
                .inventory
                .max_decoded_icc_bytes
                .min(limit.saturating_sub(metadata.logical_owned_bytes()));
        }
        let inventory = parsed.codestream_inventory(limits.inventory)?;
        if inventory.frames.len() != 1
            || inventory.image_header.animation.is_some()
            || !inventory.image_header.extra_channels.is_empty()
            || inventory.image_header.preview_size.is_some()
        {
            return Err(VarDctDecodeError::from(JpegCoefficientError::Unsupported {
                reason: "one still frame without preview or extra channels",
            })
            .into());
        }
        let codestream = GpuCodestream::from_shared(
            parsed.codestream().into(),
            0..parsed.codestream().len(),
            false,
        )?;
        let packet = match crate::vardct_packet::BoundedVarDctPacketPlan::begin_frame_source(
            &codestream,
            &inventory,
            crate::vardct_frontend::VarDctFrameRole::Presentation,
        )
        .map_err(VarDctDecodeError::from)?
        {
            crate::vardct_packet::VarDctPacketPreparation::Ready(packet) => *packet,
            crate::vardct_packet::VarDctPacketPreparation::GlobalModular(_) => {
                return Err(VarDctDecodeError::from(JpegCoefficientError::Unsupported {
                    reason: "global Modular prefix",
                })
                .into());
            }
        };
        let layout = Arc::new(
            JpegCoefficientLayout::new(
                &packet.profile,
                inventory.image_header.grayscale,
                u32::try_from(packet.groups.len()).map_err(|_| {
                    VarDctDecodeError::ArithmeticOverflow {
                        field: "JPEG LF group count",
                    }
                })?,
                limits.max_coefficient_words,
            )
            .map_err(VarDctDecodeError::from)?,
        );
        if metadata.components().len() != layout.components {
            return Err(VarDctDecodeError::from(JpegCoefficientError::Binding {
                reason: "jbrd component count",
            })
            .into());
        }
        if packet.lf_correlation.colour_factor != 84
            || packet.lf_correlation.base != [0.0, 0.0]
            || packet.lf_correlation.lf_factors != [0, 0]
        {
            return Err(VarDctDecodeError::from(JpegCoefficientError::Unsupported {
                reason: "JPEG fixed-point color correlation",
            })
            .into());
        }
        let source = prepare_jpeg_source(
            &self.backend,
            codestream,
            &inventory,
            VarDctPrepareOptions {
                output_variant: self.pipelines.output_variant,
                stream_window_limit: self.stream_window_limit(),
                memory_limit_bytes: self.memory.snapshot().limit_bytes,
            },
            packet,
            Arc::clone(&layout),
        )?;
        let runtime_stats = Arc::new(VarDctRuntimeStats {
            submissions_per_frame: Arc::new(AtomicUsize::new(source.submissions_per_frame())),
            hf_packet_stream_batch_count: AtomicUsize::new(
                source.packet_window_batches(super::window_plan::PacketStage::Hf),
            ),
        });
        Ok((
            JpegCoefficientSession {
                inner: FrameDecodeSession {
                    backend: self.backend.clone(),
                    pipelines: Arc::clone(&self.pipelines),
                    memory_stats: source.memory,
                    runtime_stats,
                    source: Some(source),
                    memory: self.memory.clone(),
                },
                layout,
                metadata,
            },
            inventory,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jpeg_layout_rejects_floating_point_sample_declarations() {
        let case = jxl_test_support::corpus::jpeg_reconstruction::CASES
            .iter()
            .find(|case| case.name == "rgb_sequential")
            .unwrap();
        let parsed = jxl_gpu_bitstream::parse(case.input, Default::default()).unwrap();
        let mut inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let limits = JpegCoefficientLimits::default();
        let profile = StandardVarDctProfile::negotiate(&inventory).unwrap();
        assert!(
            JpegCoefficientLayout::new(&profile, false, 1, limits.max_coefficient_words).is_ok()
        );

        // The general VarDCT frontend supports floating point, including 8 total bits. JPEG
        // coefficient output must reject that declaration before memory admission independently.
        for (bits_per_sample, exponent_bits_per_sample) in
            [(8, 2), (8, 3), (8, 4), (8, 5), (16, 5), (32, 8)]
        {
            inventory.image_header.bit_depth = SampleBitDepth::Float {
                bits_per_sample,
                exponent_bits_per_sample,
            };
            let profile = StandardVarDctProfile::negotiate(&inventory).unwrap();
            assert_eq!(profile.bits_per_sample(), bits_per_sample);
            assert!(matches!(
                JpegCoefficientLayout::new(&profile, false, 1, limits.max_coefficient_words),
                Err(JpegCoefficientError::Unsupported { .. })
            ));
        }
    }
}
