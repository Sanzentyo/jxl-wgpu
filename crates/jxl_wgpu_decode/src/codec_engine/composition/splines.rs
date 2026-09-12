//! Resident quantized spline programs, geometry and ordered component rendering.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{FrameInventory, FrameSectionKind};

use super::entropy_program::{self, Program};
use crate::entropy::EntropyStreamParams;
use crate::entropy_window::{EntropyStreamWindows, GroupEntropyRange, GroupStreamSegment};
use crate::modular_tree::{EntropyDecoderIr, MaTreeLimits};
use crate::{Error, GpuCodestream, Result, SplineResource};

mod geometry;
mod render;
pub(super) use geometry::{Cache, GeometryPlan, PendingGeometry};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

// Four point descriptors, 128 DCT coefficients, and four geometry-cache control words.
const HEADER_WORDS: u32 = 136;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct Params {
    entropy: EntropyStreamParams,
    capacity: u32,
    window: [u32; 4],
    limits: [u32; 4], // control points, context offset, reset, header stride
}
const _: () = assert!(std::mem::size_of::<Params>() == 48);

pub(super) type Plan = entropy_program::Plan<Params>;
pub(super) type Pending = entropy_program::Pending<Params>;

impl Program for Params {
    const LABEL: &'static str = "JPEG XL spline entropy";
    const SHADER: &'static str = include_str!("splines/decode.wgsl");

    fn update(&mut self, window: &GroupStreamSegment, capacity: u32, reset: bool) {
        self.capacity = capacity;
        self.limits[2] = u32::from(reset);
        self.window = [
            window.window_logical_start,
            window.window_upload_start,
            window.available_token_end,
            window.window_yield_end,
        ];
    }

    fn stride(&self) -> u32 {
        1
    }

    fn rejected(&self, code: u32) -> Error {
        let limit = match code {
            12 => Some((SplineResource::ControlPoints, u64::from(self.limits[0]))),
            13 => Some((SplineResource::CoordinateMagnitude, (1 << 23) - 1)),
            14 => Some((SplineResource::DeltaMagnitude, (1 << 30) - 1)),
            _ => None,
        };
        if let Some((resource, limit)) = limit {
            Error::SplineResourceLimit { resource, limit }
        } else if code == 15 {
            Error::SplineGeometry { code }
        } else {
            Error::SplineEntropy { code }
        }
    }
}

impl Plan {
    pub(super) fn new(
        source: &GpuCodestream,
        frame: &FrameInventory,
        start: Option<u64>,
        limit: u64,
    ) -> Result<Self> {
        let section = frame
            .sections
            .iter()
            .find(|section| {
                matches!(
                    section.kind,
                    FrameSectionKind::Single | FrameSectionKind::LowFrequencyGlobal
                )
            })
            .ok_or(Error::EngineContract("spline program lacks LF-global"))?;
        let end = section.bits.end().ok_or_else(overflow)?;
        let start = start.unwrap_or(section.bits.offset);
        if start < section.bits.offset || start > end {
            return Err(Error::EngineContract("spline prefix leaves LF-global"));
        }
        let mut reader = source.reader();
        reader.skip_bits(start)?;
        let entropy = EntropyDecoderIr::parse(
            &mut crate::vardct_frontend::BoundedBitInput::new(&mut reader, end),
            6,
            MaTreeLimits::default(),
        )?;
        let token_start = reader.bit_offset();
        let (width, height) = frame.color_sample_extent().ok_or_else(overflow)?;
        let max_points = ((u64::from(width) * u64::from(height) / 2).min(1 << 20)) as u32;
        let symbols = max_points * 131 + 2;
        let history_words = entropy.lz77_window_words(0, symbols)?;
        let mut metadata = entropy.pack_gpu_metadata()?.words;
        let contexts = u32::try_from(metadata.len()).map_err(|_| overflow())?;
        metadata.extend(
            entropy.context_to_cluster[..6]
                .iter()
                .map(|&value| u32::from(value)),
        );
        let windows = EntropyStreamWindows::new(
            source.logical_bytes(),
            GroupEntropyRange {
                token_bit_offset: token_start,
                token_bit_end: end,
            },
            limit,
        )?;
        let first = windows.get(0).ok_or_else(overflow)?;
        Ok(Self {
            metadata,
            windows,
            token_start,
            history_words,
            params: Params {
                entropy: EntropyStreamParams {
                    token_start: 0,
                    token_end: first.stream_token_end,
                    lz77_window_mask: history_words.saturating_sub(1),
                },
                capacity: 0,
                window: [0; 4],
                limits: [max_points, contexts, 1, HEADER_WORDS],
            },
        })
    }
}

fn overflow() -> Error {
    Error::backend("spline entropy geometry or addressing exceeds u32")
}
