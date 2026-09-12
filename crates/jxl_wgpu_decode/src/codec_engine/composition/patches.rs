//! GPU-decoded patch commands. Only status, allocation counts and the continuation bit cursor
//! cross the host boundary; positions, blend operations and pixels stay resident.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{FrameInventory, FrameSectionKind};

use super::entropy_program::{self, Program};
use crate::entropy::EntropyStreamParams;
use crate::entropy_window::{EntropyStreamWindows, GroupEntropyRange, GroupStreamSegment};
use crate::modular_tree::{EntropyDecoderIr, MaTreeLimits};
use crate::{Error, GpuCodestream, Result};

mod render;
pub(super) use render::render;
pub(super) use render::render_lf;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct Params {
    entropy: EntropyStreamParams,
    capacity: u32,
    window: [u32; 4],
    image: [u32; 4],
    limits: [u32; 4],
    references: [[u32; 4]; 4],
}
const _: () = assert!(std::mem::size_of::<Params>() == 128);

pub(super) type Plan = entropy_program::Plan<Params>;
pub(super) type Pending = entropy_program::Pending<Params>;
pub(super) use entropy_program::DecodedProgram as Dictionary;

impl Program for Params {
    const LABEL: &'static str = "JPEG XL patch entropy";
    const SHADER: &'static str = include_str!("patches/decode.wgsl");

    fn update(&mut self, window: &GroupStreamSegment, capacity: u32, reset: bool) {
        self.capacity = capacity;
        self.limits[3] = u32::from(reset);
        self.window = [
            window.window_logical_start,
            window.window_upload_start,
            window.available_token_end,
            window.window_yield_end,
        ];
    }

    fn stride(&self) -> u32 {
        self.limits[1]
    }
    fn rejected(&self, code: u32) -> Error {
        Error::PatchDictionary { code }
    }
}

impl Plan {
    pub(super) fn new(
        source: &GpuCodestream,
        frame: &FrameInventory,
        extras: &[jxl_gpu_bitstream::ExtraChannelInventory],
        references: [[u32; 4]; 4],
        limit: u64,
    ) -> Result<Self> {
        let extra_count = u32::try_from(extras.len()).map_err(|_| overflow())?;
        let section = frame
            .sections
            .iter()
            .find(|section| {
                matches!(
                    section.kind,
                    FrameSectionKind::Single | FrameSectionKind::LowFrequencyGlobal
                )
            })
            .ok_or(Error::EngineContract("patch dictionary lacks LF-global"))?;
        let mut reader = source.reader();
        reader.skip_bits(section.bits.offset)?;
        let token_end = section.bits.end().ok_or_else(overflow)?;
        let entropy = EntropyDecoderIr::parse(
            &mut crate::vardct_frontend::BoundedBitInput::new(&mut reader, token_end),
            10,
            MaTreeLimits::default(),
        )?;
        let token_start = reader.bit_offset();
        let (mut width, mut height) = frame.color_sample_extent().ok_or_else(overflow)?;
        if frame.encoding == jxl_gpu_bitstream::FrameEncoding::VarDct {
            // Dictionary bounds include JPEG-aligned padding before frame upsampling.
            // Equal raw sampling factors can require 16-pixel alignment without any shifted
            // components, so normalized chroma shifts cannot determine these bounds.
            frame.validate_jpeg_sampling()?;
            let [horizontal, vertical] =
                crate::vardct_frontend::jpeg_block_alignment(frame.jpeg_upsampling);
            let block_width = 8 << horizontal;
            let block_height = 8 << vertical;
            width = width
                .div_ceil(block_width)
                .checked_mul(block_width)
                .ok_or_else(overflow)?;
            height = height
                .div_ceil(block_height)
                .checked_mul(block_height)
                .ok_or_else(overflow)?;
        }
        let max_ref = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| 1024u64.checked_add(pixels / 4))
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(overflow)?;
        let max_positions = max_ref.checked_mul(4).ok_or_else(overflow)?;
        let stride = extra_count
            .checked_add(1)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_add(8))
            .ok_or_else(overflow)?;
        let symbols = max_positions
            .checked_mul(stride)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(overflow)?;
        let history_words = entropy.lz77_window_words(0, symbols)?;
        let mut metadata = entropy.pack_gpu_metadata()?.words;
        let contexts = u32::try_from(metadata.len()).map_err(|_| overflow())?;
        metadata.extend(
            entropy.context_to_cluster[..10]
                .iter()
                .map(|&cluster| u32::from(cluster)),
        );
        metadata.extend(extras.iter().map(|extra| {
            u32::from(matches!(
                extra.channel_type,
                jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { associated: true }
            ))
        }));
        let windows = EntropyStreamWindows::new(
            source.logical_bytes(),
            GroupEntropyRange {
                token_bit_offset: token_start,
                token_bit_end: token_end,
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
                image: [width, height, extra_count, max_ref],
                limits: [max_positions, stride, contexts, 1],
                references,
            },
        })
    }
}

fn overflow() -> Error {
    Error::backend("patch dictionary geometry or addressing exceeds u32")
}

#[cfg(test)]
mod shader_tests {
    #[test]
    fn patch_shaders_validate() {
        let decode = include_str!("patches/decode.wgsl")
            .replace(
                "/*__JXL_MODULAR_ENTROPY_ABI__*/",
                include_str!("../../modular_entropy_abi.wgsl"),
            )
            .replace(
                "/*__JXL_MODULAR_ENTROPY__*/",
                include_str!("../../modular_entropy.wgsl"),
            );
        for source in [decode.as_str(), include_str!("patches/render.wgsl")] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}
