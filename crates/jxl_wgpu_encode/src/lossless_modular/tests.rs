#[cfg(all(test, target_arch = "wasm32"))]
use std::future::Future;
#[cfg(all(test, not(target_arch = "wasm32")))]
use std::num::NonZeroU64;
#[cfg(all(test, not(target_arch = "wasm32")))]
use std::sync::Arc;
#[cfg(all(test, not(target_arch = "wasm32")))]
use std::task::Context;

#[cfg(not(target_arch = "wasm32"))]
use super::dispatch::LosslessModularBackend;
#[cfg(not(target_arch = "wasm32"))]
use super::grid::{LosslessModularGroup, LosslessModularGroupGrid};
#[cfg(not(target_arch = "wasm32"))]
use super::memory::{align_up, event_capacity};
#[cfg(not(target_arch = "wasm32"))]
use super::serializer::{ModularFrameHeader, frame_header, image_header, parse_group_artifact};
#[cfg(not(target_arch = "wasm32"))]
use super::streaming::{MapCompletion, StreamingAdvance, StreamingCursor, StreamingPass};
#[cfg(not(target_arch = "wasm32"))]
use super::types::lossless_modular_source_spec;
#[cfg(not(target_arch = "wasm32"))]
use super::types::{
    EVENT_WORDS, LosslessModularFormat, ModularArtifactHeader, ModularEvent, ModularParams,
    OUTPUT_HEADER_WORDS, SHADER,
};
use crate::EncodeError;
use crate::LosslessModularSubmission;
#[cfg(not(target_arch = "wasm32"))]
use crate::buffer_pool::EncoderBufferPool;
#[cfg(not(target_arch = "wasm32"))]
use crate::prefix::{LZ77_SYMBOLS, RAW_SYMBOLS};
#[cfg(not(target_arch = "wasm32"))]
use crate::{
    AnimationHeader, BackendError, BlendMode, CodestreamAssembler,
    DEFAULT_ENCODER_BUFFER_POOL_BYTES, FrameBlend, FrameGroupLayout, FrameIndex, FrameOptions,
    FramePacketSet, GpuFrameArtifacts, GroupPacket, GroupPacketKind, KernelStage,
    LOSSLESS_MODULAR_GROUP_DIMENSION, LosslessModularAnimationDescriptor, LosslessModularEncoder,
    LosslessModularTreeMode, WgpuContext,
};
#[cfg(target_arch = "wasm32")]
use crate::{GpuEncodeJob, LosslessModularJob};
#[cfg(all(test, not(target_arch = "wasm32")))]
use jxl_gpu_bitstream::{ACCELERATION_INDEX_BOX_TYPE, Gray8AccelerationIndex};
#[cfg(all(test, not(target_arch = "wasm32")))]
use jxl_gpu_formats::{
    Channel, ColorSpecification, PackingFieldKind, PixelFormat, RgbChannelOrder, SampleKind,
};
#[cfg(all(test, not(target_arch = "wasm32")))]
use wgpu::util::DeviceExt;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_compile_contract {
    use super::*;

    fn assert_gpu_job<T: GpuEncodeJob>() {}

    fn assert_runtime_neutral_future<T>()
    where
        T: Future<Output = Result<Vec<u8>, EncodeError>>,
    {
    }

    #[test]
    fn browser_streaming_types_implement_the_public_completion_contracts() {
        assert_gpu_job::<LosslessModularJob>();
        assert_runtime_neutral_future::<LosslessModularSubmission>();
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod native_tests {
    use std::num::NonZeroU32;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use super::*;
    use jxl::api::{
        Endianness, JxlBitDepth, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions,
        JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
    };
    use jxl_gpu_formats::{ImageLayout, PitchLinearPlaneLayout};
    use jxl_gpu_protocol::Extent2d;
    use std::process::Command;

    fn checked_in_gpu_gray8_lossless() -> Vec<u8> {
        fn nibble(byte: u8) -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => panic!("invalid checked-in fixture hex digit"),
            }
        }

        let digits = include_str!("../../test-data/gpu_gray8_lossless.jxl.hex")
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        assert_eq!(digits.len() % 2, 0, "fixture hex must contain whole bytes");
        digits
            .chunks_exact(2)
            .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
            .collect()
    }

    struct ReentrantWake {
        completion: Arc<MapCompletion>,
        observed_unlocked: Arc<AtomicBool>,
    }

    impl std::task::Wake for ReentrantWake {
        fn wake(self: Arc<Self>) {
            let _guard = self
                .completion
                .state
                .try_lock()
                .expect("completion mutex must be unlocked before invoking a waker");
            self.observed_unlocked.store(true, Ordering::Release);
        }
    }

    #[test]
    fn completion_wakes_after_releasing_its_mutex() {
        let completion = Arc::new(MapCompletion::default());
        let observed_unlocked = Arc::new(AtomicBool::new(false));
        let waker = std::task::Waker::from(Arc::new(ReentrantWake {
            completion: Arc::clone(&completion),
            observed_unlocked: Arc::clone(&observed_unlocked),
        }));
        let context = Context::from_waker(&waker);
        assert!(completion.poll(&context).is_none());
        completion.complete(Ok(()));
        assert!(observed_unlocked.load(Ordering::Acquire));
    }

    #[test]
    fn streaming_cursor_visits_every_batch_in_both_passes() {
        let mut cursor = StreamingCursor::new(3).unwrap();
        let mut visited = Vec::new();
        loop {
            visited.push((cursor.pass, cursor.batch_index));
            if cursor.advance() == StreamingAdvance::Complete {
                break;
            }
        }
        assert_eq!(
            visited,
            vec![
                (StreamingPass::Histogram, 0),
                (StreamingPass::Histogram, 1),
                (StreamingPass::Histogram, 2),
                (StreamingPass::Serialize, 0),
                (StreamingPass::Serialize, 1),
                (StreamingPass::Serialize, 2),
            ]
        );
    }

    #[test]
    fn streaming_cursor_rejects_an_empty_dispatch_plan() {
        assert!(matches!(
            StreamingCursor::new(0),
            Err(EncodeError::Backend(BackendError::Invariant(
                "streaming dispatch plan has no batches"
            )))
        ));
    }

    #[test]
    fn naga_validates_the_streaming_modular_shader() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("Modular WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("Modular WGSL validates with portable WebGPU capabilities");
    }

    #[test]
    fn modular_params_are_word_padded() {
        assert_eq!(std::mem::size_of::<ModularParams>(), 256);
        assert_eq!(std::mem::align_of::<ModularParams>(), 4);
        let params = ModularParams {
            width: 1,
            height: 2,
            row_stride: 3,
            byte_offset: 4,
            output_word_offset: 5,
            channel: 6,
            channels: 7,
            bytes_per_sample: 8,
            sample_mask: 9,
            _padding: [0; 55],
        };
        let words = bytemuck::cast::<ModularParams, [u32; 64]>(params);
        assert_eq!(&words[..9], &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert!(words[9..].iter().all(|&word| word == 0));
    }

    #[test]
    fn modular_artifact_records_are_word_aligned_and_ordered() {
        assert_eq!(std::mem::size_of::<ModularArtifactHeader>(), 53 * 4);
        assert_eq!(std::mem::align_of::<ModularArtifactHeader>(), 4);
        assert_eq!(std::mem::size_of::<ModularEvent>(), 4 * 4);
        assert_eq!(std::mem::align_of::<ModularEvent>(), 4);

        let header = ModularArtifactHeader {
            event_count: 7,
            raw_counts: std::array::from_fn(|index| 100 + index as u32),
            lz77_counts: std::array::from_fn(|index| 200 + index as u32),
        };
        let words = bytemuck::cast::<ModularArtifactHeader, [u32; 53]>(header);
        assert_eq!(words[0], 7);
        assert_eq!(words[1..20], header.raw_counts);
        assert_eq!(words[20..53], header.lz77_counts);

        let event = ModularEvent {
            kind: 1,
            token: 2,
            extra_bit_count: 3,
            extra_bits: 4,
        };
        assert_eq!(
            bytemuck::cast::<ModularEvent, [u32; 4]>(event),
            [1, 2, 3, 4]
        );
    }

    #[test]
    fn modular_input_contract_is_explicit_and_does_not_relabel_defined_color() {
        for format in [
            LosslessModularFormat::Gray,
            LosslessModularFormat::Rgb,
            LosslessModularFormat::Rgba,
        ] {
            for bits_per_sample in 1..=16 {
                let pixel_format = format.pixel_format(bits_per_sample).unwrap();
                let spec = lossless_modular_source_spec(&pixel_format).unwrap();
                assert_eq!(spec.format, format);
                assert_eq!(spec.bits_per_sample, bits_per_sample);
                assert_eq!(spec.bytes_per_sample, u8::from(bits_per_sample > 8) + 1);
                let word = &pixel_format.planes[0].words[0];
                assert_eq!(word.bits(), u32::from(spec.bytes_per_sample) * 8);
                assert!(matches!(
                    word.fields.last().map(|field| field.kind),
                    Some(PackingFieldKind::Channel(Channel::X))
                ));
            }
        }
        assert!(LosslessModularFormat::Gray.pixel_format(0).is_err());
        assert!(LosslessModularFormat::Gray.pixel_format(17).is_err());
        let undefined = ColorSpecification::Undefined;
        assert!(
            lossless_modular_source_spec(
                &PixelFormat::rgb8(RgbChannelOrder::Rgb, true, undefined,)
            )
            .is_err()
        );
        for order in [RgbChannelOrder::Bgr, RgbChannelOrder::Bgra] {
            assert!(
                lossless_modular_source_spec(&PixelFormat::rgb8(order, false, undefined)).is_err()
            );
        }
        let defined = ColorSpecification::Defined(jxl_gpu_formats::ColorSpec::bt709(
            jxl_gpu_formats::ColorRange::Full,
            jxl_gpu_formats::ChromaLocation2d::CENTER,
        ));
        assert!(
            lossless_modular_source_spec(&PixelFormat::rgb8(RgbChannelOrder::Rgb, false, defined,))
                .is_err()
        );
    }

    #[test]
    fn animation_metadata_and_frame_contracts_are_standard_inventory() {
        let animation = AnimationHeader::Animation {
            ticks_per_second_numerator: NonZeroU32::new(24_000).unwrap(),
            ticks_per_second_denominator: NonZeroU32::new(1_001).unwrap(),
            num_loops: 7,
            have_timecodes: true,
        };
        let header = image_header(4, 3, LosslessModularFormat::Rgba, 12, animation).unwrap();
        let mut assembler = CodestreamAssembler::new(header).unwrap();
        let slot_one = crate::ReferenceSlot::new(1).unwrap();
        let first = ModularFrameHeader {
            animation,
            canvas_width: 4,
            canvas_height: 3,
            options: FrameOptions {
                timing: crate::FrameTiming {
                    duration_ticks: 2,
                    timecode: Some(0x1122_3344),
                },
                save_as_reference: slot_one,
                ..FrameOptions::default()
            },
            is_last: false,
        };
        let second = ModularFrameHeader {
            animation,
            canvas_width: 4,
            canvas_height: 3,
            options: FrameOptions {
                timing: crate::FrameTiming {
                    duration_ticks: 257,
                    timecode: Some(0xaabb_ccdd),
                },
                crop: Some(crate::FrameCrop::new(-1, 1, 3, 2).unwrap()),
                color_blend: FrameBlend {
                    mode: BlendMode::Add,
                    source_reference: slot_one,
                    clamp: false,
                },
                extra_channel_blends: vec![FrameBlend {
                    mode: BlendMode::Multiply,
                    source_reference: slot_one,
                    clamp: true,
                }],
                ..FrameOptions::default()
            },
            is_last: true,
        };
        for (index, frame) in [(0, first), (1, second)] {
            let packets = FramePacketSet::new(
                frame_header(LosslessModularFormat::Rgba, &frame).unwrap(),
                FrameGroupLayout::new(1, 1, 1).unwrap(),
                [GroupPacket::new(GroupPacketKind::Single, Vec::new())],
            )
            .unwrap();
            assembler
                .insert(GpuFrameArtifacts {
                    frame_index: FrameIndex::new(index),
                    is_last: frame.is_last,
                    packets,
                    acceleration: None,
                })
                .unwrap();
        }
        let encoded = assembler.finish_raw().unwrap();
        let parsed =
            jxl_gpu_bitstream::parse(&encoded, jxl_gpu_bitstream::ParseLimits::default()).unwrap();
        let inventory = parsed
            .codestream_inventory(jxl_gpu_bitstream::InventoryLimits::default())
            .unwrap();
        assert_eq!(
            inventory.image_header.animation,
            Some(jxl_gpu_bitstream::AnimationInventory {
                ticks_per_second_numerator: 24_000,
                ticks_per_second_denominator: 1_001,
                num_loops: 7,
                have_timecodes: true,
            })
        );
        assert_eq!(inventory.frames.len(), 2);
        assert_eq!(inventory.frames[0].duration_ticks, 2);
        assert_eq!(inventory.frames[0].timecode, Some(0x1122_3344));
        assert_eq!(inventory.frames[0].save_as_reference, 1);
        assert!(!inventory.frames[0].is_last);
        assert_eq!((inventory.frames[1].x0, inventory.frames[1].y0), (-1, 1));
        assert_eq!(
            (inventory.frames[1].width, inventory.frames[1].height),
            (3, 2)
        );
        assert_eq!(inventory.frames[1].duration_ticks, 257);
        assert_eq!(inventory.frames[1].timecode, Some(0xaabb_ccdd));
        assert_eq!(
            inventory.frames[1].color_blend.mode,
            jxl_gpu_bitstream::FrameBlendMode::Add
        );
        assert_eq!(inventory.frames[1].color_blend.source, 1);
        assert_eq!(
            inventory.frames[1].extra_channel_blends[0].mode,
            jxl_gpu_bitstream::FrameBlendMode::Multiply
        );
        assert!(inventory.frames[1].extra_channel_blends[0].clamp);
        assert_eq!(inventory.frames[1].extra_channel_blends[0].source, 1);
        assert!(inventory.frames[1].is_last);
    }

    #[test]
    fn group_grid_is_row_major_and_covers_edge_tiles_exactly() {
        let grid = LosslessModularGroupGrid::for_extent(513, 257).unwrap();
        assert_eq!(
            grid,
            LosslessModularGroupGrid {
                width: 513,
                height: 257,
                columns: 3,
                rows: 2,
                groups: 6,
                lf_columns: 1,
                lf_rows: 1,
                lf_groups: 1,
            }
        );
        let groups = grid.ordered_groups().collect::<Vec<_>>();
        assert_eq!(groups.len(), 6);
        assert_eq!(
            groups[0],
            LosslessModularGroup {
                index: 0,
                column: 0,
                row: 0,
                x: 0,
                y: 0,
                width: 256,
                height: 256,
            }
        );
        assert_eq!(groups[2].x, 512);
        assert_eq!(groups[2].width, 1);
        assert_eq!(groups[3].y, 256);
        assert_eq!(groups[3].height, 1);
        assert_eq!((groups[5].x, groups[5].y), (512, 256));
        assert!(grid.group(6).is_none());

        assert_eq!(
            LosslessModularGroupGrid::for_extent(1, 1).unwrap().groups,
            1
        );
        assert!(LosslessModularGroupGrid::for_extent(0, 1).is_err());
        assert!(LosslessModularGroupGrid::for_extent(1, 0).is_err());
    }

    fn artifact_bytes(header: ModularArtifactHeader, events: &[ModularEvent]) -> Vec<u8> {
        let mut bytes = bytemuck::bytes_of(&header).to_vec();
        bytes.extend_from_slice(bytemuck::cast_slice(events));
        bytes
    }

    #[test]
    fn packet_builder_rejects_impossible_histogram_bins() {
        let mut header = ModularArtifactHeader {
            event_count: 1,
            raw_counts: [0; RAW_SYMBOLS],
            lz77_counts: [0; LZ77_SYMBOLS],
        };
        header.raw_counts[0] = 1;
        header.raw_counts[12] = 1;
        let bytes = artifact_bytes(
            header,
            &[ModularEvent {
                kind: 0,
                token: 0,
                extra_bit_count: 0,
                extra_bits: 0,
            }],
        );
        assert!(parse_group_artifact(1, 1, 1, &bytes).is_err());
    }

    #[test]
    fn packet_builder_rejects_noncanonical_events_and_histogram_mismatches() {
        let mut header = ModularArtifactHeader {
            event_count: 1,
            raw_counts: [0; RAW_SYMBOLS],
            lz77_counts: [0; LZ77_SYMBOLS],
        };
        header.raw_counts[2] = 1;
        let malformed = artifact_bytes(
            header,
            &[ModularEvent {
                kind: 0,
                token: 2,
                extra_bit_count: 0,
                extra_bits: 0,
            }],
        );
        assert!(parse_group_artifact(1, 1, 1, &malformed).is_err());

        header.raw_counts = [0; RAW_SYMBOLS];
        header.raw_counts[1] = 1;
        let mismatched = artifact_bytes(
            header,
            &[ModularEvent {
                kind: 0,
                token: 0,
                extra_bit_count: 0,
                extra_bits: 0,
            }],
        );
        assert!(parse_group_artifact(1, 1, 1, &mismatched).is_err());
    }

    #[test]
    fn packet_builder_rejects_event_streams_with_the_wrong_sample_count() {
        let mut header = ModularArtifactHeader {
            event_count: 1,
            raw_counts: [0; RAW_SYMBOLS],
            lz77_counts: [0; LZ77_SYMBOLS],
        };
        header.raw_counts[0] = 1;
        let bytes = artifact_bytes(
            header,
            &[ModularEvent {
                kind: 0,
                token: 0,
                extra_bit_count: 0,
                extra_bits: 0,
            }],
        );
        assert!(parse_group_artifact(2, 1, 1, &bytes).is_err());
    }

    /// Mirrors only the event-admission control flow in `encode` WGSL. A
    /// `true` sample is a zero packed residual; the actual token value is
    /// irrelevant to the number of four-word event records.
    fn simulated_shader_event_count(
        width: usize,
        height: usize,
        is_zero: impl Fn(usize) -> bool,
    ) -> usize {
        let mut run = 0usize;
        let mut events = 0usize;
        for y in 0..height {
            for chunk_x in (0..width).step_by(8) {
                let count = 8.min(width - chunk_x);
                let mut prefix = 0usize;
                while prefix < count && is_zero(y * width + chunk_x + prefix) {
                    prefix += 1;
                }
                if prefix == count && (run > 0 || prefix > 7) {
                    run += prefix;
                } else if prefix + run > 7 {
                    events += usize::from(run + prefix > 0);
                    events += count - prefix;
                    run = 0;
                } else {
                    events += count;
                }
            }
        }
        events + usize::from(run > 0)
    }

    #[test]
    fn event_allocation_bounds_every_shader_write() {
        // Exhaust every zero/non-zero residual stream up to 16 samples and
        // vary row boundaries because a run is intentionally frame-global.
        for width in 1usize..=16 {
            for height in 1usize..=(16 / width) {
                let pixels = width * height;
                let capacity = event_capacity(pixels).expect("small capacity is representable");
                for mask in 0u32..(1u32 << pixels) {
                    let events = simulated_shader_event_count(width, height, |index| {
                        mask & (1u32 << index) != 0
                    });
                    assert!(events <= capacity, "{width}x{height}, mask={mask:#x}");
                }
            }
        }

        let pixels = usize::try_from(
            u64::from(LOSSLESS_MODULAR_GROUP_DIMENSION)
                * u64::from(LOSSLESS_MODULAR_GROUP_DIMENSION),
        )
        .expect("maximum Modular profile dimensions fit usize");
        let capacity = event_capacity(pixels).expect("maximum event capacity fits usize");
        for events in [
            simulated_shader_event_count(pixels, 1, |_| false),
            simulated_shader_event_count(pixels, 1, |_| true),
            simulated_shader_event_count(pixels, 1, |index| index % 2 == 0),
            simulated_shader_event_count(pixels, 1, |index| index % 17 < 8),
        ] {
            assert!(events <= capacity);
        }

        let words = OUTPUT_HEADER_WORDS + capacity * EVENT_WORDS;
        let last_event_word = OUTPUT_HEADER_WORDS + (capacity - 1) * EVENT_WORDS + 3;
        assert!(last_event_word < words);
        assert_eq!(words * 4 % wgpu::COPY_BUFFER_ALIGNMENT as usize, 0);
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
            label: Some("jxl-wgpu lossless encoder test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        WgpuContext::new(Arc::new(device), Arc::new(queue)).ok()
    }

    fn packed_test_source(
        context: &WgpuContext,
        width: u32,
        height: u32,
    ) -> crate::BufferImageSource {
        packed_gray8_source(context, width, height, packed_test_pixels(width, height))
    }

    fn packed_gray8_source(
        context: &WgpuContext,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> crate::BufferImageSource {
        assert_eq!(pixels.len(), (width * height) as usize);
        let extent = Extent2d::new(width, height);
        let layout = ImageLayout::packed(
            extent,
            PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        )
        .unwrap();
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu encoder pool test source"),
                contents: &pixels,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        crate::BufferImageSource::new(buffer, layout).unwrap()
    }

    fn packed_rgba8_source(
        context: &WgpuContext,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> crate::BufferImageSource {
        assert_eq!(pixels.len(), (width * height * 4) as usize);
        let extent = Extent2d::new(width, height);
        let layout =
            ImageLayout::packed(extent, LosslessModularFormat::Rgba.pixel_format(8).unwrap())
                .unwrap();
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu RGBA animation source"),
                contents: &pixels,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        crate::BufferImageSource::new(buffer, layout).unwrap()
    }

    fn packed_test_pixels(width: u32, height: u32) -> Vec<u8> {
        let byte_count = usize::try_from(u64::from(width) * u64::from(height))
            .expect("test source size fits usize");
        (0..byte_count)
            .map(|index| ((index * 29 + index / 7) & 255) as u8)
            .collect()
    }

    fn packed_color_test_source(
        context: &WgpuContext,
        width: u32,
        height: u32,
        format: LosslessModularFormat,
    ) -> (crate::BufferImageSource, Vec<u8>) {
        let channels = format.channel_count();
        assert!(matches!(channels, 3 | 4));
        let extent = Extent2d::new(width, height);
        let row_bytes = u64::from(width) * u64::from(channels);
        let row_stride = row_bytes + 5;
        let offset = 4u64;
        let allocation_size = align_up(offset + row_stride * u64::from(height), 4).unwrap();
        let mut allocation = vec![0xa5; allocation_size as usize];
        let mut expected = Vec::with_capacity((width * height * channels) as usize);
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    let value = ((x * 37 + y * 71 + channel * 53 + (x * y + channel * y) % 251)
                        & 255) as u8;
                    let address =
                        offset + u64::from(y) * row_stride + u64::from(x * channels + channel);
                    allocation[address as usize] = value;
                    expected.push(value);
                }
            }
        }
        let order = match format {
            LosslessModularFormat::Rgb => RgbChannelOrder::Rgb,
            LosslessModularFormat::Rgba => RgbChannelOrder::Rgba,
            LosslessModularFormat::Gray => unreachable!(),
        };
        let pixel_format = PixelFormat::rgb8(order, false, ColorSpecification::Undefined);
        let layout = ImageLayout::from_planes(
            extent,
            pixel_format,
            vec![PitchLinearPlaneLayout {
                plane_index: 0,
                offset,
                row_stride,
                sample_extent: extent,
                row_bytes,
            }],
        )
        .unwrap();
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu packed color test source"),
                contents: &allocation,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        (
            crate::BufferImageSource::new(buffer, layout).unwrap(),
            expected,
        )
    }

    fn modular_integer_test_source(
        context: &WgpuContext,
        width: u32,
        height: u32,
        format: LosslessModularFormat,
        bits_per_sample: u8,
    ) -> (crate::BufferImageSource, Vec<u16>) {
        let channels = format.channel_count();
        let bytes_per_sample = if bits_per_sample <= 8 { 1u64 } else { 2u64 };
        let max_value = (1u32 << bits_per_sample) - 1;
        let extent = Extent2d::new(width, height);
        let row_bytes = u64::from(width) * u64::from(channels) * bytes_per_sample;
        let row_stride = row_bytes + 5;
        let offset = 5u64;
        let allocation_size = align_up(offset + row_stride * u64::from(height), 4).unwrap();
        let mut allocation = vec![0xa5; allocation_size as usize];
        let mut expected = Vec::with_capacity((width * height * channels) as usize);
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    let selector = (x + y * 3 + channel * 5) % 7;
                    let generated = x * 37 + y * 71 + channel * 53 + (x * y + channel * y) % 251;
                    let value = match selector {
                        0 => 0,
                        1 => max_value,
                        2 => 1.min(max_value),
                        3 => max_value.saturating_sub(1),
                        _ => generated & max_value,
                    } as u16;
                    let sample_index = u64::from(x * channels + channel);
                    let address =
                        offset + u64::from(y) * row_stride + sample_index * bytes_per_sample;
                    if bytes_per_sample == 1 {
                        let padding = !max_value as u8;
                        allocation[address as usize] = value as u8 | padding;
                    } else {
                        let storage = value | (!max_value as u16);
                        allocation[address as usize..address as usize + 2]
                            .copy_from_slice(&storage.to_le_bytes());
                    }
                    expected.push(value);
                }
            }
        }
        let layout = ImageLayout::from_planes(
            extent,
            format.pixel_format(bits_per_sample).unwrap(),
            vec![PitchLinearPlaneLayout {
                plane_index: 0,
                offset,
                row_stride,
                sample_extent: extent,
                row_bytes,
            }],
        )
        .unwrap();
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu packed integer Modular test source"),
                contents: &allocation,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        (
            crate::BufferImageSource::new(buffer, layout).unwrap(),
            expected,
        )
    }

    fn expected_artifact_storage_bytes(width: u32, height: u32, alignment: u64) -> u64 {
        LosslessModularGroupGrid::for_extent(width, height)
            .unwrap()
            .ordered_groups()
            .fold(0, |offset, group| {
                let pixels =
                    usize::try_from(u64::from(group.width) * u64::from(group.height)).unwrap();
                let words = OUTPUT_HEADER_WORDS + event_capacity(pixels).unwrap() * EVENT_WORDS;
                align_up(offset, alignment).unwrap() + u64::try_from(words).unwrap() * 4
            })
    }

    #[test]
    fn pool_exclusively_leases_real_gpu_buffer_sets_and_clear_invalidates_live_returns() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU encoder pool lease test: no wgpu adapter");
            return;
        };
        let pool = EncoderBufferPool::new(64 * 1024);
        let first = pool.checkout(context.device(), 16, 1024, false);
        let first_artifact = Arc::clone(&first.buffers().artifact);
        let second = pool.checkout(context.device(), 16, 1024, false);
        assert!(!Arc::ptr_eq(&first_artifact, &second.buffers().artifact));
        assert_eq!(pool.stats().leased_buffer_sets, 2);
        assert_eq!(pool.stats().allocation_misses, 2);

        drop(first);
        let third = pool.checkout(context.device(), 16, 1024, false);
        assert!(Arc::ptr_eq(&first_artifact, &third.buffers().artifact));
        assert_eq!(pool.stats().reuse_hits, 1);
        assert_eq!(pool.stats().leased_buffer_sets, 2);

        pool.clear();
        drop(second);
        drop(third);
        let stats = pool.stats();
        assert_eq!(stats.leased_buffer_sets, 0);
        assert_eq!(stats.idle_buffer_sets, 0);
        assert_eq!(stats.idle_buffers, 0);
        assert_eq!(stats.evicted_buffer_sets, 2);
        assert_eq!(stats.evicted_buffers, 6);
    }

    #[test]
    fn sequential_gpu_jobs_reuse_exact_buffer_sets_and_enforce_the_idle_limit() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU encoder reuse test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 17, 13);
        let encoder = LosslessModularEncoder::with_buffer_pool_limit(context, 8 * 1024 * 1024);
        let allocation_bytes = encoder.memory_plan(&source).unwrap().owned_bytes_per_job;

        encoder.submit(source.clone()).unwrap().wait().unwrap();
        let first = encoder.buffer_pool_stats();
        assert_eq!(first.allocation_misses, 1);
        assert_eq!(first.reuse_hits, 0);
        assert_eq!(first.idle_buffer_sets, 1);
        assert_eq!(first.idle_buffers, 3);
        assert_eq!(first.idle_bytes, allocation_bytes);

        encoder.submit(source).unwrap().wait().unwrap();
        let reused = encoder.buffer_pool_stats();
        assert_eq!(reused.allocation_misses, 1);
        assert_eq!(reused.reuse_hits, 1);
        assert_eq!(reused.idle_buffer_sets, 1);

        encoder.set_buffer_pool_limit(allocation_bytes - 1);
        let evicted = encoder.buffer_pool_stats();
        assert_eq!(evicted.limit_bytes, allocation_bytes - 1);
        assert_eq!(evicted.idle_bytes, 0);
        assert_eq!(evicted.idle_buffer_sets, 0);
        assert_eq!(evicted.evicted_buffer_sets, 1);
        assert_eq!(evicted.evicted_buffers, 3);
        assert_eq!(evicted.evicted_bytes, allocation_bytes);
    }

    #[test]
    fn abandoned_gpu_future_returns_buffers_and_live_memory_after_mapping() {
        let Some(context) = test_context() else {
            eprintln!("skipping abandoned GPU encoder reuse test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 71, 121);
        let encoder = LosslessModularEncoder::with_buffer_pool_limit(
            context.clone(),
            DEFAULT_ENCODER_BUFFER_POOL_BYTES,
        );
        let dropped = encoder.submit(source.clone()).unwrap();
        assert_eq!(encoder.buffer_pool_stats().leased_buffer_sets, 1);
        assert_eq!(encoder.buffer_pool_stats().idle_buffer_sets, 0);
        assert!(encoder.in_flight_memory_stats().reserved_bytes > 0);
        drop(dropped);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let pool = encoder.buffer_pool_stats();
            let memory = encoder.in_flight_memory_stats();
            if pool.leased_buffer_sets == 0
                && pool.idle_buffer_sets == 1
                && memory.reserved_bytes == 0
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "abandoned submission did not release resources: pool={pool:?}, memory={memory:?}"
            );
            let _ = context.device().poll(wgpu::PollType::Poll);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        encoder.submit(source).unwrap().wait().unwrap();
        let stats = encoder.buffer_pool_stats();
        assert_eq!(stats.allocation_misses, 1);
        assert_eq!(stats.reuse_hits, 1);
        assert_eq!(stats.leased_buffer_sets, 0);
    }

    #[test]
    fn concurrent_real_gpu_submissions_reuse_only_completed_buffer_sets() {
        let Some(context) = test_context() else {
            eprintln!("skipping concurrent GPU encoder reuse test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 71, 121);
        let encoder = LosslessModularEncoder::with_buffer_pool_limit(context, 32 * 1024 * 1024);
        let per_job = encoder.memory_plan(&source).unwrap().owned_bytes_per_job;
        let jobs = (0..8)
            .map(|_| encoder.submit(source.clone()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, per_job * 8);
        let first_outputs = jobs
            .into_iter()
            .map(LosslessModularSubmission::wait)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(first_outputs.windows(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);

        let first_stats = encoder.buffer_pool_stats();
        assert_eq!(first_stats.reuse_hits + first_stats.allocation_misses, 8);
        assert_eq!(first_stats.leased_buffer_sets, 0);
        assert!(first_stats.idle_buffer_sets >= 1);
        let guaranteed_hits = first_stats.idle_buffer_sets;

        let jobs = (0..8)
            .map(|_| encoder.submit(source.clone()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let second_outputs = jobs
            .into_iter()
            .map(LosslessModularSubmission::wait)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            second_outputs
                .iter()
                .all(|encoded| encoded == &first_outputs[0])
        );
        let second_stats = encoder.buffer_pool_stats();
        assert!(second_stats.reuse_hits >= first_stats.reuse_hits + guaranteed_hits);
        assert_eq!(second_stats.reuse_hits + second_stats.allocation_misses, 16);
        assert_eq!(second_stats.leased_buffer_sets, 0);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }

    #[test]
    fn poll_admission_failure_happens_before_submit_and_returns_the_pool_lease() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU encoder poll admission test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 17, 13);
        let encoder = LosslessModularEncoder::new(context.clone());
        let permits = (0..jxl_wgpu::SUBMISSION_POLLER_CAPACITY)
            .map(|_| context.submission_poller().try_reserve().unwrap())
            .collect::<Vec<_>>();

        let error = match encoder.submit(source.clone()) {
            Ok(_) => panic!("saturated poll admission must reject before queue submission"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            EncodeError::PollBackpressure(jxl_wgpu::SubmissionPollerError::Full { .. })
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        let rejected = encoder.buffer_pool_stats();
        assert_eq!(rejected.leased_buffer_sets, 0);
        assert_eq!(rejected.idle_buffer_sets, 1);
        assert_eq!(rejected.allocation_misses, 1);

        drop(permits);
        encoder.submit(source).unwrap().wait().unwrap();
        let recovered = encoder.buffer_pool_stats();
        assert_eq!(recovered.reuse_hits, 1);
        assert_eq!(recovered.leased_buffer_sets, 0);
    }

    #[test]
    fn concurrent_encoder_jobs_use_owned_byte_backpressure() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU encoder backpressure test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 2, 2);
        let plan = LosslessModularBackend::new(&context)
            .memory_plan(&source)
            .unwrap();
        let limited = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(plan.owned_bytes_per_job).unwrap(),
        )
        .unwrap();
        let encoder = LosslessModularEncoder::new(limited);

        let first = encoder.submit(source.clone()).unwrap();
        assert_eq!(
            encoder.in_flight_memory_stats().reserved_bytes,
            plan.owned_bytes_per_job
        );
        assert!(matches!(
            encoder.submit(source.clone()),
            Err(EncodeError::MemoryBackpressure(
                jxl_wgpu::MemoryBudgetError::Exhausted { .. }
            ))
        ));
        first.wait().unwrap();
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        encoder.submit(source).unwrap().wait().unwrap();
    }

    fn decode_gray8(encoded: &[u8]) -> Result<((usize, usize), Vec<u8>), String> {
        let mut input = encoded;
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before image info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let size = decoder.basic_info().size;
        decoder.set_pixel_format(JxlPixelFormat {
            color_type: JxlColorType::Grayscale,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: Vec::new(),
        });
        let mut frame = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before frame info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let mut pixels = vec![0u8; size.0 * size.1];
        {
            let mut buffers = [JxlOutputBuffer::new(&mut pixels, size.1, size.0)];
            loop {
                match frame
                    .process(&mut input, &mut buffers, None)
                    .map_err(|error| error.to_string())?
                {
                    ProcessingResult::Complete { .. } => break,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if input.is_empty() {
                            return Err("decoder needed more input while rendering".into());
                        }
                        frame = fallback;
                    }
                }
            }
        }
        Ok((size, pixels))
    }

    type DecodedAnimation8 = (jxl::api::JxlAnimation, Vec<(Option<f64>, Vec<u8>)>);

    fn decode_animation8(
        encoded: &[u8],
        format: LosslessModularFormat,
    ) -> Result<DecodedAnimation8, String> {
        let mut input = encoded;
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before animation info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let size = decoder.basic_info().size;
        let animation =
            decoder.basic_info().animation.clone().ok_or_else(|| {
                "Rust jxl decoded a still image instead of an animation".to_string()
            })?;
        let pixel_format = match format {
            LosslessModularFormat::Gray => JxlPixelFormat {
                color_type: JxlColorType::Grayscale,
                color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
                extra_channel_format: Vec::new(),
            },
            LosslessModularFormat::Rgb => JxlPixelFormat::rgb8(0),
            LosslessModularFormat::Rgba => JxlPixelFormat::rgba8(1),
        };
        decoder.set_pixel_format(pixel_format);
        let channels = usize::try_from(format.channel_count())
            .map_err(|_| "animation channel count overflow".to_string())?;
        let mut decoded = Vec::new();
        loop {
            let mut frame = loop {
                match decoder
                    .process(&mut input, None)
                    .map_err(|error| error.to_string())?
                {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if input.is_empty() {
                            return Err("decoder needed more input before animation frame".into());
                        }
                        decoder = fallback;
                    }
                }
            };
            let duration = frame.frame_header().duration;
            let mut pixels = vec![0u8; size.0 * size.1 * channels];
            {
                let mut buffers = [JxlOutputBuffer::new(&mut pixels, size.1, size.0 * channels)];
                decoder = loop {
                    match frame
                        .process(&mut input, &mut buffers, None)
                        .map_err(|error| error.to_string())?
                    {
                        ProcessingResult::Complete { result } => break result,
                        ProcessingResult::NeedsMoreInput { fallback, .. } => {
                            if input.is_empty() {
                                return Err(
                                    "decoder needed more input while rendering animation".into()
                                );
                            }
                            frame = fallback;
                        }
                    }
                };
            }
            decoded.push((duration, pixels));
            if !decoder.has_more_frames() {
                break;
            }
        }
        Ok((animation, decoded))
    }

    fn decode_color8(
        encoded: &[u8],
        format: LosslessModularFormat,
    ) -> Result<((usize, usize), Vec<u8>), String> {
        let mut input = encoded;
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before image info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let size = decoder.basic_info().size;
        let pixel_format = match format {
            LosslessModularFormat::Rgb => JxlPixelFormat::rgb8(0),
            LosslessModularFormat::Rgba => JxlPixelFormat::rgba8(1),
            LosslessModularFormat::Gray => {
                return Err("color decoder helper requires RGB or RGBA".into());
            }
        };
        decoder.set_pixel_format(pixel_format);
        let mut frame = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before frame info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let channels = usize::try_from(format.channel_count())
            .map_err(|_| "channel count overflow".to_string())?;
        let mut pixels = vec![0u8; size.0 * size.1 * channels];
        {
            let mut buffers = [JxlOutputBuffer::new(&mut pixels, size.1, size.0 * channels)];
            loop {
                match frame
                    .process(&mut input, &mut buffers, None)
                    .map_err(|error| error.to_string())?
                {
                    ProcessingResult::Complete { .. } => break,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if input.is_empty() {
                            return Err("decoder needed more input while rendering".into());
                        }
                        frame = fallback;
                    }
                }
            }
        }
        Ok((size, pixels))
    }

    fn decode_integer(
        encoded: &[u8],
        format: LosslessModularFormat,
        bits_per_sample: u8,
    ) -> Result<((usize, usize), Vec<u16>), String> {
        let mut input = encoded;
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before image info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let basic_info = decoder.basic_info();
        let size = basic_info.size;
        if basic_info.bit_depth
            != (JxlBitDepth::Int {
                bits_per_sample: u32::from(bits_per_sample),
            })
        {
            return Err(format!(
                "codestream depth is {:?}, expected {bits_per_sample}-bit integer",
                basic_info.bit_depth
            ));
        }
        let data_format = if bits_per_sample <= 8 {
            JxlDataFormat::U8 {
                bit_depth: bits_per_sample,
            }
        } else {
            JxlDataFormat::U16 {
                endianness: Endianness::LittleEndian,
                bit_depth: bits_per_sample,
            }
        };
        let color_type = match format {
            LosslessModularFormat::Gray => JxlColorType::Grayscale,
            LosslessModularFormat::Rgb => JxlColorType::Rgb,
            LosslessModularFormat::Rgba => JxlColorType::Rgba,
        };
        decoder.set_pixel_format(JxlPixelFormat {
            color_type,
            color_data_format: Some(data_format),
            extra_channel_format: vec![None; usize::from(format.has_alpha())],
        });
        let mut frame = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before frame info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let channels = usize::try_from(format.channel_count())
            .map_err(|_| "channel count overflow".to_string())?;
        let bytes_per_sample = data_format.bytes_per_sample();
        let row_bytes = size
            .0
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(bytes_per_sample))
            .ok_or_else(|| "decoder output row size overflow".to_string())?;
        let mut bytes = vec![0u8; row_bytes * size.1];
        {
            let mut buffers = [JxlOutputBuffer::new(&mut bytes, size.1, row_bytes)];
            loop {
                match frame
                    .process(&mut input, &mut buffers, None)
                    .map_err(|error| error.to_string())?
                {
                    ProcessingResult::Complete { .. } => break,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if input.is_empty() {
                            return Err("decoder needed more input while rendering".into());
                        }
                        frame = fallback;
                    }
                }
            }
        }
        let pixels = if bytes_per_sample == 1 {
            bytes.into_iter().map(u16::from).collect()
        } else {
            bytes
                .chunks_exact(2)
                .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
                .collect()
        };
        Ok((size, pixels))
    }

    fn decode_integer_with_djxl_if_available(
        encoded: &[u8],
        format: LosslessModularFormat,
        bits_per_sample: u8,
    ) -> Option<Result<Vec<u16>, String>> {
        static DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);
        let djxl = "/opt/homebrew/bin/djxl";
        if Command::new(djxl).arg("-V").output().is_err() {
            return None;
        }
        let id = DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("jxl-wgpu-integer-{}-{id}", std::process::id()));
        if let Err(error) = std::fs::create_dir(&directory) {
            return Some(Err(format!(
                "could not create djxl test directory: {error}"
            )));
        }
        let input = directory.join("gpu.jxl");
        let output = directory.join("gpu.pam");
        let result = (|| {
            std::fs::write(&input, encoded)
                .map_err(|error| format!("could not write djxl input: {error}"))?;
            let command = Command::new(djxl)
                .arg(&input)
                .arg(&output)
                .arg("--quiet")
                .arg(format!("--bits_per_sample={bits_per_sample}"))
                .output()
                .map_err(|error| format!("could not execute djxl: {error}"))?;
            if !command.status.success() {
                return Err(format!(
                    "djxl rejected GPU integer codestream: {}",
                    String::from_utf8_lossy(&command.stderr)
                ));
            }
            let pam = std::fs::read(&output)
                .map_err(|error| format!("could not read djxl PAM: {error}"))?;
            parse_integer_pam(&pam, format, bits_per_sample)
        })();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_dir(directory);
        Some(result)
    }

    fn parse_integer_pam(
        bytes: &[u8],
        format: LosslessModularFormat,
        bits_per_sample: u8,
    ) -> Result<Vec<u16>, String> {
        let marker = b"ENDHDR\n";
        let header_end = bytes
            .windows(marker.len())
            .position(|window| window == marker)
            .map(|position| position + marker.len())
            .ok_or_else(|| "djxl PAM is missing ENDHDR".to_string())?;
        let header = std::str::from_utf8(&bytes[..header_end])
            .map_err(|error| format!("djxl PAM header is not UTF-8: {error}"))?;
        if !header.starts_with("P7\n") {
            return Err("djxl did not emit a PAM P7 image".into());
        }
        let value = |key: &str| -> Result<usize, String> {
            header
                .lines()
                .find_map(|line| line.strip_prefix(key))
                .ok_or_else(|| format!("djxl PAM is missing {key}"))?
                .trim()
                .parse::<usize>()
                .map_err(|error| format!("invalid djxl PAM {key}: {error}"))
        };
        let width = value("WIDTH")?;
        let height = value("HEIGHT")?;
        let depth = value("DEPTH")?;
        let max_value = value("MAXVAL")?;
        let expected_depth = usize::try_from(format.channel_count())
            .map_err(|_| "PAM channel count overflow".to_string())?;
        let expected_max = (1usize << bits_per_sample) - 1;
        if depth != expected_depth || max_value != expected_max {
            return Err(format!(
                "djxl PAM has depth/maxval {depth}/{max_value}, expected {expected_depth}/{expected_max}"
            ));
        }
        let pixels = bytes
            .get(header_end..)
            .ok_or_else(|| "djxl PAM pixels are truncated".to_string())?;
        let samples = width
            .checked_mul(height)
            .and_then(|value| value.checked_mul(depth))
            .ok_or_else(|| "djxl PAM dimensions overflow".to_string())?;
        let bytes_per_sample = usize::from(bits_per_sample > 8) + 1;
        if pixels.len() != samples * bytes_per_sample {
            return Err(format!(
                "djxl PAM has {} bytes, expected {}",
                pixels.len(),
                samples * bytes_per_sample
            ));
        }
        Ok(if bytes_per_sample == 1 {
            pixels.iter().copied().map(u16::from).collect()
        } else {
            pixels
                .chunks_exact(2)
                .map(|sample| u16::from_be_bytes([sample[0], sample[1]]))
                .collect()
        })
    }

    fn decode_with_djxl_if_available(encoded: &[u8]) -> Option<Result<Vec<u8>, String>> {
        static DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);
        if Command::new("djxl").arg("-V").output().is_err() {
            return None;
        }
        let id = DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("jxl-wgpu-gray8-{}-{id}", std::process::id()));
        if let Err(error) = std::fs::create_dir(&directory) {
            return Some(Err(format!(
                "could not create djxl test directory: {error}"
            )));
        }
        let input = directory.join("gpu.jxl");
        let output = directory.join("gpu.pgm");
        let result = (|| {
            std::fs::write(&input, encoded)
                .map_err(|error| format!("could not write djxl input: {error}"))?;
            let command = Command::new("djxl")
                .arg(&input)
                .arg(&output)
                .arg("--quiet")
                .output()
                .map_err(|error| format!("could not execute djxl: {error}"))?;
            if !command.status.success() {
                return Err(format!(
                    "djxl rejected GPU codestream: {}",
                    String::from_utf8_lossy(&command.stderr)
                ));
            }
            let pgm = std::fs::read(&output)
                .map_err(|error| format!("could not read djxl PGM: {error}"))?;
            parse_pgm(&pgm)
        })();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_dir(directory);
        Some(result)
    }

    fn decode_animation_with_djxl_if_available(
        encoded: &[u8],
        format: LosslessModularFormat,
    ) -> Option<Result<Vec<Vec<u8>>, String>> {
        static DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);
        let djxl = "/opt/homebrew/bin/djxl";
        if Command::new(djxl).arg("-V").output().is_err() {
            return None;
        }
        let id = DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("jxl-wgpu-animation-{}-{id}", std::process::id()));
        if let Err(error) = std::fs::create_dir(&directory) {
            return Some(Err(format!(
                "could not create djxl animation directory: {error}"
            )));
        }
        let input = directory.join("gpu.jxl");
        let extension = if format == LosslessModularFormat::Gray {
            "pgm"
        } else {
            "pam"
        };
        let output = directory.join(format!("frame.{extension}"));
        let result = (|| {
            std::fs::write(&input, encoded)
                .map_err(|error| format!("could not write djxl animation input: {error}"))?;
            let command = Command::new(djxl)
                .arg(&input)
                .arg(&output)
                .arg("--quiet")
                .arg("--output_frames")
                .arg("--bits_per_sample=8")
                .output()
                .map_err(|error| format!("could not execute djxl animation decode: {error}"))?;
            if !command.status.success() {
                return Err(format!(
                    "djxl rejected GPU animation: {}",
                    String::from_utf8_lossy(&command.stderr)
                ));
            }
            let mut frames = std::fs::read_dir(&directory)
                .map_err(|error| format!("could not list djxl animation output: {error}"))?
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|actual| actual == extension))
                .collect::<Vec<_>>();
            frames.sort();
            frames
                .into_iter()
                .map(|path| {
                    let pgm = std::fs::read(&path)
                        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
                    if format == LosslessModularFormat::Gray {
                        parse_pgm(&pgm)
                    } else {
                        parse_pam(&pgm, format)
                    }
                })
                .collect()
        })();
        let _ = std::fs::remove_dir_all(directory);
        Some(result)
    }

    fn decode_color_with_djxl_if_available(
        encoded: &[u8],
        format: LosslessModularFormat,
    ) -> Option<Result<Vec<u8>, String>> {
        static DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);
        if Command::new("djxl").arg("-V").output().is_err() {
            return None;
        }
        let id = DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("jxl-wgpu-color8-{}-{id}", std::process::id()));
        if let Err(error) = std::fs::create_dir(&directory) {
            return Some(Err(format!(
                "could not create djxl test directory: {error}"
            )));
        }
        let input = directory.join("gpu.jxl");
        let output = directory.join("gpu.pam");
        let result = (|| {
            std::fs::write(&input, encoded)
                .map_err(|error| format!("could not write djxl input: {error}"))?;
            let command = Command::new("djxl")
                .arg(&input)
                .arg(&output)
                .arg("--quiet")
                .output()
                .map_err(|error| format!("could not execute djxl: {error}"))?;
            if !command.status.success() {
                return Err(format!(
                    "djxl rejected GPU color codestream: {}",
                    String::from_utf8_lossy(&command.stderr)
                ));
            }
            let pam = std::fs::read(&output)
                .map_err(|error| format!("could not read djxl PAM: {error}"))?;
            parse_pam(&pam, format)
        })();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_dir(directory);
        Some(result)
    }

    fn parse_pam(bytes: &[u8], format: LosslessModularFormat) -> Result<Vec<u8>, String> {
        let marker = b"ENDHDR\n";
        let header_end = bytes
            .windows(marker.len())
            .position(|window| window == marker)
            .map(|position| position + marker.len())
            .ok_or_else(|| "djxl PAM is missing ENDHDR".to_string())?;
        let header = std::str::from_utf8(&bytes[..header_end])
            .map_err(|error| format!("djxl PAM header is not UTF-8: {error}"))?;
        if !header.starts_with("P7\n") {
            return Err("djxl did not emit a PAM P7 image".into());
        }
        let value = |key: &str| -> Result<usize, String> {
            header
                .lines()
                .find_map(|line| line.strip_prefix(key))
                .ok_or_else(|| format!("djxl PAM is missing {key}"))?
                .trim()
                .parse::<usize>()
                .map_err(|error| format!("invalid djxl PAM {key}: {error}"))
        };
        let width = value("WIDTH")?;
        let height = value("HEIGHT")?;
        let depth = value("DEPTH")?;
        let max_value = value("MAXVAL")?;
        let expected_depth = usize::try_from(format.channel_count())
            .map_err(|_| "PAM channel count overflow".to_string())?;
        if depth != expected_depth || max_value != 255 {
            return Err(format!(
                "djxl PAM has depth/maxval {depth}/{max_value}, expected {expected_depth}/255"
            ));
        }
        let pixels = bytes
            .get(header_end..)
            .ok_or_else(|| "djxl PAM pixels are truncated".to_string())?;
        let expected_bytes = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(depth))
            .ok_or_else(|| "djxl PAM dimensions overflow".to_string())?;
        if pixels.len() != expected_bytes {
            return Err(format!(
                "djxl PAM has {} bytes, expected {expected_bytes}",
                pixels.len()
            ));
        }
        Ok(pixels.to_vec())
    }

    fn parse_pgm(bytes: &[u8]) -> Result<Vec<u8>, String> {
        let mut cursor = 0usize;
        let mut token = || -> Result<&[u8], String> {
            loop {
                while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                    cursor += 1;
                }
                if bytes.get(cursor) == Some(&b'#') {
                    while bytes.get(cursor).is_some_and(|byte| *byte != b'\n') {
                        cursor += 1;
                    }
                    continue;
                }
                break;
            }
            let start = cursor;
            while bytes
                .get(cursor)
                .is_some_and(|byte| !byte.is_ascii_whitespace())
            {
                cursor += 1;
            }
            bytes
                .get(start..cursor)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| "truncated PGM header".into())
        };
        if token()? != b"P5" {
            return Err("djxl did not emit a binary grayscale PGM".into());
        }
        let width = std::str::from_utf8(token()?)
            .map_err(|error| error.to_string())?
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        let height = std::str::from_utf8(token()?)
            .map_err(|error| error.to_string())?
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        if token()? != b"255" {
            return Err("djxl PGM did not contain 8-bit samples".into());
        }
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        let pixels = bytes
            .get(cursor..)
            .ok_or_else(|| "truncated PGM pixels".to_string())?;
        if pixels.len() != width * height {
            return Err(format!(
                "djxl PGM has {} samples, expected {}",
                pixels.len(),
                width * height
            ));
        }
        Ok(pixels.to_vec())
    }

    #[test]
    fn gpu_groups_cover_safe_boundary_extents_and_decode_exactly() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU multi-group lossless encode test: no wgpu adapter");
            return;
        };
        let encoder = LosslessModularEncoder::new(context.clone());
        let extents = [
            (1, 1),
            (17, 13),
            (257, 1),
            (1, 257),
            (257, 257),
            (513, 3),
            (3, 513),
            (513, 513),
            (4_097, 1),
            (1, 4_097),
        ];
        for (width, height) in extents {
            let expected = packed_test_pixels(width, height);
            let source = packed_test_source(&context, width, height);
            let memory = encoder.memory_plan(&source).unwrap();
            assert_eq!(
                memory.parameter_storage_bytes,
                u64::from(memory.group_grid.groups) * std::mem::size_of::<ModularParams>() as u64
            );
            assert_eq!(
                memory.artifact_storage_bytes,
                expected_artifact_storage_bytes(
                    width,
                    height,
                    u64::from(
                        context
                            .device()
                            .limits()
                            .min_storage_buffer_offset_alignment
                    )
                    .max(4),
                )
            );
            assert_eq!(
                memory.readback_bytes,
                if memory.direct_readback {
                    0
                } else {
                    memory.artifact_storage_bytes
                }
            );
            assert_eq!(
                memory.owned_bytes_per_job,
                memory.parameter_storage_bytes
                    + memory.artifact_storage_bytes
                    + memory.readback_bytes
            );
            assert_eq!(
                memory.addressed_bytes_per_job,
                memory.owned_bytes_per_job + memory.source_binding_bytes
            );

            let submission = encoder.submit(source.clone()).unwrap();
            assert_eq!(
                submission.ordered_groups().collect::<Vec<_>>(),
                memory.group_grid.ordered_groups().collect::<Vec<_>>()
            );
            let encoded = submission.wait().unwrap();
            let (size, decoded) = decode_gray8(&encoded)
                .unwrap_or_else(|error| panic!("Rust jxl rejected {width}x{height}: {error}"));
            assert_eq!(size, (width as usize, height as usize));
            assert_eq!(decoded, expected, "Rust jxl mismatch for {width}x{height}");
            let container = encoder.encode_container(source).unwrap();
            let parsed =
                jxl_gpu_bitstream::parse(&container, jxl_gpu_bitstream::ParseLimits::default())
                    .unwrap();
            assert_eq!(parsed.codestream(), encoded);
            let (_, container_decoded) = decode_gray8(&container).unwrap_or_else(|error| {
                panic!("Rust jxl rejected {width}x{height} container: {error}")
            });
            assert_eq!(container_decoded, expected);
            if let Some(decoded) = decode_with_djxl_if_available(&container) {
                assert_eq!(
                    decoded.unwrap_or_else(|error| {
                        panic!("djxl rejected {width}x{height} container: {error}")
                    }),
                    expected,
                    "djxl mismatch for {width}x{height}"
                );
            }
        }
    }

    #[test]
    fn packed_rgb8_and_rgba8_roundtrip_across_aspect_ratios_with_both_decoders() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU color encode test: no wgpu adapter");
            return;
        };
        let encoder = LosslessModularEncoder::new(context.clone());
        assert!(
            encoder
                .capabilities()
                .has_stage(KernelStage::ColorTransform)
        );
        assert!(
            encoder
                .capabilities()
                .has_stage(KernelStage::ModularTransform)
        );
        for format in [LosslessModularFormat::Rgb, LosslessModularFormat::Rgba] {
            for (width, height) in [(1, 513), (513, 1), (257, 3), (17, 13)] {
                let (source, expected) = packed_color_test_source(&context, width, height, format);
                let memory = encoder.memory_plan(&source).unwrap();
                assert_eq!(memory.format, format);
                assert_eq!(memory.channel_count, format.channel_count());
                assert_eq!(
                    memory.parameter_storage_bytes,
                    u64::from(memory.group_grid.groups)
                        * u64::from(format.channel_count())
                        * std::mem::size_of::<ModularParams>() as u64
                );
                let encoded = encoder.encode(source.clone()).unwrap();
                let (size, decoded) = decode_color8(&encoded, format).unwrap_or_else(|error| {
                    panic!("Rust decoder rejected {format:?} {width}x{height}: {error}")
                });
                assert_eq!(size, (width as usize, height as usize));
                assert_eq!(decoded, expected, "{format:?} {width}x{height}");
                if let Some(decoded) = decode_color_with_djxl_if_available(&encoded, format) {
                    assert_eq!(
                        decoded.unwrap_or_else(|error| {
                            panic!("djxl rejected {format:?} {width}x{height}: {error}")
                        }),
                        expected,
                        "{format:?} {width}x{height}"
                    );
                }
                if (width, height) == (17, 13) {
                    let async_encoded = pollster::block_on(encoder.submit(source).unwrap())
                        .expect("runtime-neutral color submission succeeds");
                    assert_eq!(async_encoded, encoded);
                }
            }
        }
    }

    #[test]
    fn streamed_local_ma_rgb8_16k_panorama_is_exact_and_runtime_neutral() {
        let Some(context) = test_context() else {
            eprintln!("skipping streamed 16K GPU encode test: no wgpu adapter");
            return;
        };
        let width = 16_384;
        let height = 1;
        let format = LosslessModularFormat::Rgb;
        let (source, expected) = packed_color_test_source(&context, width, height, format);
        let encoder =
            LosslessModularEncoder::with_tree_mode(context, LosslessModularTreeMode::LocalPerGroup);
        let memory = encoder.memory_plan(&source).unwrap();
        assert!(memory.streaming);
        assert!(memory.batch_count > 1);
        assert_eq!(memory.gpu_submission_count, memory.batch_count * 2);
        assert!(memory.artifact_storage_bytes < memory.total_artifact_bytes);
        assert!(memory.peak_source_binding_bytes < memory.source_binding_bytes);

        let encoded = encoder.encode(source.clone()).unwrap();
        let (size, decoded) = decode_color8(&encoded, format)
            .unwrap_or_else(|error| panic!("Rust decoder rejected 16K RGB8: {error}"));
        assert_eq!(size, (width as usize, height as usize));
        assert_eq!(decoded, expected);
        if let Some(decoded) = decode_color_with_djxl_if_available(&encoded, format) {
            assert_eq!(
                decoded.unwrap_or_else(|error| panic!("djxl rejected 16K RGB8: {error}")),
                expected
            );
        }

        let async_encoded = pollster::block_on(encoder.submit(source).unwrap())
            .expect("runtime-neutral streamed 16K submission succeeds");
        assert_eq!(async_encoded, encoded);
    }

    #[test]
    fn packed_integer_depths_roundtrip_at_group_boundaries_with_both_decoders() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU integer encode test: no wgpu adapter");
            return;
        };
        let encoder = LosslessModularEncoder::new(context.clone());
        let formats = [
            LosslessModularFormat::Gray,
            LosslessModularFormat::Rgb,
            LosslessModularFormat::Rgba,
        ];
        let depths = 1u8..=16;
        let extents = [(1, 257), (255, 3), (256, 2), (257, 1)];
        for (format_index, format) in formats.into_iter().enumerate() {
            for (depth_index, bits_per_sample) in depths.clone().enumerate() {
                let (width, height) = extents[(format_index + depth_index) % extents.len()];
                let (source, expected) =
                    modular_integer_test_source(&context, width, height, format, bits_per_sample);
                let memory = encoder.memory_plan(&source).unwrap();
                assert_eq!(memory.format, format);
                assert_eq!(memory.bits_per_sample, bits_per_sample);
                assert_eq!(memory.bytes_per_sample, u8::from(bits_per_sample > 8) + 1);
                assert_eq!(memory.channel_count, format.channel_count());
                let submission = encoder.submit(source.clone()).unwrap();
                assert_eq!(submission.bits_per_sample(), bits_per_sample);
                let encoded = submission.wait().unwrap();
                if let Some(decoded) =
                    decode_integer_with_djxl_if_available(&encoded, format, bits_per_sample)
                {
                    assert_eq!(
                        decoded.unwrap_or_else(|error| {
                            panic!(
                                "djxl rejected {format:?} {bits_per_sample}-bit {width}x{height}: {error}"
                            )
                        }),
                        expected,
                        "djxl mismatch for {format:?} {bits_per_sample}-bit {width}x{height}"
                    );
                }
                let (size, decoded) = decode_integer(&encoded, format, bits_per_sample)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Rust jxl rejected {format:?} {bits_per_sample}-bit {width}x{height}: {error}"
                        )
                    });
                assert_eq!(size, (width as usize, height as usize));
                assert_eq!(
                    decoded, expected,
                    "Rust jxl mismatch for {format:?} {bits_per_sample}-bit {width}x{height}"
                );
                if format == LosslessModularFormat::Rgba && bits_per_sample == 12 {
                    let async_encoded = pollster::block_on(encoder.submit(source).unwrap())
                        .expect("runtime-neutral 12-bit RGBA submission succeeds");
                    assert_eq!(async_encoded, encoded);
                }
            }
        }
    }

    #[test]
    fn animation_session_composites_crop_and_reference_with_both_decoders() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU animation encode test: no wgpu adapter");
            return;
        };
        let canvas_width = 4;
        let canvas_height = 3;
        let first_pixels = (0..canvas_width * canvas_height)
            .map(|index| 20 + index as u8)
            .collect::<Vec<_>>();
        let patch_pixels = vec![1, 2, 3, 4];
        let last_pixels = (0..canvas_width * canvas_height)
            .map(|index| 100 + index as u8)
            .collect::<Vec<_>>();
        let mut expected_patch = first_pixels.clone();
        for (index, value) in [(5usize, 1u8), (6, 2), (9, 3), (10, 4)] {
            expected_patch[index] += value;
        }

        let animation = AnimationHeader::Animation {
            ticks_per_second_numerator: NonZeroU32::new(100).unwrap(),
            ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
            num_loops: 2,
            have_timecodes: true,
        };
        let descriptor = LosslessModularAnimationDescriptor::new(
            canvas_width,
            canvas_height,
            LosslessModularFormat::Gray,
            8,
            animation,
        )
        .unwrap();
        let encoder = LosslessModularEncoder::new(context.clone());
        assert!(encoder.capabilities().animation);
        let mut session = encoder.begin_animation(descriptor).unwrap();
        let slot_one = crate::ReferenceSlot::new(1).unwrap();
        let slot_two = crate::ReferenceSlot::new(2).unwrap();
        let first = session
            .submit_frame(
                packed_gray8_source(&context, canvas_width, canvas_height, first_pixels.clone()),
                FrameOptions {
                    timing: crate::FrameTiming {
                        duration_ticks: 2,
                        timecode: Some(100),
                    },
                    save_as_reference: slot_one,
                    ..FrameOptions::default()
                },
            )
            .unwrap();
        let second = session
            .submit_frame(
                packed_gray8_source(&context, 2, 2, patch_pixels),
                FrameOptions {
                    timing: crate::FrameTiming {
                        duration_ticks: 3,
                        timecode: Some(101),
                    },
                    crop: Some(crate::FrameCrop::new(1, 1, 2, 2).unwrap()),
                    color_blend: FrameBlend {
                        mode: BlendMode::Add,
                        source_reference: slot_one,
                        clamp: false,
                    },
                    save_as_reference: slot_two,
                    ..FrameOptions::default()
                },
            )
            .unwrap();
        let last = session
            .submit_last_frame(
                packed_gray8_source(&context, canvas_width, canvas_height, last_pixels.clone()),
                FrameOptions {
                    timing: crate::FrameTiming {
                        duration_ticks: 4,
                        timecode: Some(102),
                    },
                    ..FrameOptions::default()
                },
            )
            .unwrap();

        let last = pollster::block_on(last).expect("runtime-neutral Future completes");
        let first = first.wait().expect("blocking animation wait completes");
        let second = pollster::block_on(second).expect("second Future completes");
        session.insert(last).unwrap();
        session.insert(second).unwrap();
        session.insert(first).unwrap();
        let container = session.finish_container().unwrap();

        let parsed =
            jxl_gpu_bitstream::parse(&container, jxl_gpu_bitstream::ParseLimits::default())
                .unwrap();
        let inventory = parsed
            .codestream_inventory(jxl_gpu_bitstream::InventoryLimits::default())
            .unwrap();
        assert_eq!(inventory.frames.len(), 3);
        assert_eq!(inventory.frames[0].duration_ticks, 2);
        assert_eq!(inventory.frames[0].timecode, Some(100));
        assert_eq!(inventory.frames[0].save_as_reference, 1);
        assert_eq!((inventory.frames[1].x0, inventory.frames[1].y0), (1, 1));
        assert_eq!(
            (inventory.frames[1].width, inventory.frames[1].height),
            (2, 2)
        );
        assert_eq!(
            inventory.frames[1].color_blend.mode,
            jxl_gpu_bitstream::FrameBlendMode::Add
        );
        assert_eq!(inventory.frames[1].color_blend.source, 1);
        assert_eq!(inventory.frames[1].save_as_reference, 2);
        assert!(inventory.frames[2].is_last);

        let (decoded_animation, decoded_frames) =
            decode_animation8(&container, LosslessModularFormat::Gray)
                .unwrap_or_else(|error| panic!("Rust jxl rejected GPU animation: {error}"));
        assert_eq!(decoded_animation.tps_numerator, 100);
        assert_eq!(decoded_animation.tps_denominator, 1);
        assert_eq!(decoded_animation.num_loops, 2);
        assert!(decoded_animation.have_timecodes);
        assert_eq!(decoded_frames.len(), 3);
        assert_eq!(decoded_frames[0].1, first_pixels);
        assert_eq!(decoded_frames[1].1, expected_patch);
        assert_eq!(decoded_frames[2].1, last_pixels);
        for (actual, expected) in decoded_frames
            .iter()
            .map(|frame| frame.0)
            .zip([20.0, 30.0, 40.0])
        {
            assert_eq!(actual, Some(expected));
        }
        if let Some(decoded) =
            decode_animation_with_djxl_if_available(&container, LosslessModularFormat::Gray)
        {
            assert_eq!(
                decoded.unwrap_or_else(|error| panic!("djxl rejected GPU animation: {error}")),
                vec![first_pixels, expected_patch, last_pixels]
            );
        }
    }

    #[test]
    fn rgba_animation_uses_standard_alpha_weighted_blending() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU RGBA animation encode test: no wgpu adapter");
            return;
        };
        let width = 3;
        let height = 2;
        let mut first_pixels = Vec::new();
        for index in 0..width * height {
            first_pixels.extend_from_slice(&[
                10 + index as u8,
                20 + index as u8,
                30 + index as u8,
                255,
            ]);
        }
        let patch_pixels = vec![200, 1, 2, 255, 3, 201, 4, 255];
        let mut expected = first_pixels.clone();
        expected[4..12].copy_from_slice(&patch_pixels);
        let animation = AnimationHeader::Animation {
            ticks_per_second_numerator: NonZeroU32::new(60).unwrap(),
            ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
            num_loops: 0,
            have_timecodes: false,
        };
        let encoder = LosslessModularEncoder::new(context.clone());
        let mut session = encoder
            .begin_animation(
                LosslessModularAnimationDescriptor::new(
                    width,
                    height,
                    LosslessModularFormat::Rgba,
                    8,
                    animation,
                )
                .unwrap(),
            )
            .unwrap();
        let reference = crate::ReferenceSlot::new(1).unwrap();
        let first = session
            .submit_frame(
                packed_rgba8_source(&context, width, height, first_pixels.clone()),
                FrameOptions {
                    timing: crate::FrameTiming {
                        duration_ticks: 1,
                        timecode: None,
                    },
                    save_as_reference: reference,
                    ..FrameOptions::default()
                },
            )
            .unwrap();
        let alpha_blend = FrameBlend {
            mode: BlendMode::Blend,
            source_reference: reference,
            clamp: false,
        };
        let last = session
            .submit_last_frame(
                packed_rgba8_source(&context, 2, 1, patch_pixels),
                FrameOptions {
                    timing: crate::FrameTiming {
                        duration_ticks: 2,
                        timecode: None,
                    },
                    crop: Some(crate::FrameCrop::new(1, 0, 2, 1).unwrap()),
                    color_blend: alpha_blend,
                    extra_channel_blends: vec![alpha_blend],
                    ..FrameOptions::default()
                },
            )
            .unwrap();
        session.insert(first.wait().unwrap()).unwrap();
        session.insert(pollster::block_on(last).unwrap()).unwrap();
        let raw = session.finish_raw().unwrap();

        let parsed =
            jxl_gpu_bitstream::parse(&raw, jxl_gpu_bitstream::ParseLimits::default()).unwrap();
        let inventory = parsed
            .codestream_inventory(jxl_gpu_bitstream::InventoryLimits::default())
            .unwrap();
        assert_eq!(inventory.image_header.extra_channel_count, 1);
        assert_eq!(
            inventory.frames[1].color_blend.mode,
            jxl_gpu_bitstream::FrameBlendMode::Blend
        );
        assert_eq!(inventory.frames[1].color_blend.alpha_channel, Some(0));
        assert_eq!(
            inventory.frames[1].extra_channel_blends[0].alpha_channel,
            Some(0)
        );

        let (_, decoded) = decode_animation8(&raw, LosslessModularFormat::Rgba)
            .unwrap_or_else(|error| panic!("Rust jxl rejected RGBA animation: {error}"));
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].1, first_pixels);
        assert_eq!(decoded[1].1, expected);
        if let Some(decoded) =
            decode_animation_with_djxl_if_available(&raw, LosslessModularFormat::Rgba)
        {
            assert_eq!(
                decoded.unwrap_or_else(|error| panic!("djxl rejected RGBA animation: {error}")),
                vec![first_pixels, expected]
            );
        }
    }

    #[test]
    fn multi_group_container_and_runtime_neutral_future_are_deterministic() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU multi-group container test: no wgpu adapter");
            return;
        };
        let width = 257;
        let height = 257;
        let expected = packed_test_pixels(width, height);
        let source = packed_test_source(&context, width, height);
        let encoder = LosslessModularEncoder::new(context);
        let raw = encoder.encode(source.clone()).unwrap();
        let async_raw = pollster::block_on(encoder.submit(source.clone()).unwrap()).unwrap();
        assert_eq!(async_raw, raw);

        let container = encoder.encode_container(source.clone()).unwrap();
        let parsed =
            jxl_gpu_bitstream::parse(&container, jxl_gpu_bitstream::ParseLimits::default())
                .unwrap();
        assert_eq!(parsed.codestream(), raw);
        assert_eq!(
            parsed.boxes_of_type(ACCELERATION_INDEX_BOX_TYPE).count(),
            0,
            "the current private acceleration index is intentionally single-group"
        );
        let (size, decoded) = decode_gray8(&container).unwrap();
        assert_eq!(size, (width as usize, height as usize));
        assert_eq!(decoded, expected);
        if let Some(decoded) = decode_with_djxl_if_available(&container) {
            assert_eq!(decoded.unwrap(), expected);
        }
        assert_eq!(encoder.encode_container(source).unwrap(), container);
    }

    #[test]
    fn gpu_tokens_form_a_reference_decodable_lossless_codestream() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU lossless encode test: no wgpu adapter");
            return;
        };
        let width = 17u32;
        let height = 13u32;
        let row_stride = 20u64;
        let binding_alignment = u64::from(
            context
                .device()
                .limits()
                .min_storage_buffer_offset_alignment,
        )
        .max(4);
        let offset = binding_alignment + 4;
        let allocation_size = align_up(offset + row_stride * u64::from(height), 4)
            .expect("test allocation size is representable");
        let mut allocation = vec![0u8; allocation_size as usize];
        let mut expected = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let value = if y < 3 {
                    0
                } else {
                    ((x * 17 + y * 31 + (x * y) % 19) & 255) as u8
                };
                allocation[(offset + u64::from(y) * row_stride + u64::from(x)) as usize] = value;
                expected.push(value);
            }
        }
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu gray8 test source"),
                contents: &allocation,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        let extent = Extent2d::new(width, height);
        let format = PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]);
        let layout = ImageLayout::from_planes(
            extent,
            format,
            vec![PitchLinearPlaneLayout {
                plane_index: 0,
                offset,
                row_stride,
                sample_extent: extent,
                row_bytes: u64::from(width),
            }],
        )
        .expect("test image layout is valid");
        let source = crate::BufferImageSource::new(buffer, layout).expect("test source is valid");
        let encoder = LosslessModularEncoder::new(context);
        let memory = encoder
            .memory_plan(&source)
            .expect("test source has a checked memory plan");
        let pixel_count = usize::try_from(width * height).expect("test dimensions fit usize");
        let expected_output_words = OUTPUT_HEADER_WORDS
            + event_capacity(pixel_count).expect("test event capacity") * EVENT_WORDS;
        let expected_output_bytes =
            u64::try_from(expected_output_words * 4).expect("test artifact size fits u64");
        assert_eq!(memory.group_grid.groups, 1);
        assert_eq!(memory.batch_count, 1);
        assert_eq!(memory.gpu_submission_count, 1);
        let parameter_bytes = std::mem::size_of::<ModularParams>() as u64;
        assert_eq!(memory.parameter_storage_bytes, parameter_bytes);
        assert_eq!(memory.artifact_storage_bytes, expected_output_bytes);
        assert_eq!(
            memory.readback_bytes,
            if memory.direct_readback {
                0
            } else {
                expected_output_bytes
            }
        );
        assert_eq!(
            memory.owned_bytes_per_job,
            parameter_bytes + expected_output_bytes + memory.readback_bytes
        );
        assert_eq!(
            memory.addressed_bytes_per_job,
            memory.owned_bytes_per_job + memory.source_binding_bytes
        );
        let in_flight = memory
            .for_in_flight(4)
            .expect("four-job memory total is representable");
        assert_eq!(in_flight.max_in_flight_jobs, 4);
        assert_eq!(in_flight.total_owned_bytes, memory.owned_bytes_per_job * 4);
        assert_eq!(
            in_flight.total_addressed_bytes,
            memory.addressed_bytes_per_job * 4
        );
        let limits = encoder.memory_limits();
        assert_eq!(
            limits.min_storage_buffer_offset_alignment.max(4),
            binding_alignment
        );
        let encoded = encoder
            .encode(source.clone())
            .expect("GPU lossless encode succeeds");
        let (size, decoded) = decode_gray8(&encoded).expect("jxl reference decoder accepts output");
        assert_eq!(size, (width as usize, height as usize));
        assert_eq!(decoded, expected);
        if let Some(decoded) = decode_with_djxl_if_available(&encoded) {
            assert_eq!(decoded.expect("djxl accepts GPU codestream"), expected);
        }
        let submission = encoder
            .submit(source.clone())
            .expect("runtime-neutral Future submission succeeds");
        let async_encoded =
            pollster::block_on(submission).expect("runtime-neutral Future encode succeeds");
        assert_eq!(async_encoded, encoded);

        let container = encoder
            .encode_container(source.clone())
            .expect("GPU lossless container encode succeeds");
        let parsed =
            jxl_gpu_bitstream::parse(&container, jxl_gpu_bitstream::ParseLimits::default())
                .expect("container is structurally valid");
        assert_eq!(parsed.codestream(), encoded);
        let boxes = parsed
            .boxes_of_type(ACCELERATION_INDEX_BOX_TYPE)
            .collect::<Vec<_>>();
        assert_eq!(boxes.len(), 1);
        let index = Gray8AccelerationIndex::parse_bound(boxes[0].payload, parsed.codestream())
            .expect("jwgp index is bound to the exact codestream");
        assert_eq!(index.width(), width);
        assert_eq!(index.height(), height);
        assert_eq!(index.sample_count(), width * height);
        let (_, decoded) =
            decode_gray8(&container).expect("jxl reference decoder ignores the private box");
        assert_eq!(decoded, expected);
        if let Some(decoded) = decode_with_djxl_if_available(&container) {
            assert_eq!(
                decoded.expect("djxl ignores jwgp and decodes jxlc"),
                expected
            );
        }
        let second = encoder
            .encode_container(source)
            .expect("second deterministic container encode succeeds");
        assert_eq!(container, second);
        assert_eq!(container, checked_in_gpu_gray8_lossless());
        if let Some(path) = std::env::var_os("JXL_WGPU_WRITE_FIXTURE") {
            std::fs::write(path, &container).expect("requested fixture path is writable");
        }
    }
}
