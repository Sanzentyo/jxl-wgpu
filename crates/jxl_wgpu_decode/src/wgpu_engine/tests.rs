use std::sync::Arc;
use std::task::{Context, Waker};

use jxl_gpu_formats::{
    ColorSpecification, PixelFormat, PixelFormatClass, SampleKind, classify_pixel_format,
};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::MemoryBudget;

use crate::entropy::EntropyStreamParams;
use crate::entropy_window::{
    GroupStreamSegment, MIN_STREAM_WINDOW_BYTES, STREAM_OVERLAP_BYTES, STREAM_SENTINEL_BYTES,
};
use crate::modular_tree::MaTreeNodeIr;
use crate::profile::ModularGroup;
use crate::{Error, F64OutputPolicy, GpuOutputRequest, ModularPredictor, NumericSampleMapping};

use super::{execution::*, lifetime::*, pipeline::*, types::*};
use jxl_gpu_formats::vpi::VpiPitchLinearFormat as Vpi;
use jxl_wgpu_encode::LosslessModularFormat;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Wake;

struct ReentrantCompletionWake {
    completion: Arc<MapCompletion>,
    entered: AtomicBool,
}

impl Wake for ReentrantCompletionWake {
    fn wake(self: Arc<Self>) {
        self.entered.store(true, Ordering::SeqCst);
        let waker = Waker::from(Arc::clone(&self));
        let context = Context::from_waker(&waker);
        assert_eq!(self.completion.poll(&context), Some(Ok(())));
    }
}

const PORTABLE_CAPABILITIES: WgpuDecodeCapabilities = WgpuDecodeCapabilities {
    native_f64_arithmetic: false,
};
const GENERIC_RECONSTRUCTION: ModularReconstructionSpecialization =
    ModularReconstructionSpecialization::GenericMetaAdaptive;
const FIXED_GRADIENT_RECONSTRUCTION: ModularReconstructionSpecialization =
    ModularReconstructionSpecialization::ChannelFixed {
        predictor: ModularPredictor::Gradient,
        offset: 0,
        multiplier: 1,
        channel_count: 4,
        clusters: [1, 2, 3, 4],
    };

#[test]
fn stream_batches_rebase_unaligned_group_bits_and_respect_peak_window() {
    let codestream = [0u8; 32];
    let group = |start, end| ModularGroup {
        token_bit_offset: start,
        token_bit_end: end,
        x: 0,
        y: 0,
        width: 1,
        height: 1,
        stream_index: 0,
    };
    let groups = [group(3, 67), group(75, 139), group(147, 211)];
    let (segments, batches, peak) =
        build_stream_batches(codestream.len() as u64, &groups, 40, 1).unwrap();
    assert_eq!(
        batches
            .iter()
            .map(|batch| (batch.first_group, batch.group_count))
            .collect::<Vec<_>>(),
        [(0, 1), (1, 1), (2, 1)]
    );
    assert_eq!(peak, 16);
    for (segment, original) in segments.iter().zip(groups) {
        assert_eq!(segment.upload_offset, 0);
        assert_eq!(
            segment.window_upload_start,
            (original.token_bit_offset & 7) as u32
        );
        assert_eq!(segment.window_logical_start, 0);
        assert_eq!(
            segment.stream_token_end,
            u32::try_from(original.token_bit_end - original.token_bit_offset).unwrap()
        );
        assert_eq!(segment.available_token_end, segment.stream_token_end);
        assert_ne!(segment.flags & GroupStreamSegment::FIRST, 0);
        assert_ne!(segment.flags & GroupStreamSegment::FINAL, 0);
        assert_eq!(
            segment.input_start,
            (original.token_bit_offset / 8) as usize
        );
        assert_eq!(
            segment.input_end,
            original.token_bit_end.div_ceil(8) as usize
        );
    }
}

#[test]
fn stream_batches_never_alias_more_groups_than_scratch_lanes() {
    let codestream = [0u8; 32];
    let groups = (0..5)
        .map(|index| ModularGroup {
            token_bit_offset: index * 16 + 3,
            token_bit_end: index * 16 + 11,
            x: index as u32,
            y: 0,
            width: 1,
            height: 1,
            stream_index: index as u32,
        })
        .collect::<Vec<_>>();
    let (segments, batches, _) =
        build_stream_batches(codestream.len() as u64, &groups, 1024, 2).unwrap();
    assert_eq!(
        batches
            .iter()
            .map(|batch| (batch.first_group, batch.group_count))
            .collect::<Vec<_>>(),
        [(0, 2), (2, 2), (4, 1)]
    );
    assert_eq!(segments[0].upload_offset, 0);
    assert_eq!(segments[2].upload_offset, 0);
    assert_eq!(segments[4].upload_offset, 0);
}

#[test]
fn oversized_unaligned_group_is_split_with_overlap_and_single_lane_batches() {
    let codestream = vec![0u8; 256];
    let groups = [ModularGroup {
        token_bit_offset: 3,
        token_bit_end: 1603,
        x: 0,
        y: 0,
        width: 1,
        height: 1,
        stream_index: 0,
    }];
    let (segments, batches, peak) =
        build_stream_batches(codestream.len() as u64, &groups, 64, 8).unwrap();
    assert!(segments.len() > 2);
    assert_eq!(segments.len(), batches.len());
    assert_eq!(peak, 64);
    assert_ne!(segments[0].flags & GroupStreamSegment::FIRST, 0);
    assert_eq!(segments[0].flags & GroupStreamSegment::FINAL, 0);
    assert_eq!(
        segments.last().unwrap().flags & GroupStreamSegment::FIRST,
        0
    );
    assert_ne!(
        segments.last().unwrap().flags & GroupStreamSegment::FINAL,
        0
    );
    assert_eq!(segments.last().unwrap().available_token_end, 1600);
    assert_eq!(segments.last().unwrap().stream_token_end, 1600);
    for (index, (segment, batch)) in segments.iter().zip(&batches).enumerate() {
        assert_eq!(segment.group_index, 0);
        assert_eq!(segment.window_upload_start, 3);
        assert!(segment.input_end - segment.input_start <= 60);
        assert_eq!(batch.first_group, 0);
        assert_eq!(batch.group_count, 1);
        assert_eq!(batch.segments, index..index + 1);
    }
    for adjacent in segments.windows(2) {
        assert!(adjacent[1].window_logical_start < adjacent[0].window_yield_end);
        assert!(adjacent[0].available_token_end > adjacent[0].window_yield_end);
        assert!(adjacent[1].window_yield_end > adjacent[0].window_yield_end);
    }
}

#[test]
fn every_oversized_modular_group_uses_bounded_segments() {
    let codestream = vec![0u8; 256];
    let groups = [ModularGroup {
        token_bit_offset: 0,
        token_bit_end: 1600,
        x: 0,
        y: 0,
        width: 1,
        height: 1,
        stream_index: 0,
    }];
    let (segments, batches, peak) =
        build_stream_batches(codestream.len() as u64, &groups, 64, 1).unwrap();
    assert!(segments.len() > 1);
    assert_eq!(segments.len(), batches.len());
    assert_eq!(peak, 64);
    assert!(matches!(
        build_stream_batches(codestream.len() as u64, &groups, 39, 1),
        Err(Error::StreamWindowTooSmall {
            limit_bytes: 39,
            minimum_bytes: 40,
        })
    ));
}

#[test]
fn adaptive_stream_layout_coalesces_or_trades_lanes_for_the_byte_budget() {
    let codestream = vec![0u8; 8 * 1024];
    let groups = (0..8)
        .map(|index| ModularGroup {
            token_bit_offset: index * 8 * 1024,
            token_bit_end: (index + 1) * 8 * 1024,
            x: index as u32,
            y: 0,
            width: 1,
            height: 1,
            stream_index: index as u32,
        })
        .collect::<Vec<_>>();
    let (lanes, _, batches, peak) = select_parallel_group_layout(
        codestream.len() as u64,
        &groups,
        ParallelGroupLimits {
            stream_limit: 64 * 1024,
            lane_cap: 8,
            lane_stride: 4096,
            fixed_bytes: 1024,
            per_frame_target: 64 * 1024,
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(lanes, 8);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].first_group, 0);
    assert_eq!(batches[0].group_count, 8);
    assert_eq!(peak, 8 * 1024 + STREAM_SENTINEL_BYTES);

    let (lanes, _, batches, peak) = select_parallel_group_layout(
        codestream.len() as u64,
        &groups,
        ParallelGroupLimits {
            stream_limit: 64 * 1024,
            lane_cap: 8,
            lane_stride: 4096,
            fixed_bytes: 1024,
            per_frame_target: 20 * 1024,
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(lanes, 4);
    assert!(batches.iter().all(|batch| batch.group_count == 2));
    assert_eq!(peak, 2 * 1024 + STREAM_SENTINEL_BYTES);
}

#[test]
fn aligned_output_requires_word_isolated_plane_rows_and_internal_group_edges() {
    let extent = Extent2d::new(516, 3);
    let groups = [
        ModularGroup {
            token_bit_offset: 0,
            token_bit_end: 1,
            x: 0,
            y: 0,
            width: 256,
            height: 3,
            stream_index: 0,
        },
        ModularGroup {
            token_bit_offset: 1,
            token_bit_end: 2,
            x: 256,
            y: 0,
            width: 256,
            height: 3,
            stream_index: 1,
        },
        ModularGroup {
            token_bit_offset: 2,
            token_bit_end: 3,
            x: 512,
            y: 0,
            width: 4,
            height: 3,
            stream_index: 2,
        },
    ];
    let mut cases = Vpi::ALL
        .iter()
        .filter_map(|&format| {
            let pixel_format = format.pixel_format();
            let request = match classify_pixel_format(&pixel_format).ok()? {
                PixelFormatClass::Numeric(numeric) => {
                    let mapping = if numeric.sample_kind == SampleKind::Float
                        && numeric.bits_per_component == 64
                    {
                        NumericSampleMapping::NormalizedGray8F64(F64OutputPolicy::ExactF32Widening)
                    } else {
                        NumericSampleMapping::NormalizedGray8
                    };
                    GpuOutputRequest::numeric(pixel_format, mapping).ok()?
                }
                PixelFormatClass::Color(_) => GpuOutputRequest::color(pixel_format).ok()?,
            };
            Some((format.name(), request, crate::ModularChannels::Gray))
        })
        .collect::<Vec<_>>();
    cases.extend([
        (
            "native-gray8",
            GpuOutputRequest::numeric(
                LosslessModularFormat::Gray.pixel_format(8).unwrap(),
                NumericSampleMapping::NativeUnsigned,
            )
            .unwrap(),
            crate::ModularChannels::Gray,
        ),
        (
            "native-rgb8",
            GpuOutputRequest::color(LosslessModularFormat::Rgb.pixel_format(8).unwrap()).unwrap(),
            crate::ModularChannels::Rgb,
        ),
        (
            "native-rgba8",
            GpuOutputRequest::color(LosslessModularFormat::Rgba.pixel_format(8).unwrap()).unwrap(),
            crate::ModularChannels::Rgba,
        ),
    ]);
    for (name, request, source_channels) in cases {
        let output = OutputPlan::new(extent, &request, source_channels, 8, PORTABLE_CAPABILITIES)
            .unwrap_or_else(|error| panic!("{name} output plan failed: {error}"));
        assert_eq!(
            output.write_path_for_groups(&groups).unwrap(),
            OutputWritePath::WordAligned,
            "{name} standard 256-pixel group boundaries"
        );
    }

    let rgb_request = GpuOutputRequest::color(Vpi::Rgb8.pixel_format()).unwrap();
    let mut rgb = OutputPlan::new(
        extent,
        &rgb_request,
        crate::ModularChannels::Gray,
        8,
        PORTABLE_CAPABILITIES,
    )
    .unwrap();
    let nonisolated = [ModularGroup {
        token_bit_offset: 0,
        token_bit_end: 1,
        x: 1,
        y: 0,
        width: 255,
        height: 3,
        stream_index: 0,
    }];
    assert_eq!(
        rgb.write_path_for_groups(&nonisolated).unwrap(),
        OutputWritePath::AtomicBytes
    );
    rgb.layout.planes[0].row_stride += 1;
    assert_eq!(
        rgb.write_path_for_groups(&groups).unwrap(),
        OutputWritePath::AtomicBytes
    );
}

#[test]
fn distance_one_lz_history_uses_no_storage_scratch() {
    assert_eq!(lz77_scratch_words(0), 0);
    assert_eq!(lz77_scratch_words(1), 0);
    assert_eq!(lz77_scratch_words(2), 2);
    assert_eq!(lz77_scratch_words(1 << 20), 1 << 20);
}

#[test]
fn channel_fixed_gradient_proof_pins_channel_cluster_order_and_fallbacks() {
    let leaf = |cluster| MaTreeNodeIr::Leaf {
        cluster,
        predictor: ModularPredictor::Gradient.index(),
        offset: 0,
        multiplier: 1,
    };
    let nodes = [
        MaTreeNodeIr::Decision {
            property: 0,
            threshold: 1,
            left: 1,
            right: 4,
        },
        MaTreeNodeIr::Decision {
            property: 0,
            threshold: 2,
            left: 2,
            right: 3,
        },
        leaf(4),
        leaf(3),
        MaTreeNodeIr::Decision {
            property: 0,
            threshold: 0,
            left: 5,
            right: 6,
        },
        leaf(2),
        leaf(1),
    ];
    assert_eq!(
        channel_fixed_gradient_specialization(&nodes, 4, false),
        ModularReconstructionSpecialization::ChannelFixed {
            predictor: ModularPredictor::Gradient,
            offset: 0,
            multiplier: 1,
            channel_count: 4,
            clusters: [1, 2, 3, 4],
        }
    );

    let mut non_channel = nodes;
    non_channel[0] = MaTreeNodeIr::Decision {
        property: 3,
        threshold: 1,
        left: 1,
        right: 4,
    };
    assert_eq!(
        channel_fixed_gradient_specialization(&non_channel, 4, false),
        ModularReconstructionSpecialization::GenericMetaAdaptive
    );

    let mut bad_unused_channel = nodes;
    bad_unused_channel[2] = MaTreeNodeIr::Leaf {
        cluster: 4,
        predictor: ModularPredictor::West.index(),
        offset: 0,
        multiplier: 1,
    };
    assert_eq!(
        channel_fixed_gradient_specialization(&bad_unused_channel, 1, false),
        ModularReconstructionSpecialization::GenericMetaAdaptive
    );

    let cycle = [MaTreeNodeIr::Decision {
        property: 0,
        threshold: 0,
        left: 0,
        right: 0,
    }];
    assert_eq!(
        channel_fixed_gradient_specialization(&cycle, 1, false),
        ModularReconstructionSpecialization::GenericMetaAdaptive
    );
    assert_eq!(
        channel_fixed_gradient_specialization(&nodes, 4, true),
        ModularReconstructionSpecialization::GenericMetaAdaptive
    );
}

#[test]
fn direct_output_proof_accepts_only_normalized_single_channel_gray8() {
    let request = GpuOutputRequest::numeric(
        Vpi::U8.pixel_format(),
        NumericSampleMapping::NormalizedGray8,
    )
    .unwrap();
    let normalized_u8 = OutputPlan::new(
        Extent2d::new(17, 13),
        &request,
        crate::ModularChannels::Gray,
        8,
        PORTABLE_CAPABILITIES,
    )
    .unwrap();
    let fixed_gray = ModularReconstructionSpecialization::ChannelFixed {
        predictor: ModularPredictor::Gradient,
        offset: 0,
        multiplier: 1,
        channel_count: 1,
        clusters: [1, 2, 3, 4],
    };
    assert_eq!(
        fixed_gradient_output_mode(1, 8, &normalized_u8, fixed_gray),
        FixedGradientOutputMode::DirectNormalizedGray8
    );
    assert_eq!(
        refine_fixed_gradient_output_mode(FixedGradientOutputMode::DirectNormalizedGray8, 1),
        FixedGradientOutputMode::CompactNormalizedGray8
    );
    assert_eq!(
        refine_fixed_gradient_output_mode(FixedGradientOutputMode::DirectNormalizedGray8, 2),
        FixedGradientOutputMode::DirectNormalizedGray8
    );
    assert_eq!(
        refine_fixed_gradient_output_mode(FixedGradientOutputMode::FinalizePass, 1),
        FixedGradientOutputMode::FinalizePass
    );
    assert_eq!(
        fixed_gradient_output_mode(
            1,
            8,
            &normalized_u8,
            ModularReconstructionSpecialization::GenericMetaAdaptive,
        ),
        FixedGradientOutputMode::FinalizePass
    );
    assert_eq!(
        fixed_gradient_output_mode(3, 8, &normalized_u8, fixed_gray),
        FixedGradientOutputMode::FinalizePass
    );

    let native_request = GpuOutputRequest::numeric(
        LosslessModularFormat::Gray.pixel_format(8).unwrap(),
        NumericSampleMapping::NativeUnsigned,
    )
    .unwrap();
    let native = OutputPlan::new(
        Extent2d::new(17, 13),
        &native_request,
        crate::ModularChannels::Gray,
        8,
        PORTABLE_CAPABILITIES,
    )
    .unwrap();
    assert_eq!(
        fixed_gradient_output_mode(1, 8, &native, fixed_gray),
        FixedGradientOutputMode::FinalizePass
    );

    let signed_request = GpuOutputRequest::numeric(
        Vpi::S8.pixel_format(),
        NumericSampleMapping::NormalizedGray8,
    )
    .unwrap();
    let signed = OutputPlan::new(
        Extent2d::new(17, 13),
        &signed_request,
        crate::ModularChannels::Gray,
        8,
        PORTABLE_CAPABILITIES,
    )
    .unwrap();
    assert_eq!(
        fixed_gradient_output_mode(1, 8, &signed, fixed_gray),
        FixedGradientOutputMode::FinalizePass
    );

    for width in [1, 3, 256, 257] {
        for height in [1, 2, 3, 257] {
            let group = ModularGroup {
                token_bit_offset: 0,
                token_bit_end: 1,
                x: 0,
                y: 0,
                width,
                height,
                stream_index: 0,
            };
            let physical_words = compact_gray8_sample_words(group).unwrap();
            let logical_words = group.sample_count().unwrap();
            assert_eq!(physical_words, width * height.min(2));
            assert!(physical_words <= logical_words);
            for y in 0..height {
                for x in 0..width {
                    assert!((y & 1) * width + x < physical_words);
                    if y != 0 {
                        assert!(((y - 1) & 1) * width + x < physical_words);
                    }
                }
            }
        }
    }
}

#[test]
fn output_negotiation_rejects_rgb_without_explicit_transfer_and_range() {
    let format = PixelFormat::rgb8(
        jxl_gpu_formats::RgbChannelOrder::Rgb,
        false,
        ColorSpecification::Undefined,
    );
    let request = GpuOutputRequest::color(format).unwrap();
    assert!(matches!(
        OutputPlan::new(
            Extent2d::new(2, 2),
            &request,
            crate::ModularChannels::Gray,
            8,
            PORTABLE_CAPABILITIES,
        ),
        Err(Error::UnsupportedOutputFormat(_))
    ));
}

#[test]
fn output_negotiation_rejects_shader_address_overflow() {
    let request = GpuOutputRequest::color(Vpi::Rgba8.pixel_format()).unwrap();
    assert!(matches!(
        OutputPlan::new(
            Extent2d::new(u32::MAX, 1),
            &request,
            crate::ModularChannels::Gray,
            8,
            PORTABLE_CAPABILITIES,
        ),
        Err(Error::Backend(_))
    ));
}

#[test]
fn output_negotiation_covers_all_vpi_pitch_linear_formats() {
    let color_formats = [
        (Vpi::Y8, OutputKind::Luma, 1, 0, 8, 1, true, 1),
        (Vpi::Y8Er, OutputKind::Luma, 1, 0, 8, 1, false, 1),
        (Vpi::Y16, OutputKind::Luma, 1, 0, 16, 1, true, 1),
        (Vpi::Y16Er, OutputKind::Luma, 1, 0, 16, 1, false, 1),
        (Vpi::Nv12, OutputKind::YuvSemiplanar, 3, 0, 8, 2, true, 1),
        (Vpi::Nv12Er, OutputKind::YuvSemiplanar, 3, 0, 8, 2, false, 1),
        (Vpi::Nv24, OutputKind::YuvSemiplanar, 3, 0, 8, 2, true, 1),
        (Vpi::Nv24Er, OutputKind::YuvSemiplanar, 3, 0, 8, 2, false, 1),
        (Vpi::Uyvy, OutputKind::Yuv422Packed, 3, 1, 8, 1, true, 1),
        (Vpi::UyvyEr, OutputKind::Yuv422Packed, 3, 1, 8, 1, false, 1),
        (Vpi::Yuyv, OutputKind::Yuv422Packed, 3, 0, 8, 1, true, 1),
        (Vpi::YuyvEr, OutputKind::Yuv422Packed, 3, 0, 8, 1, false, 1),
        (Vpi::Rgb8, OutputKind::RgbInterleaved, 3, 0, 8, 1, false, 2),
        (Vpi::Bgr8, OutputKind::RgbInterleaved, 3, 1, 8, 1, false, 2),
        (Vpi::Rgba8, OutputKind::RgbInterleaved, 4, 2, 8, 1, false, 2),
        (Vpi::Bgra8, OutputKind::RgbInterleaved, 4, 3, 8, 1, false, 2),
        (Vpi::Rgb8Planar, OutputKind::RgbPlanar, 3, 0, 8, 3, false, 2),
        (Vpi::Bgr8Planar, OutputKind::RgbPlanar, 3, 1, 8, 3, false, 2),
        (
            Vpi::Rgba8Planar,
            OutputKind::RgbPlanar,
            4,
            2,
            8,
            4,
            false,
            2,
        ),
        (
            Vpi::Bgra8Planar,
            OutputKind::RgbPlanar,
            4,
            3,
            8,
            4,
            false,
            2,
        ),
    ];
    assert_eq!(color_formats.len(), 20);
    for (format, kind, channels, order, bits, planes, limited, transfer) in color_formats {
        let pixel_format = format.pixel_format();
        assert!(matches!(
            classify_pixel_format(&pixel_format),
            Ok(PixelFormatClass::Color(_))
        ));
        let request = GpuOutputRequest::color(pixel_format).unwrap();
        let output = OutputPlan::new(
            Extent2d::new(5, 3),
            &request,
            crate::ModularChannels::Gray,
            8,
            PORTABLE_CAPABILITIES,
        )
        .unwrap_or_else(|error| panic!("{} must be supported: {error}", format.name()));
        assert_eq!(output.kind, kind, "{} kind", format.name());
        assert_eq!(output.channels, channels, "{} channels", format.name());
        assert_eq!(output.order, order, "{} order", format.name());
        assert_eq!(output.bits, bits, "{} bits", format.name());
        assert_eq!(output.storage_bits, bits, "{} storage bits", format.name());
        assert_eq!(
            output.layout.planes.len(),
            planes,
            "{} planes",
            format.name()
        );
        assert_eq!(output.limited_range, limited, "{} range", format.name());
        assert_eq!(output.transfer, transfer, "{} transfer", format.name());
        assert!(output.layout.logical_size <= u64::from(u32::MAX));
    }

    let numeric_formats = [
        Vpi::U8,
        Vpi::S8,
        Vpi::U16,
        Vpi::U32,
        Vpi::S32,
        Vpi::S16,
        Vpi::TwoS16,
        Vpi::F32,
        Vpi::F64,
        Vpi::TwoF32,
    ];
    assert_eq!(numeric_formats.len(), 10);
    for format in numeric_formats {
        let mapping = if format == Vpi::F64 {
            NumericSampleMapping::NormalizedGray8F64(F64OutputPolicy::ExactF32Widening)
        } else {
            NumericSampleMapping::NormalizedGray8
        };
        let request = GpuOutputRequest::numeric(format.pixel_format(), mapping).unwrap();
        let output = OutputPlan::new(
            Extent2d::new(5, 3),
            &request,
            crate::ModularChannels::Gray,
            8,
            PORTABLE_CAPABILITIES,
        )
        .unwrap_or_else(|error| panic!("{} must be supported: {error}", format.name()));
        let numeric = classify_pixel_format(request.format())
            .unwrap()
            .numeric()
            .unwrap();
        assert_eq!(
            output.kind,
            match numeric.sample_kind {
                SampleKind::Unsigned => OutputKind::NumericUnsigned,
                SampleKind::Signed => OutputKind::NumericSigned,
                SampleKind::Float => OutputKind::NumericFloat,
            },
            "{} kind",
            format.name()
        );
        assert_eq!(output.channels, u32::from(numeric.components));
        assert_eq!(output.bits, u32::from(numeric.bits_per_component));
        assert_eq!(output.numeric_mapping, 1);
        assert_eq!(output.layout.planes.len(), 1);
        if format == Vpi::F64 {
            let plane = &output.layout.planes[0];
            assert_eq!(output.layout.logical_size, 5 * 3 * 8);
            assert!(plane.offset.is_multiple_of(8));
            assert!(plane.row_stride.is_multiple_of(8));
            assert_eq!(
                output.f64_output_path,
                Some(F64OutputPath::ExactF32Widening)
            );
        }
    }
}

#[test]
fn output_request_requires_mapping_to_match_the_format_class() {
    assert!(matches!(
        GpuOutputRequest::color(Vpi::U8.pixel_format()),
        Err(Error::NumericMappingRequired)
    ));
    assert!(matches!(
        GpuOutputRequest::numeric(
            Vpi::Rgba8.pixel_format(),
            NumericSampleMapping::NormalizedGray8,
        ),
        Err(Error::NumericMappingForColorOutput)
    ));
    assert!(matches!(
        GpuOutputRequest::numeric(
            Vpi::F64.pixel_format(),
            NumericSampleMapping::NormalizedGray8,
        ),
        Err(Error::F64OutputPolicyRequired)
    ));
    assert!(matches!(
        GpuOutputRequest::numeric(
            Vpi::U8.pixel_format(),
            NumericSampleMapping::NormalizedGray8F64(F64OutputPolicy::NativeRequired),
        ),
        Err(Error::F64OutputPolicyForNonF64)
    ));
}

#[test]
fn f64_policy_resolution_never_silently_downgrades_native_required() {
    assert!(matches!(
        resolve_f64_output_path(F64OutputPolicy::NativeRequired, PORTABLE_CAPABILITIES),
        Err(Error::NativeF64Unavailable)
    ));
    assert_eq!(
        resolve_f64_output_path(
            F64OutputPolicy::NativeOrExactF32Widening,
            PORTABLE_CAPABILITIES,
        )
        .unwrap(),
        F64OutputPath::ExactF32Widening
    );
    let native = WgpuDecodeCapabilities {
        native_f64_arithmetic: true,
    };
    assert_eq!(
        resolve_f64_output_path(F64OutputPolicy::NativeRequired, native).unwrap(),
        F64OutputPath::NativeArithmetic
    );
    assert_eq!(
        resolve_f64_output_path(F64OutputPolicy::NativeOrExactF32Widening, native).unwrap(),
        F64OutputPath::NativeArithmetic
    );
    assert_eq!(
        resolve_f64_output_path(F64OutputPolicy::ExactF32Widening, native).unwrap(),
        F64OutputPath::ExactF32Widening
    );
}

#[test]
fn map_completion_wakes_after_releasing_state_lock() {
    let completion = Arc::new(MapCompletion::default());
    let wake = Arc::new(ReentrantCompletionWake {
        completion: Arc::clone(&completion),
        entered: AtomicBool::new(false),
    });
    let waker = Waker::from(Arc::clone(&wake));
    let context = Context::from_waker(&waker);
    assert_eq!(completion.poll(&context), None);

    completion.complete(Ok(()));
    assert!(wake.entered.load(Ordering::SeqCst));
}

#[test]
fn shader_abi_and_stream_sentinel_are_explicit() {
    assert_eq!(STREAM_OVERLAP_BYTES, 16);
    assert_eq!(STREAM_SENTINEL_BYTES, 4);
    assert_eq!(MIN_STREAM_WINDOW_BYTES, 40);
    assert_eq!(align16(1).unwrap(), 16);
    assert_eq!(align16(16).unwrap(), 16);
    assert_eq!(std::mem::size_of::<ShaderParams>(), 244);
    assert_eq!(std::mem::align_of::<ShaderParams>(), 4);
    let params = ShaderParams {
        entropy: EntropyStreamParams {
            token_start: 1,
            token_end: 2,
            lz77_window_mask: 3,
        },
        window_logical_start: 4,
        window_upload_start: 5,
        stream_token_end: 6,
        window_yield_end: 7,
        window_flags: 8,
        entropy_state_offset: 9,
        width: 10,
        height: 11,
        origin_x: 12,
        origin_y: 13,
        sample_count: 14,
        initialize_chroma: 15,
        source_channels: 16,
        channel_layout_offset: 17,
        metadata_base: 18,
        source_bits: 19,
        source_mask: 20,
        needs_self_correcting: 21,
        output_kind: 22,
        transfer: 23,
        limited_range: 24,
        channels: 25,
        order: 26,
        bits: 27,
        storage_bits: 28,
        plane0_offset: 29,
        plane0_stride: 30,
        plane1_offset: 31,
        plane1_stride: 32,
        plane2_offset: 33,
        plane2_stride: 34,
        plane3_offset: 35,
        plane3_stride: 36,
        chroma_width: 37,
        chroma_height: 38,
        logical_size: 39,
        numeric_mapping: 40,
        status_index: 41,
        stream_index: 42,
        fixed_leaf_predictor: 43,
        fixed_leaf_offset: 44,
        fixed_leaf_multiplier: 45,
        fixed_leaf_cluster0: 46,
        fixed_leaf_cluster1: 47,
        fixed_leaf_cluster2: 48,
        fixed_leaf_cluster3: 49,
        fixed_output_mode: 50,
        wp_p1: 51,
        wp_p2: 52,
        wp_p3a: 53,
        wp_p3b: 54,
        wp_p3c: 55,
        wp_p3d: 56,
        wp_p3e: 57,
        wp_w0: 58,
        wp_w1: 59,
        wp_w2: 60,
        wp_w3: 61,
    };
    assert_eq!(
        bytemuck::cast::<ShaderParams, [u32; 61]>(params),
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        ]
    );
    assert_eq!(std::mem::size_of::<EntropyExecutionState>(), 32);
    assert_eq!(std::mem::align_of::<EntropyExecutionState>(), 16);
    let execution = EntropyExecutionState {
        bit_cursor: 1,
        ans_state: 2,
        copy_remaining: 3,
        copy_position: 4,
        entropy_decoded: 5,
        last_value: 6,
        consumer_decoded: 7,
        error_code: 8,
    };
    assert_eq!(
        bytemuck::cast::<EntropyExecutionState, [u32; 8]>(execution),
        [1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(GENERIC_PREDICTOR_EXECUTION_STATE_BYTES, 48);
    let generic = GenericPredictorExecutionState {
        entropy: execution,
        predictor_prev_grad: 9,
        _padding: [0; 3],
    };
    assert_eq!(
        bytemuck::cast::<GenericPredictorExecutionState, [u32; 12]>(generic),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0]
    );
    assert_eq!(std::mem::size_of::<WeightedModularExecutionState>(), 112);
    assert_eq!(std::mem::align_of::<WeightedModularExecutionState>(), 16);
    let weighted = WeightedModularExecutionState {
        entropy: execution,
        predictor_prev_grad: 9,
        wp_true_err_w: 10,
        wp_true_err_nw: 11,
        wp_true_err_n: 12,
        wp_true_err_ne: 13,
        wp_subpred_nw_ww: [14, 15, 16, 17],
        wp_subpred_n_w: [18, 19, 20, 21],
        wp_subpred_ne: [22, 23, 24, 25],
        _padding: [0; 3],
    };
    assert_eq!(
        bytemuck::cast::<WeightedModularExecutionState, [u32; 28]>(weighted),
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 0, 0, 0,
        ]
    );
    assert_eq!(
        modular_execution_state_bytes(FIXED_GRADIENT_RECONSTRUCTION, false),
        32
    );
    assert_eq!(
        modular_execution_state_bytes(GENERIC_RECONSTRUCTION, false),
        48
    );
    assert_eq!(
        modular_execution_state_bytes(GENERIC_RECONSTRUCTION, true),
        112
    );
    assert_eq!(std::mem::size_of::<DispatchControl>(), 16);
    assert_eq!(std::mem::align_of::<DispatchControl>(), 4);
    let control = DispatchControl {
        first_group: 1,
        group_count: 2,
        lane_stride_words: 3,
        _padding: 4,
    };
    assert_eq!(
        bytemuck::cast::<DispatchControl, [u32; 4]>(control),
        [1, 2, 3, 4]
    );

    assert_eq!(std::mem::size_of::<DecodeStatus>(), 16);
    assert_eq!(std::mem::align_of::<DecodeStatus>(), 4);
    let status = DecodeStatus {
        code: 1,
        decoded_samples: 2,
        cursor: 3,
        expected_cursor: 4,
    };
    assert_eq!(
        bytemuck::cast::<DecodeStatus, [u32; 4]>(status),
        [1, 2, 3, 4]
    );
}

#[test]
fn every_reconstruction_write_and_f64_shader_variant_validates() {
    let portable_atomic = shader_source(
        F64OutputPath::ExactF32Widening,
        OutputWritePath::AtomicBytes,
        GENERIC_RECONSTRUCTION,
    );
    let portable_aligned = shader_source(
        F64OutputPath::ExactF32Widening,
        OutputWritePath::WordAligned,
        GENERIC_RECONSTRUCTION,
    );
    let native_atomic = shader_source(
        F64OutputPath::NativeArithmetic,
        OutputWritePath::AtomicBytes,
        GENERIC_RECONSTRUCTION,
    );
    let native_aligned = shader_source(
        F64OutputPath::NativeArithmetic,
        OutputWritePath::WordAligned,
        GENERIC_RECONSTRUCTION,
    );
    let descriptor_atomic = shader_source(
        F64OutputPath::ExactF32Widening,
        OutputWritePath::AtomicBytes,
        ModularReconstructionSpecialization::DescriptorMetaAdaptive,
    );
    let descriptor_aligned = shader_source(
        F64OutputPath::ExactF32Widening,
        OutputWritePath::WordAligned,
        ModularReconstructionSpecialization::DescriptorMetaAdaptive,
    );
    let native_descriptor_atomic = shader_source(
        F64OutputPath::NativeArithmetic,
        OutputWritePath::AtomicBytes,
        ModularReconstructionSpecialization::DescriptorMetaAdaptive,
    );
    let native_descriptor_aligned = shader_source(
        F64OutputPath::NativeArithmetic,
        OutputWritePath::WordAligned,
        ModularReconstructionSpecialization::DescriptorMetaAdaptive,
    );
    let fixed_gradient_atomic = shader_source(
        F64OutputPath::ExactF32Widening,
        OutputWritePath::AtomicBytes,
        FIXED_GRADIENT_RECONSTRUCTION,
    );
    let fixed_gradient_aligned = shader_source(
        F64OutputPath::ExactF32Widening,
        OutputWritePath::WordAligned,
        FIXED_GRADIENT_RECONSTRUCTION,
    );
    let native_fixed_gradient_atomic = shader_source(
        F64OutputPath::NativeArithmetic,
        OutputWritePath::AtomicBytes,
        FIXED_GRADIENT_RECONSTRUCTION,
    );
    let native_fixed_gradient_aligned = shader_source(
        F64OutputPath::NativeArithmetic,
        OutputWritePath::WordAligned,
        FIXED_GRADIENT_RECONSTRUCTION,
    );
    let native_without_capability = naga::front::wgsl::parse_str(&native_aligned)
        .expect("native F64 WGSL syntax must parse before capability validation");
    let error = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&native_without_capability)
    .expect_err("native F64 WGSL must be rejected without Naga FLOAT64 capability");
    assert!(format!("{error:?}").contains("FLOAT64"));

    for (name, source, capabilities) in [
        (
            "portable-atomic",
            portable_atomic,
            naga::valid::Capabilities::empty(),
        ),
        (
            "portable-aligned",
            portable_aligned,
            naga::valid::Capabilities::empty(),
        ),
        (
            "native-f64-atomic",
            native_atomic,
            naga::valid::Capabilities::FLOAT64,
        ),
        (
            "native-f64-aligned",
            native_aligned,
            naga::valid::Capabilities::FLOAT64,
        ),
        (
            "portable-descriptor-atomic",
            descriptor_atomic,
            naga::valid::Capabilities::empty(),
        ),
        (
            "portable-descriptor-aligned",
            descriptor_aligned,
            naga::valid::Capabilities::empty(),
        ),
        (
            "native-f64-descriptor-atomic",
            native_descriptor_atomic,
            naga::valid::Capabilities::FLOAT64,
        ),
        (
            "native-f64-descriptor-aligned",
            native_descriptor_aligned,
            naga::valid::Capabilities::FLOAT64,
        ),
        (
            "portable-fixed-gradient-atomic",
            fixed_gradient_atomic,
            naga::valid::Capabilities::empty(),
        ),
        (
            "portable-fixed-gradient-aligned",
            fixed_gradient_aligned,
            naga::valid::Capabilities::empty(),
        ),
        (
            "native-f64-fixed-gradient-atomic",
            native_fixed_gradient_atomic,
            naga::valid::Capabilities::FLOAT64,
        ),
        (
            "native-f64-fixed-gradient-aligned",
            native_fixed_gradient_aligned,
            naga::valid::Capabilities::FLOAT64,
        ),
    ] {
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{name} WGSL did not parse: {error}"));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name} WGSL did not validate: {error:?}"));
    }
}

#[test]
fn aggregate_memory_reservations_are_bounded_and_released() {
    let budget = MemoryBudget::new(NonZeroU64::new(10).unwrap());
    let first = budget.try_reserve(6).unwrap();
    assert_eq!(budget.snapshot().reserved_bytes, 6);
    assert!(matches!(
        budget.try_reserve(5),
        Err(jxl_wgpu::MemoryBudgetError::Exhausted { .. })
    ));
    assert_eq!(budget.snapshot().reserved_bytes, 6);
    drop(first);
    assert_eq!(budget.snapshot().reserved_bytes, 0);
}
