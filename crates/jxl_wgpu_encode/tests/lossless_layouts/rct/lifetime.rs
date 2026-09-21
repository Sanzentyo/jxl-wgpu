use super::*;
use jxl_wgpu_encode::{AnimationHeader, LosslessModularAnimationDescriptor};
use std::num::NonZeroU32;

#[test]
fn local_and_global_rct_keep_admission_cancellation_and_streamed_reuse() {
    let rig = Rig::new();
    for local in [false, true] {
        for value in [4, 5, 6, 41] {
            let config = config(value, local, value as usize % 2);
            let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config);
            groups::lifetime::check_admission(
                &rig,
                &encoder,
                config.group_size,
                if value % 2 == 0 {
                    SampleKind::Float
                } else {
                    SampleKind::Unsigned
                },
            );
        }
    }
}

#[test]
fn gray_rct_rejects_before_reservation_or_submission() {
    let rig = Rig::new();
    for local in [false, true] {
        let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config(6, local, 0));
        for format in [
            LosslessModularFormat::Gray,
            LosslessModularFormat::GrayAlpha,
        ] {
            for kind in [SampleKind::Unsigned, SampleKind::Float] {
                let case = Case {
                    format,
                    bits: 16,
                    kind,
                    storage: Storage::Planar,
                    reversed: true,
                    byte_order: ByteOrder::Big,
                    shifted: true,
                };
                let extent = Extent2d::new(17, 3);
                let source = upload(&rig.context, &case, extent, &case.samples(extent), 4099);
                assert!(matches!(
                    encoder.memory_plan(&source),
                    Err(EncodeError::ModularRctColorChannels { color_channels: 1 })
                ));
                assert!(matches!(
                    encoder.submit(source.clone()),
                    Err(EncodeError::ModularRctColorChannels { color_channels: 1 })
                ));
                let descriptor = LosslessModularAnimationDescriptor::from_pixel_format(
                    17,
                    3,
                    &source.layout.format,
                    AnimationHeader::Animation {
                        ticks_per_second_numerator: NonZeroU32::new(100).unwrap(),
                        ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
                        num_loops: 0,
                        have_timecodes: false,
                    },
                )
                .unwrap();
                assert!(matches!(
                    encoder.begin_animation(descriptor),
                    Err(EncodeError::ModularRctColorChannels { color_channels: 1 })
                ));
                assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
                assert_eq!(encoder.buffer_pool_stats().allocation_misses, 0);
            }
        }
    }
}
