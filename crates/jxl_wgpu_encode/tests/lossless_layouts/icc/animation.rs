use super::*;
use jxl_wgpu_encode::{
    AnimationHeader, FrameOptions, FrameTiming, LosslessModularAnimationDescriptor,
};
use std::num::NonZeroU32;

#[test]
fn icc_animation_binds_profile_identity_and_retains_one_header_reservation() {
    let rig = Rig::new();
    let native_profile = IccProfileOracle::compile();
    for format in [
        LosslessModularFormat::GrayAlpha,
        LosslessModularFormat::Rgba,
    ] {
        let profile = profile(format.color_channel_count() == 1);
        for (kind, bits) in [(SampleKind::Unsigned, 31), (SampleKind::Float, 32)] {
            let encoder = encoder(&rig, &profile, TREES[1])
                .with_alpha_association(AlphaAssociation::Associated);
            let case = Case {
                format,
                bits,
                kind,
                storage: Storage::Planar,
                reversed: true,
                byte_order: ByteOrder::Big,
                shifted: true,
            };
            let extent = Extent2d::new(257, 3);
            let mut template = upload(&rig.context, &case, extent, &case.samples(extent), 4099);
            attach(&mut template, &profile, true);
            let descriptor = LosslessModularAnimationDescriptor::from_pixel_format(
                extent.width,
                extent.height,
                &template.layout.format,
                AnimationHeader::Animation {
                    ticks_per_second_numerator: NonZeroU32::new(100).unwrap(),
                    ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
                    num_loops: 3,
                    have_timecodes: true,
                },
            )
            .unwrap();
            let mut assembly = encoder.begin_animation(descriptor.clone()).unwrap();
            assert_eq!(assembly.descriptor(), &descriptor);
            let header_reservation = encoder.in_flight_memory_stats().reserved_bytes;
            assert!(header_reservation > 2 * profile.bytes().len() as u64);
            let mut changed = profile.bytes().to_vec();
            changed[80] ^= 1; // same color method, different original profile identity
            let changed = IccProfile::parse(changed.into(), Default::default()).unwrap();
            let mut invalid = template.clone();
            invalid.layout.format.color_spec = ColorSpecification::Icc(changed);
            assert!(matches!(
                assembly.submit_frame(invalid, FrameOptions::default()),
                Err(EncodeError::InvalidConfiguration(_))
            ));
            assert_eq!(assembly.next_frame_index().get(), 0);
            assert_eq!(
                encoder.in_flight_memory_stats().reserved_bytes,
                header_reservation
            );
            let mut jobs = Vec::new();
            let mut expected = Vec::new();
            for index in 0..3 {
                let mut samples = case.samples(extent);
                samples.rotate_left(index * format.channel_count() as usize);
                let mut source = upload(
                    &rig.context,
                    &Case {
                        storage: if index % 2 == 0 {
                            Storage::Packed
                        } else {
                            Storage::Planar
                        },
                        byte_order: if index % 2 == 0 {
                            ByteOrder::Little
                        } else {
                            ByteOrder::Big
                        },
                        ..case
                    },
                    extent,
                    &samples,
                    8195,
                );
                // Equal bytes parsed independently still satisfy the stream identity contract.
                let independent =
                    IccProfile::parse(profile.bytes().to_vec().into(), Default::default()).unwrap();
                attach(&mut source, &independent, index % 2 == 0);
                let options = FrameOptions {
                    timing: FrameTiming {
                        duration_ticks: index as u32 + 2,
                        timecode: Some(index as u32 + 20),
                    },
                    ..Default::default()
                };
                jobs.push(
                    if index == 2 {
                        assembly.submit_last_frame(source, options)
                    } else {
                        assembly.submit_frame(source, options)
                    }
                    .unwrap(),
                );
                expected.push(samples);
            }
            for (index, job) in jobs.into_iter().rev().enumerate() {
                assembly
                    .insert(if index % 2 == 0 {
                        job.wait().unwrap()
                    } else {
                        pollster::block_on(job).unwrap()
                    })
                    .unwrap();
            }
            assert_eq!(
                encoder.in_flight_memory_stats().reserved_bytes,
                header_reservation
            );
            let encoded = assembly.finish_container().unwrap();
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(
                native_profile.read(&encoded).profile,
                profile.bytes().as_ref()
            );
            check_frame_samples(
                &encoded,
                &expected.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &case,
                &original(&encoded),
            );
            color::check_numeric(&rig, &encoded, &expected, &case);
            drop(encoder.begin_animation(descriptor).unwrap());
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
