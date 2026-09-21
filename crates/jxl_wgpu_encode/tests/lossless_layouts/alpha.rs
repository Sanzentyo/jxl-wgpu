use super::*;
use jxl_wgpu_encode::AlphaAssociation;

mod animation;
mod lifetime;
mod output;

#[test]
fn gray_alpha_and_associated_sources_preserve_every_integer_depth_and_ieee_word() {
    let rig = Rig::new();
    for (format, association) in [
        (
            LosslessModularFormat::GrayAlpha,
            AlphaAssociation::Unassociated,
        ),
        (
            LosslessModularFormat::GrayAlpha,
            AlphaAssociation::Associated,
        ),
        (LosslessModularFormat::Rgba, AlphaAssociation::Associated),
    ] {
        for tree in TREES {
            let encoder = LosslessModularEncoder::with_tree_mode(rig.context.clone(), tree)
                .with_alpha_association(association);
            for bits in 1..=33 {
                let (kind, bits) = match bits {
                    32 => (SampleKind::Float, 16),
                    33 => (SampleKind::Float, 32),
                    bits => (SampleKind::Unsigned, bits),
                };
                let case = Case {
                    format,
                    kind,
                    bits,
                    storage: if bits % 2 == 0 {
                        Storage::Packed
                    } else {
                        Storage::Planar
                    },
                    reversed: true,
                    byte_order: ByteOrder::Big,
                    shifted: true,
                };
                let extent = Extent2d::new(257, 3);
                let expected = case.samples(extent);
                let source = upload(&rig.context, &case, extent, &expected, 4099);
                let plan = encoder.memory_plan(&source).unwrap();
                assert_eq!(plan.channel_count, format.channel_count());
                assert_eq!(plan.format, format);
                let encoded =
                    pollster::block_on(encoder.submit_container(source).unwrap()).unwrap();
                let canonical = upload(&rig.context, &case.canonical(), extent, &expected, 0);
                assert_eq!(encoded, encoder.encode_container(canonical).unwrap());
                let header = jxl_gpu_bitstream::parse(&encoded, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap()
                    .image_header;
                assert_eq!(header.grayscale, format == LosslessModularFormat::GrayAlpha);
                assert_eq!(header.extra_channels.len(), 1);
                assert_eq!(header.bit_depth, plan.sample_bit_depth());
                assert_eq!(header.extra_channels[0].bit_depth, header.bit_depth);
                assert_eq!(header.extra_channels[0].dimension_shift, 0);
                assert_eq!(
                    header.extra_channels[0].channel_type,
                    jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha {
                        associated: association == AlphaAssociation::Associated,
                    }
                );
                check_oracles(&encoded, &expected, &case);
                color::check_numeric(&rig, &encoded, &[expected], &case);
                assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}
