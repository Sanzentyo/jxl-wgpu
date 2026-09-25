use super::*;
use crate::{AlphaAssociation, FiniteF16};
use jxl_oxide_common::Bundle;

fn kinds() -> Vec<ExtraChannelKind> {
    vec![
        ExtraChannelKind::Alpha(AlphaAssociation::Unassociated),
        ExtraChannelKind::Alpha(AlphaAssociation::Associated),
        ExtraChannelKind::Depth,
        ExtraChannelKind::SpotColor {
            rgba: [0x3400, 0x3800, 0x3c00, 0x3a00].map(|bits| FiniteF16::from_bits(bits).unwrap()),
        },
        ExtraChannelKind::SelectionMask,
        ExtraChannelKind::Black,
        ExtraChannelKind::Cfa { channel: 0 },
        ExtraChannelKind::Cfa { channel: 1 },
        ExtraChannelKind::Cfa { channel: 3 },
        ExtraChannelKind::Cfa { channel: 18 },
        ExtraChannelKind::Cfa { channel: 19 },
        ExtraChannelKind::Cfa { channel: 274 },
        ExtraChannelKind::Thermal,
        ExtraChannelKind::Optional,
    ]
}

fn expected(kind: ExtraChannelKind) -> jxl_image::ExtraChannelType {
    use jxl_image::ExtraChannelType as T;
    match kind {
        ExtraChannelKind::Alpha(association) => T::Alpha {
            alpha_associated: association == AlphaAssociation::Associated,
        },
        ExtraChannelKind::Depth => T::Depth,
        ExtraChannelKind::SpotColor { rgba } => T::SpotColour {
            red: rgba[0].to_f32(),
            green: rgba[1].to_f32(),
            blue: rgba[2].to_f32(),
            solidity: rgba[3].to_f32(),
        },
        ExtraChannelKind::SelectionMask => T::SelectionMask,
        ExtraChannelKind::Black => T::Black,
        ExtraChannelKind::Cfa { channel } => T::Cfa {
            cfa_channel: channel,
        },
        ExtraChannelKind::Thermal => T::Thermal,
        ExtraChannelKind::Optional => T::Optional,
    }
}

#[test]
fn extra_input_metadata_kinds_names_and_count_bounds_use_independent_header_reader() {
    let precision = SamplePrecision::integer(13).unwrap();
    let definitions: Vec<_> = kinds()
        .into_iter()
        .cycle()
        .zip([0, 1, 15, 16, 47, 48, 1071].into_iter().cycle())
        .enumerate()
        .take(28)
        .map(|(i, (kind, len))| {
            ExtraChannel::new(kind, precision, (i % 4) as u8, vec![b'x'; len]).unwrap()
        })
        .collect();
    let budget = jxl_wgpu::MemoryBudget::new(NonZeroU64::new(1 << 24).unwrap());
    for count in [0, 1, 2, 17, 18, 256] {
        let extras: Vec<_> = definitions.iter().cloned().cycle().take(count).collect();
        let config = VarDctConfig {
            extra_channels: extras.clone(),
            max_extra_channel_metadata_bytes: 1 << 24,
            ..Default::default()
        };
        let plan = VarDctColorPlan::new(&config).unwrap();
        let descriptor =
            crate::ImageSequenceDescriptor::new(1, 1, crate::AnimationHeader::Still).unwrap();
        let (bytes, permit) = plan
            .image_header(&descriptor)
            .unwrap()
            .finish(&budget)
            .unwrap();
        let header =
            jxl_image::ImageHeader::parse(&mut jxl_bitstream::Bitstream::new(bytes.bytes()), ())
                .unwrap();
        assert_eq!(header.metadata.ec_info.len(), count);
        for (parsed, expected_definition) in header.metadata.ec_info.iter().zip(&extras) {
            assert_eq!(parsed.ty, expected(expected_definition.kind()));
            assert_eq!(parsed.name.as_bytes(), expected_definition.name());
            assert_eq!(parsed.bit_depth.bits_per_sample(), 13);
            assert_eq!(
                parsed.dim_shift,
                u32::from(expected_definition.dimension_shift())
            );
        }
        drop(permit);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
    }
    for (kind, shift, name) in [
        (ExtraChannelKind::Depth, 4, vec![]),
        (ExtraChannelKind::Depth, 8, vec![]),
        (ExtraChannelKind::Depth, 0, vec![b'x'; 1072]),
        (ExtraChannelKind::Depth, 0, vec![0xff]),
        (ExtraChannelKind::Cfa { channel: 275 }, 0, vec![]),
    ] {
        assert!(matches!(
            ExtraChannel::new(kind, precision, shift, name),
            Err(EncodeError::InvalidConfiguration(_))
        ));
    }
    let unicode = ExtraChannel::new(
        ExtraChannelKind::Depth,
        precision,
        0,
        "深度".as_bytes().to_vec(),
    )
    .unwrap();
    for (count, limit, packed) in [
        (257, 1 << 24, false),
        (4096, 1 << 24, false),
        (256, 1 << 24, true),
        (1, 37, false),
    ] {
        let config = VarDctConfig {
            extra_channels: vec![unicode.clone(); count],
            max_extra_channel_metadata_bytes: limit,
            alpha: packed.then_some(AlphaAssociation::Unassociated),
            ..Default::default()
        };
        assert!(matches!(
            VarDctColorPlan::new(&config),
            Err(EncodeError::InvalidConfiguration(_))
        ));
    }
}

#[test]
fn extra_input_supported_kinds_survive_native_decode_and_exact_scalar_reconstruction() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(9, 7);
    let definitions: Vec<_> = kinds()
        .into_iter()
        .map(|kind| {
            ExtraChannel::new(
                kind,
                SamplePrecision::integer(13).unwrap(),
                0,
                "追加".as_bytes().to_vec(),
            )
            .unwrap()
        })
        .collect();
    let words: Vec<Vec<u32>> = (0..definitions.len())
        .map(|index| {
            (0..extent.area().unwrap())
                .map(|i| ((i * 317 + index * 63) % 8192) as u32)
                .collect()
        })
        .collect();
    let source = color_source(&context, extent)
        .with_extra_channels(
            definitions
                .iter()
                .zip(&words)
                .map(|(d, w)| scalar_source(&context, extent, d.precision(), w))
                .collect(),
        )
        .unwrap();
    let encoder = TiledVarDctEncoder::new_with_config(
        context.clone(),
        VarDctConfig {
            extra_channels: definitions.clone(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(encoder.extra_channels(), definitions);
    let bytes = encoder.encode(source).unwrap();
    check_words(&bytes, 0, &definitions, extent, &words);
    let (_, native) =
        extra_channels::libjxl_planes(&bytes, extent.area().unwrap(), definitions.len()).unwrap();
    for ((actual, words), d) in native.iter().zip(&words).zip(&definitions) {
        assert_numeric(actual, words, d.precision());
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
