use super::*;
use crate::{BitWriter, gain_map::GainMapBundle};

fn enum_value(bits: &mut BitWriter, value: u64) {
    if value < 2 {
        bits.write_bits(value, 2).unwrap();
    } else {
        bits.write_bits(2, 2).unwrap();
        bits.write_bits(value - 2, 4).unwrap();
    }
}

#[test]
fn xyb_implicit_fields_preserve_intent_and_following_bits() {
    for intent in 0..4 {
        let mut bits = BitWriter::new();
        bits.write_bits(0, 2).unwrap(); // explicit, no ICC
        enum_value(&mut bits, 2); // XYB
        enum_value(&mut bits, intent);
        let length = bits.bit_len();
        let color = bits.clone().into_bytes();
        bits.write_bits(0x6d, 7).unwrap();
        let mut reader = Bitstream::new(bits.as_bytes());
        let ColourEncoding::Enum(e) = parse(&mut reader).unwrap() else {
            panic!("enumerated XYB");
        };
        assert_eq!(e.colour_space, ColourSpace::Xyb);
        assert_eq!(e.rendering_intent as u64, intent);
        assert_eq!(
            e.tf,
            TransferFunction::Gamma {
                g: 3_333_333,
                inverted: true
            }
        );
        assert_eq!(reader.num_read_bits(), length);
        assert_eq!(reader.read_bits(7).unwrap(), 0x6d);
        for end in 0..color.len() {
            assert!(parse(&mut Bitstream::new(&color[..end])).is_err());
        }
        let bundle = GainMapBundle::new(
            Default::default(),
            &color,
            &[],
            &[0xff, 0x0a],
            Default::default(),
        )
        .unwrap();
        assert!(matches!(
            bundle.alternate_color_encoding(),
            Some(crate::ColourEncodingInventory::Enumerated {
                colour_space: crate::ColourSpaceInventory::Xyb,
                ..
            })
        ));
        let mut bad_padding = color;
        *bad_padding.last_mut().unwrap() |= 0x80;
        assert!(
            GainMapBundle::new(
                Default::default(),
                &bad_padding,
                &[],
                &[0xff, 0x0a],
                Default::default()
            )
            .is_err()
        );
    }
}

#[test]
fn gamma_wire_boundaries_and_truncation_are_validated() {
    for gamma in [0, 1220, 1221, 3_333_333, 10_000_000, 10_000_001, 0xffffff] {
        let mut bits = BitWriter::new();
        bits.write_bits(0, 2).unwrap();
        enum_value(&mut bits, 0); // RGB
        enum_value(&mut bits, 1); // D65
        enum_value(&mut bits, 1); // sRGB primaries
        bits.write_bits(1, 1).unwrap();
        bits.write_bits(gamma, 24).unwrap();
        enum_value(&mut bits, 1);
        assert_eq!(
            parse(&mut Bitstream::new(bits.as_bytes())).is_ok(),
            (1221..=10_000_000).contains(&gamma)
        );
        for end in 0..bits.as_bytes().len() {
            assert!(parse(&mut Bitstream::new(&bits.as_bytes()[..end])).is_err());
        }
    }
}
