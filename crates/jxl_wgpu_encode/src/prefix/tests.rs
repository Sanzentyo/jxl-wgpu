use super::*;

#[test]
fn raw_u32_alphabet_roundtrips_every_exponent_and_long_canonical_codes() {
    let values = std::iter::once(0)
        .chain((0..32).flat_map(|exponent| {
            let lower = 1u32 << exponent;
            [lower, lower | (lower / 3), lower | (lower - 1)]
        }))
        .collect::<Vec<_>>();
    for counts in [[1; 33], std::array::from_fn(|index| 1u64 << index)] {
        let code = RawPrefixCode::<33>::from_counts(&counts).unwrap();
        if counts[32] > 1 {
            assert!(
                code.raw_entries()
                    .iter()
                    .any(|entry| entry.bit_len > 8 && entry.bits > 255)
            );
        }
        let mut writer = BitWriter::new();
        writer.write_bits(0, 1).unwrap(); // no LZ77; one implicit cluster
        writer.write_bits(1, 1).unwrap(); // prefix histogram
        writer.write_bits(0, 4).unwrap(); // split exponent zero
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(5, 4).unwrap();
        writer.write_bits(0, 5).unwrap(); // alphabet size 33
        code.write_raw_tree(&mut writer).unwrap();
        for &value in &values {
            let extra = 31u32.saturating_sub(value.leading_zeros());
            let token = u32::from(value != 0) + extra;
            code.write_raw(&mut writer, token, extra, value.saturating_sub(1 << extra))
                .unwrap();
        }
        writer.write_bits(0xd3, 8).unwrap();
        let bytes = writer.into_bytes();
        let mut bits = jxl_bitstream::Bitstream::new(&bytes);
        let mut decoder = jxl_coding::Decoder::parse(&mut bits, 1).unwrap();
        for &value in &values {
            assert_eq!(decoder.read_varint(&mut bits, 0).unwrap(), value);
        }
        decoder.finalize().unwrap();
        assert_eq!(bits.read_bits(8).unwrap(), 0xd3);
    }
}
