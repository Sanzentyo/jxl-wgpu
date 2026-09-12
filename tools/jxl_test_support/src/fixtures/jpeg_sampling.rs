//! Header-only mutations of independent JPEG-recompression entropy for conformance tests.
use jxl_gpu_bitstream::{BitReader, BitWriter};

/// Set nonzero LF-only correlation without changing packet lengths or any entropy bits.
pub fn correlated(data: &[u8]) -> Vec<u8> {
    use jxl_wgpu_decode::vardct::frontend::LfGlobalPrefix;
    let inventory = jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let mut range = inventory.frames[0].sections[0].bits;
    assert_eq!(inventory.frames[0].flags & 1, 1);
    range.offset += 80;
    range.length -= 80;
    let prefix = LfGlobalPrefix::parse(data, range).unwrap();
    assert_eq!(prefix.lf_correlation.colour_factor, 84);
    assert_eq!(prefix.lf_correlation.base, [0.0; 2]);
    assert_eq!(prefix.lf_correlation.lf_factors, [0; 2]);
    // The explicit two eight-bit LF factors precede the final global-MA-tree flag.
    let start = prefix.suffix_bit_offset as usize - 17;
    let mut output = data.to_vec();
    for (index, value) in [144_u8, 104].into_iter().enumerate() {
        for bit in 0..8 {
            let at = start + index * 8 + bit;
            let mask = 1 << (at % 8);
            output[at / 8] = (output[at / 8] & !mask) | (((value >> bit) & 1) << (at % 8));
        }
    }
    let checked = LfGlobalPrefix::parse(&output, range).unwrap();
    let mut expected = prefix;
    expected.lf_correlation.lf_factors = [16, -24];
    assert_eq!(checked, expected);
    output
}

fn copy_bits(writer: &mut BitWriter, data: &[u8], start: u64, end: u64) {
    let mut reader = BitReader::new(data);
    reader.skip_bits(start).unwrap();
    let mut remaining = end.checked_sub(start).unwrap();
    while remaining != 0 {
        let count = remaining.min(56) as u8;
        writer
            .write_bits(reader.read_bits(count).unwrap(), count)
            .unwrap();
        remaining -= u64::from(count);
    }
}

fn flags(writer: &mut BitWriter, value: u64) {
    match value {
        0 => writer.write_bits(0, 2).unwrap(),
        1..=16 => {
            writer.write_bits(1, 2).unwrap();
            writer.write_bits(value - 1, 4).unwrap();
        }
        17..=272 => {
            writer.write_bits(2, 2).unwrap();
            writer.write_bits(value - 17, 8).unwrap();
        }
        _ => panic!("unexpected fixture flags"),
    }
}

/// Preserve all image entropy while changing sampling selectors and the smoothing flag.
/// The result may intentionally violate the LF-smoothing constraint; it is not reparsed here.
pub fn rewrite(data: &[u8], selectors: [u32; 3], smoothing: bool) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    assert_eq!(inventory.frames.len(), 1);
    let frame = &inventory.frames[0];
    assert!(frame.do_ycbcr && !frame.uses_lf_frame() && !frame.toc_permuted);
    assert!(selectors.into_iter().all(|value| value < 4));
    let data = parsed.codestream();
    let start = frame.header_bits.offset;
    let mut reader = BitReader::new(data);
    reader.skip_bits(start).unwrap();
    assert_eq!(reader.read_bits(1).unwrap(), 0); // explicit frame header
    let mut old_flags = BitWriter::new();
    flags(&mut old_flags, frame.flags);
    let flags_end = start + 4 + old_flags.bit_len() as u64;
    let mut writer = BitWriter::new();
    copy_bits(&mut writer, data, 0, start + 4);
    flags(
        &mut writer,
        (frame.flags & !128) | if smoothing { 0 } else { 128 },
    );
    writer.write_bits(1, 1).unwrap(); // do_YCbCr
    for value in selectors {
        writer.write_bits(u64::from(value), 2).unwrap();
    }
    copy_bits(
        &mut writer,
        data,
        flags_end + 7,
        frame.header_bits.end().unwrap(),
    );
    writer.write_bits(0, 1).unwrap(); // no TOC permutation
    writer.align_to_byte().unwrap();
    for section in &frame.sections {
        let length = section.bytes.length;
        let (selector, (base, count)) = [(0, 10), (1024, 14), (17408, 22), (4211712, 30)]
            .into_iter()
            .enumerate()
            .find(|(_, (base, count))| (*base..*base + (1 << count)).contains(&length))
            .unwrap();
        writer.write_bits(selector as u64, 2).unwrap();
        writer.write_bits(length - base, count).unwrap();
    }
    writer.align_to_byte().unwrap();
    let mut output = writer.into_bytes();
    for section in &frame.sections {
        output.extend_from_slice(
            &data[section.bytes.offset as usize..section.bytes.end().unwrap() as usize],
        );
    }
    output
}
