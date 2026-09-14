use crate::TransformKind;

#[test]
fn every_default_matrix_and_natural_order_matches_pinned_native_libjxl() {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test-data/vardct_metadata.bin"
    ))
    .expect("committed native quantization oracle");
    assert_eq!(&bytes[..8], b"JXLQNT01");
    let mut words = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word));
    assert_eq!(words.next(), Some(27));
    let mut peak = 0.0_f32;
    for (id, transform) in TransformKind::ALL.into_iter().enumerate() {
        let extent = transform.pixel_extent();
        let area = extent.area().unwrap();
        assert_eq!(words.next(), Some(id as u32));
        assert_eq!(words.next(), Some(area as u32));
        let expected_order = words.by_ref().take(area).collect::<Vec<_>>();
        let order = transform.natural_order();
        assert_eq!(order, expected_order, "{transform:?}");
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..area as u32).collect::<Vec<_>>());
        let width = extent.width.max(extent.height) as usize;
        let lf_count = transform.lf_extent().area().unwrap();
        assert!(order[..lf_count].iter().all(|&index| {
            (index as usize % width) < width / 8
                && (index as usize / width) < extent.width.min(extent.height) as usize / 8
        }));
        let matrix = transform.default_dequant_matrix();
        assert_eq!(matrix.transform, transform);
        assert_eq!(matrix.scales.len(), area);
        for channel in 0..3 {
            for (index, actual) in matrix.scales.iter().enumerate() {
                let expected = f32::from_bits(words.next().unwrap());
                let actual = actual[channel];
                assert!(actual.is_finite() && actual > 0.0);
                // LLF values use the separate LF quantizer. Native DCT2 puts
                // an explicit 0xBAD sentinel in its unused DC matrix entry;
                // compare only the AC entries that either codec consumes.
                if index % width < width / 8
                    && index / width < extent.width.min(extent.height) as usize / 8
                {
                    continue;
                }
                let relative = (actual - expected).abs() / expected;
                // Independent native fast-power and Rust power implementations
                // may round differently. This bound is set before executing it.
                assert!(
                    relative <= 3e-6,
                    "{transform:?}, channel {channel}, index {index}: {actual} != {expected}, relative {relative}"
                );
                peak = peak.max(relative);
            }
        }
    }
    assert_eq!(words.next(), None);
    eprintln!("27 native matrix/order cases; peak relative matrix error={peak:e}");
}
