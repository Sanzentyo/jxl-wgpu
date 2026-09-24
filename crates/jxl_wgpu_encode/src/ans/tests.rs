use super::*;

#[test]
fn integer_rate_estimate_covers_all_frequencies_without_float_selection() {
    let mut previous = u128::MAX;
    for frequency in 1..=TABLE_SIZE as u32 {
        let mut frequencies = [0; ALPHABET];
        frequencies[0] = frequency;
        frequencies[1] = TABLE_SIZE as u32 - frequency;
        let code = AnsCode::from_frequencies(frequencies).unwrap();
        let mut counts = [0; ALPHABET];
        counts[0] = 1;
        let estimate = code.estimated_data_bits(&counts).unwrap();
        assert!(estimate <= previous);
        previous = estimate;
        let error = estimate as f64 / 1_048_576.0 - (12.0 - f64::from(frequency).log2());
        assert!(
            (-1e-9..=2f64.powi(-19)).contains(&error),
            "{frequency}: {error}"
        );
    }
    let mut frequencies = [0; ALPHABET];
    frequencies[255] = TABLE_SIZE as u32;
    let code = AnsCode::from_frequencies(frequencies).unwrap();
    let mut counts = [0; ALPHABET];
    counts[255] = u64::MAX;
    assert_eq!(code.estimated_data_bits(&counts).unwrap(), 0);
    counts[0] = 1;
    assert!(code.estimated_data_bits(&counts).is_err());
    let uniform = AnsCode::from_frequencies([16; ALPHABET]).unwrap();
    assert_eq!(
        uniform.estimated_data_bits(&[u64::MAX; ALPHABET]).unwrap(),
        u128::from(u64::MAX) * 256 * 8 * (1 << 20)
    );
}

#[test]
fn independent_decoder_checks_every_alias_residue_and_histogram_form() {
    for log in 5u8..=8 {
        let alphabet = AnsAlphabet::for_symbols(1 << log).unwrap();
        let symbols = alphabet.symbols();
        let mut distributions = vec![
            [0; ALPHABET],
            std::array::from_fn(|symbol| if symbol < symbols { u64::MAX } else { 0 }),
        ];
        for symbol in 0..symbols {
            let mut counts = [0; ALPHABET];
            counts[symbol] = u64::MAX;
            distributions.push(counts);
        }
        for active in 2..=symbols {
            // Sparse, skewed populations force every count width and non-final omitted bins.
            let counts = std::array::from_fn(|symbol| {
                if symbol < active {
                    1u64 << ((symbol * 19 + active) % 64)
                } else {
                    0
                }
            });
            distributions.push(counts);
        }
        let mut sparse = [0; ALPHABET];
        sparse[1] = 1;
        sparse[symbols - 1] = u64::MAX;
        distributions.push(sparse);
        for counts in distributions {
            let code = AnsHistogram::from_counts(&counts, alphabet)
                .unwrap()
                .compile()
                .unwrap();
            assert_eq!(
                code.gpu_words(),
                AnsHistogram::from_counts(&counts, alphabet)
                    .unwrap()
                    .compile()
                    .unwrap()
                    .gpu_words()
            );
            assert_eq!(
                code.histogram.frequencies.iter().sum::<u32>(),
                TABLE_SIZE as u32
            );
            if counts.iter().any(|&count| count != 0) {
                for (count, frequency) in counts.iter().zip(code.histogram.frequencies) {
                    assert_eq!(*count == 0, frequency == 0);
                }
            }
            let mut writer = BitWriter::new();
            writer.write_bits(0, 1).unwrap(); // no LZ77, implicit single context
            writer.write_bits(0, 1).unwrap(); // ANS
            writer.write_bits(u64::from(log - 5), 2).unwrap();
            writer
                .write_bits(u64::from(log), (8 - log.leading_zeros()) as u8)
                .unwrap(); // literal symbols
            code.write_histogram(&mut writer).unwrap();
            let header_bits = writer.bit_len();
            writer.align_to_byte().unwrap();
            let bytes = writer.into_bytes();
            let mut bits = jxl_bitstream::Bitstream::new(&bytes);
            let decoder = jxl_coding::Decoder::parse(&mut bits, 1).unwrap();
            assert_eq!(bits.num_read_bits(), header_bits);
            let mut visited = [false; TABLE_SIZE];
            for (symbol, &frequency) in code.histogram.frequencies.iter().enumerate() {
                let offset = code.words[ALPHABET + symbol] as usize;
                for rank in 0..frequency as usize {
                    let residue = code.words[2 * ALPHABET + offset + rank];
                    assert!(!visited[residue as usize]);
                    visited[residue as usize] = true;
                    // An arbitrary valid state keeps this single lookup above the refill threshold.
                    // Full-stream terminal states are checked by the GPU serialization tests.
                    let state = ((1u32 << 28) | residue).to_le_bytes();
                    let mut input = jxl_bitstream::Bitstream::new(&state);
                    let mut decoder = decoder.clone();
                    decoder.begin(&mut input).unwrap();
                    assert_eq!(decoder.read_varint(&mut input, 0).unwrap(), symbol as u32);
                    assert_eq!(input.num_read_bits(), 32);
                }
            }
            assert!(visited.into_iter().all(|value| value));
        }
    }
}
