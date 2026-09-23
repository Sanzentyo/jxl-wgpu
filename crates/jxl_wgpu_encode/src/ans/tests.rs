use super::*;

#[test]
fn independent_decoder_checks_every_alias_residue_and_histogram_form() {
    let mut distributions = vec![[0; ALPHABET], [u64::MAX; ALPHABET]];
    for symbol in 0..ALPHABET {
        let mut counts = [0; ALPHABET];
        counts[symbol] = u64::MAX;
        distributions.push(counts);
    }
    for active in 2..=ALPHABET {
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
    sparse[255] = u64::MAX;
    distributions.push(sparse);
    for counts in distributions {
        let code = AnsCode::from_counts(&counts).unwrap();
        assert_eq!(
            code.gpu_words(),
            AnsCode::from_counts(&counts).unwrap().gpu_words()
        );
        assert_eq!(code.frequencies.iter().sum::<u32>(), TABLE_SIZE as u32);
        if counts.iter().any(|&count| count != 0) {
            for (count, frequency) in counts.iter().zip(code.frequencies) {
                assert_eq!(*count == 0, frequency == 0);
            }
        }
        let mut writer = BitWriter::new();
        writer.write_bits(0, 1).unwrap(); // no LZ77, implicit single context
        writer.write_bits(0, 1).unwrap(); // ANS
        writer.write_bits(3, 2).unwrap(); // log alphabet=8
        writer.write_bits(8, 4).unwrap(); // literal symbols, no hybrid extra bits
        code.write_histogram(&mut writer).unwrap();
        let header_bits = writer.bit_len();
        writer.align_to_byte().unwrap();
        let bytes = writer.into_bytes();
        let mut bits = jxl_bitstream::Bitstream::new(&bytes);
        let decoder = jxl_coding::Decoder::parse(&mut bits, 1).unwrap();
        assert_eq!(bits.num_read_bits(), header_bits);
        let mut visited = [false; TABLE_SIZE];
        for (symbol, &frequency) in code.frequencies.iter().enumerate() {
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
