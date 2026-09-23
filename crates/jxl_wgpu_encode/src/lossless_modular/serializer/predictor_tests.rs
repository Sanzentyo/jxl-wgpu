use super::*;

#[test]
fn independent_entropy_decoder_reads_every_predictor_and_weighted_header() {
    for predictor in LosslessModularPredictor::ALL {
        let codes = std::array::from_fn(|_| PrefixCode::fixed_unused_channel());
        let weighted =
            LosslessModularWeightedPredictor::new([31, 0, 1, 2, 3, 4, 5], [0, 15, 1, 2]).unwrap();
        let mut output = BitWriter::new();
        write_dc_global(
            &mut output,
            &codes,
            TransformHeader {
                rct: None,
                squeeze: LosslessModularSqueeze::None,
                channels: 4,
            },
            predictor,
            weighted,
            None,
        )
        .unwrap();
        let bit_len = output.bit_len();
        output.align_to_byte().unwrap();
        let bytes = output.into_bytes();
        let mut bits = jxl_bitstream::Bitstream::new(&bytes);
        assert!(bits.read_bool().unwrap()); // default LF dequantization
        assert!(bits.read_bool().unwrap()); // global MA tree
        let mut tree = jxl_coding::Decoder::parse(&mut bits, 6).unwrap();
        tree.begin(&mut bits).unwrap();
        for threshold in [2, 4, 0] {
            assert_eq!(tree.read_varint(&mut bits, 1).unwrap(), 1);
            assert_eq!(tree.read_varint(&mut bits, 0).unwrap(), threshold);
        }
        for _ in 0..4 {
            assert_eq!(tree.read_varint(&mut bits, 1).unwrap(), 0);
            assert_eq!(tree.read_varint(&mut bits, 2).unwrap(), predictor.value());
            for context in 3..6 {
                assert_eq!(tree.read_varint(&mut bits, context).unwrap(), 0);
            }
        }
        tree.finalize().unwrap();
        let entropy = jxl_coding::Decoder::parse(&mut bits, 4).unwrap();
        assert_eq!(entropy.cluster_map(), &[4, 3, 2, 1, 0]);
        assert!(bits.read_bool().unwrap()); // uses global tree
        assert!(!bits.read_bool().unwrap()); // explicit WP
        for expected in weighted.coefficients() {
            assert_eq!(bits.read_bits(5).unwrap(), u32::from(expected));
        }
        for expected in weighted.max_weights() {
            assert_eq!(bits.read_bits(4).unwrap(), u32::from(expected));
        }
        assert_eq!(bits.read_bits(2).unwrap(), 0); // no transforms
        assert_eq!(bits.num_read_bits(), bit_len);
    }
}

#[test]
fn weighted_parameters_reject_each_unrepresentable_field() {
    for index in 0..7 {
        let mut coefficients = [0; 7];
        coefficients[index] = 32;
        assert!(
            matches!(LosslessModularWeightedPredictor::new(coefficients, [0; 4]),
            Err(EncodeError::WeightedPredictorParameter { name: "coefficient", index: actual, value: 32, maximum: 31 }) if actual == index)
        );
    }
    for index in 0..4 {
        let mut weights = [0; 4];
        weights[index] = 16;
        assert!(
            matches!(LosslessModularWeightedPredictor::new([0; 7], weights),
            Err(EncodeError::WeightedPredictorParameter { name: "max_weight", index: actual, value: 16, maximum: 15 }) if actual == index)
        );
    }
}
