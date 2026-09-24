use super::super::super::types::{ModularArtifactHeader, ModularEvent};
use super::super::tests::{gpu_fragment, independent_decode, length, raw};
use super::*;

#[test]
fn all_length_configurations_exactly_rebin_the_complete_twenty_bit_domain() {
    let configurations = LengthCoding::candidates();
    assert_eq!(
        configurations
            .iter()
            .map(|code| code.packed())
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4]
    );
    let mut canonical = [[0; LZ77_SYMBOLS]; 4];
    for value in 0..1u32 << 20 {
        let count = 1 + u64::from(value % 13);
        canonical[0][length(value + 7, 2).token as usize] += count;
    }
    for configuration in configurations {
        let split = configuration.packed();
        let mut expected = [0; LENGTH_SYMBOLS];
        let mut extra_bits = 0;
        for value in 0..1u32 << 20 {
            let width = 32 - value.leading_zeros();
            let (token, extra) = if width <= split {
                (value, 0)
            } else {
                ((1 << split) + width - split - 1, width - 1)
            };
            let count = 1 + u64::from(value % 13);
            expected[token as usize] += count;
            extra_bits += u128::from(count) * u128::from(extra);
        }
        let rebinned = configuration.histograms(&canonical).unwrap();
        assert_eq!(rebinned.counts[0], expected);
        assert_eq!(rebinned.extra_bits[0], extra_bits);
        assert_eq!(rebinned.counts[1..], [[0; LENGTH_SYMBOLS]; 3]);
    }
    let mut overflow = [[0; LZ77_SYMBOLS]; 4];
    overflow[0][2] = u64::MAX;
    overflow[0][3] = 1;
    assert!(LengthCoding::candidates()[0].histograms(&overflow).is_err());
    overflow[0][32] = 1;
    assert!(LengthCoding::canonical().histograms(&overflow).is_err());
}

fn artifacts(channels: &[Vec<ModularEvent>]) -> Vec<ValidatedModularArtifact<'_>> {
    channels
        .iter()
        .map(|events| ValidatedModularArtifact {
            header: ModularArtifactHeader {
                event_count: events.len() as u32,
                raw_counts: [0; RAW_SYMBOLS],
                lz77_counts: [0; LZ77_SYMBOLS],
                distance_counts: [0; RAW_SYMBOLS],
            },
            events,
            palette_counts: None,
        })
        .collect()
}

#[test]
fn every_length_and_residual_configuration_decodes_gpu_matches_and_rejects_bad_events() {
    let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let pipeline = pipeline(&context);
    for mode in [LosslessModularLz77::ZeroRuns, LosslessModularLz77::Greedy] {
        for length_code in LengthCoding::candidates() {
            for (raw_index, &raw_code) in hybrid::HybridConfig::candidates().iter().enumerate() {
                let Ok(coding) = coding::CodingPlan::new(raw_code.symbols(32), length_code) else {
                    // Three 228-symbol residual configurations need a shorter length alphabet.
                    assert_eq!((raw_code.symbols(32), length_code.symbols()), (228, 32));
                    continue;
                };
                let code = EntropyCode::Ans(Box::new(AnsCodebook {
                    tables: vec![hybrid::HybridCode {
                        config: raw_code,
                        code: AnsHistogram::from_counts(
                            &std::array::from_fn(|symbol| {
                                u64::from(symbol < coding.alphabet.symbols())
                            }),
                            coding.alphabet,
                        )
                        .unwrap()
                        .compile()
                        .unwrap(),
                    }],
                    context_map: [0; clustering::CONTEXTS],
                    mode,
                    coding,
                }));
                let mut expected = vec![vec![0, 1, 7, u32::MAX], vec![], vec![5]];
                let mut channels: Vec<Vec<_>> = expected
                    .iter()
                    .map(|values| values.iter().map(|&value| raw(value, 0)).collect())
                    .collect();
                let lengths: Vec<_> = (7..=24)
                    .chain([38, 39, 262, 263])
                    .chain((raw_index == 0).then_some((1 << 20) - 1))
                    .collect();
                for copied in lengths {
                    let channel = 2;
                    if mode == LosslessModularLz77::ZeroRuns {
                        channels[channel].push(length(copied, 1));
                        expected[channel].extend(std::iter::repeat_n(0, copied as usize + 1));
                    } else {
                        channels[channel].push(length(copied, 2));
                        channels[channel].push(raw(120, 3));
                        expected[channel].extend(std::iter::repeat_n(5, copied as usize));
                    }
                }
                let (plan, words) =
                    gpu_fragment(&context, &pipeline, code.ans().unwrap(), &channels, None);
                let artifacts = artifacts(&channels);
                let fragment =
                    validate_encoded_group(plan, bytemuck::cast_slice(&words), &artifacts).unwrap();
                independent_decode(&code, fragment, &expected);
            }
        }
    }
    let code = AnsCodebook {
        tables: vec![hybrid::HybridCode {
            config: Default::default(),
            code: AnsCode::from_counts(&[1; ALPHABET]).unwrap(),
        }],
        context_map: [0; clustering::CONTEXTS],
        mode: LosslessModularLz77::ZeroRuns,
        coding: coding::CodingPlan::canonical(),
    };
    for (token, extra_bit_count, extra_bits) in [
        (32, 20, 0),
        (16, 0, 0),
        (0, 1, 0),
        (15, 0, 1),
        (31, 19, 1 << 19),
    ] {
        let channels = [vec![ModularEvent {
            kind: 1,
            token,
            extra_bit_count,
            extra_bits,
        }]];
        let (plan, words) = gpu_fragment(&context, &pipeline, &code, &channels, None);
        assert_eq!(words[plan.byte_offset as usize / 4], 2);
        assert!(
            validate_encoded_group(plan, bytemuck::cast_slice(&words), &artifacts(&channels))
                .is_err()
        );
    }
}
