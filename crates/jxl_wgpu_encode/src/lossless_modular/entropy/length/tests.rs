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
                let code = EntropyCode::Ans(Box::new(AnsCodebook {
                    tables: vec![hybrid::HybridCode {
                        config: raw_code,
                        code: AnsCode::from_counts(&[1; ALPHABET]).unwrap(),
                    }],
                    context_map: [0; clustering::CONTEXTS],
                    mode,
                    length: length_code,
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
        length: LengthCoding::canonical(),
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

#[test]
fn global_length_choice_matches_independent_joint_rates_including_repeated_headers() {
    let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let pipelines = AnsPipelines::new(&context);
    for mode in [LosslessModularLz77::ZeroRuns, LosslessModularLz77::Greedy] {
        for seed in 0..2 {
            let channels: Vec<Vec<_>> = (0..6)
                .map(|channel| {
                    (0..96)
                        .flat_map(|index| {
                            let mut events = vec![raw((1 << (channel + 3)) + (index % 8), 0)];
                            let copied = if seed == 0 {
                                7 + ((channel + index) % 16)
                            } else {
                                7 + (1 << (channel + 3))
                            };
                            events.push(length(
                                copied,
                                if mode == LosslessModularLz77::ZeroRuns {
                                    1
                                } else {
                                    2
                                },
                            ));
                            if mode == LosslessModularLz77::Greedy {
                                events.push(raw(120, 3));
                            }
                            events
                        })
                        .collect()
                })
                .collect();
            let (mut histograms, bytes, offset) =
                hybrid::tests::gpu_profiles(&context, &pipelines.profile, mode, &channels);
            histograms
                .read_profiles(mode, offset, channels.len(), bytemuck::cast_slice(&bytes))
                .unwrap();
            for copies in [1, 1024] {
                let mut rates = Vec::new();
                let mut length_headers = Vec::new();
                let mut max_samples = 0u64;
                for length_code in LengthCoding::candidates() {
                    let mut header = BitWriter::new();
                    length_code.write(&mut header).unwrap();
                    length_headers.push(header.bit_len());
                    let lengths = length_code.histograms(&histograms.lz77).unwrap();
                    let profiles = histograms.candidates(mode, &lengths).unwrap();
                    let mut unions = [[0.0; hybrid::PROFILES]; 31];
                    for subset in 1..32 {
                        for (profile_index, profile) in profiles.iter().enumerate() {
                            let merged = std::array::from_fn(|symbol| {
                                (0..clustering::CONTEXTS)
                                    .filter(|&context| subset & (1 << context) != 0)
                                    .map(|context| profile.counts[context][symbol])
                                    .sum()
                            });
                            max_samples = max_samples.max(merged.iter().sum());
                            let table = AnsCode::from_counts(&merged).unwrap();
                            let mut header = BitWriter::new();
                            table.write_histogram(&mut header).unwrap();
                            profile.config.write(&mut header).unwrap();
                            let cost = &mut unions[subset - 1][profile_index];
                            *cost = header.bit_len() as f64 * copies as f64;
                            for context in 0..clustering::CONTEXTS {
                                if subset & (1 << context) != 0 {
                                    *cost += profile.extra_bits[context] as f64;
                                }
                            }
                            for (&count, &frequency) in merged.iter().zip(table.gpu_words()) {
                                if count != 0 {
                                    *cost += count as f64 * (12.0 - f64::from(frequency).log2());
                                }
                            }
                        }
                    }
                    rates.push(unions);
                }
                let map_cost =
                    |length_index: usize, map: [u8; 5], tables: Option<&[hybrid::HybridCode]>| {
                        let clusters = *map.iter().max().unwrap() as usize + 1;
                        let width = if clusters == 1 {
                            0
                        } else {
                            (clusters - 1).ilog2() as usize + 1
                        };
                        let mut cost =
                            (length_headers[length_index] + 3 + 5 * width) as f64 * copies as f64;
                        for cluster in 0..clusters {
                            let subset: usize = (0..5)
                                .filter(|&context| map[context] as usize == cluster)
                                .map(|context| 1 << context)
                                .sum();
                            cost += if let Some(tables) = tables {
                                let index = hybrid::HybridConfig::candidates()
                                    .iter()
                                    .position(|&config| config == tables[cluster].config)
                                    .unwrap();
                                rates[length_index][subset - 1][index]
                            } else {
                                rates[length_index][subset - 1]
                                    .into_iter()
                                    .fold(f64::INFINITY, f64::min)
                            };
                        }
                        cost
                    };
                let minimum = (0..5)
                    .flat_map(|length_index| {
                        clustering::tests::maps()
                            .into_iter()
                            .map(move |map| (length_index, map))
                    })
                    .map(|(length_index, map)| map_cost(length_index, map, None))
                    .fold(f64::INFINITY, f64::min);
                let selected = AnsCodebook::new(mode, &histograms, copies).unwrap();
                let length_index = LengthCoding::candidates()
                    .iter()
                    .position(|&code| code == selected.length)
                    .unwrap();
                let actual = map_cost(length_index, selected.context_map, Some(&selected.tables));
                assert!(
                    actual - minimum <= max_samples as f64 * 2f64.powi(-19),
                    "mode={mode:?} seed={seed} copies={copies} actual={actual} minimum={minimum}"
                );
            }
        }
    }
}
