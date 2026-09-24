use super::super::super::types::ModularArtifactHeader;
use super::*;

fn maps() -> Vec<ContextMap> {
    // Independent Cartesian enumeration; canonical labels remove equivalent maps.
    (0..CONTEXTS.pow(CONTEXTS as u32))
        .map(|mut value| {
            std::array::from_fn(|_| {
                let digit = (value % CONTEXTS) as u8;
                value /= CONTEXTS;
                digit
            })
        })
        .filter(|map| {
            map[0] == 0 && (1..CONTEXTS).all(|i| map[i] <= *map[..i].iter().max().unwrap() + 1)
        })
        .collect()
}

#[test]
fn every_context_partition_has_an_independently_decodable_map() {
    let maps = maps();
    assert_eq!(maps.len(), 52);
    for map in maps {
        let mut writer = BitWriter::new();
        write_context_map(&mut writer, &map).unwrap();
        let length = writer.bit_len();
        writer.align_to_byte().unwrap();
        let bytes = writer.into_bytes();
        let mut bits = jxl_bitstream::Bitstream::new(&bytes);
        let (count, decoded) = jxl_coding::read_clusters(&mut bits, CONTEXTS as u32).unwrap();
        assert_eq!(count, u32::from(*map.iter().max().unwrap()) + 1);
        assert_eq!(decoded, map.into_iter().rev().collect::<Vec<_>>());
        assert_eq!(bits.num_read_bits(), length);
    }
}

#[test]
fn clustering_preserves_distinct_populations_and_charges_repeated_headers() {
    let empty = [[0; ALPHABET]; CONTEXTS];
    assert_eq!(cluster(&empty, 1).unwrap().1, [0; CONTEXTS]);
    let identical =
        std::array::from_fn(|_| std::array::from_fn(|symbol| u64::from(symbol == 1) * 100_000));
    assert_eq!(cluster(&identical, 1).unwrap().1, [0; CONTEXTS]);
    let distinct = std::array::from_fn(|context| {
        std::array::from_fn(|symbol| u64::from(symbol == context * 61) * 100_000)
    });
    assert_eq!(cluster(&distinct, 1).unwrap().1, [0, 1, 2, 3, 4]);
    let pairs = std::array::from_fn(|context| {
        std::array::from_fn(|symbol| u64::from(symbol == (context % 2) * 255) * 100_000)
    });
    assert_eq!(cluster(&pairs, 1).unwrap().1, [0, 1, 0, 1, 0]);
    let mut tied = pairs;
    tied[2] = [0; ALPHABET];
    assert_eq!(cluster(&tied, 1).unwrap().1, [0, 1, 0, 1, 0]);
    let similar = std::array::from_fn(|context| {
        std::array::from_fn(|symbol| {
            if symbol > 1 {
                0
            } else if symbol == context % 2 {
                999
            } else {
                1
            }
        })
    });
    assert_eq!(cluster(&similar, 1).unwrap().0.len(), 2);
    assert_eq!(cluster(&similar, 10_000).unwrap().0.len(), 1);
    assert!(cluster(&empty, 0).is_err());
    let mut overflow = empty;
    overflow[0][0] = u64::MAX;
    overflow[1][0] = 1;
    assert!(matches!(
        cluster(&overflow, 1),
        Err(EncodeError::Backend(BackendError::InvalidArtifact(_)))
    ));
}

fn reference_cost(counts: &[[u64; ALPHABET]; CONTEXTS], map: ContextMap, copies: u64) -> f64 {
    let clusters = *map.iter().max().unwrap() as usize + 1;
    let width = if clusters == 1 {
        0
    } else {
        (clusters - 1).ilog2() + 1
    };
    let mut cost = (3 + CONTEXTS as u64 * u64::from(width)) as f64 * copies as f64;
    for cluster in 0..clusters {
        let merged = std::array::from_fn(|symbol| {
            counts
                .iter()
                .enumerate()
                .filter(|(context, _)| map[*context] as usize == cluster)
                .map(|(_, counts)| counts[symbol])
                .sum()
        });
        let code = AnsCode::from_counts(&merged).unwrap();
        let mut header = BitWriter::new();
        code.write_histogram(&mut header).unwrap();
        cost += (header.bit_len() + 4) as f64 * copies as f64;
        for (&count, &frequency) in merged.iter().zip(code.gpu_words()) {
            if count != 0 {
                cost += count as f64 * (12.0 - f64::from(frequency).log2());
            }
        }
    }
    cost
}

#[test]
fn exhaustive_selection_matches_independent_f64_cost_and_is_deterministic() {
    for seed in 0..24 {
        let counts = std::array::from_fn(|context| {
            std::array::from_fn(|symbol| {
                if (symbol * 13 + context * 7 + seed) % (seed + 3) < 2 {
                    1u64 << ((symbol * 3 + context + seed) % 20)
                } else {
                    0
                }
            })
        });
        for copies in [1, 257] {
            let (tables, selected) = cluster(&counts, copies).unwrap();
            let (repeated, second) = cluster(&counts, copies).unwrap();
            assert_eq!(selected, second);
            assert!(
                tables
                    .iter()
                    .zip(&repeated)
                    .all(|(a, b)| a.config == b.config && a.code.gpu_words() == b.code.gpu_words())
            );
            let best = maps()
                .into_iter()
                .map(|map| reference_cost(&counts, map, copies))
                .fold(f64::INFINITY, f64::min);
            let selected = reference_cost(&counts, selected, copies);
            // Q20 rounding is bounded per symbol, independently of stream length.
            let allowance = counts.iter().flatten().sum::<u64>() as f64 * 2f64.powi(-19);
            assert!(
                selected - best <= allowance,
                "seed={seed} copies={copies}: {selected} > {best} + {allowance}"
            );
        }
    }
}

#[test]
fn every_partition_and_reversed_cluster_label_decodes_actual_gpu_symbols() {
    use super::super::tests::{gpu_fragment, independent_decode, length, raw};
    let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let pipeline = pipeline(&context);
    let counts: [[u64; ALPHABET]; CONTEXTS] = std::array::from_fn(|context| {
        std::array::from_fn(|symbol| 1 + ((symbol * 17 + context * 53) % 71) as u64)
    });
    for mode in [LosslessModularLz77::ZeroRuns, LosslessModularLz77::Greedy] {
        let mut expected = Vec::new();
        let channels: Vec<_> = (0..6)
            .map(|channel| {
                let mut values = vec![0, 1, (1 << (channel + 8)) - 1, u32::MAX];
                let mut events: Vec<_> = values.iter().map(|&value| raw(value, 0)).collect();
                if channel == 1 {
                    events.clear();
                    values.clear();
                } else if mode == LosslessModularLz77::ZeroRuns {
                    events.push(length(23, 1));
                    values.extend([0; 24]);
                } else {
                    events.push(length(23, 2));
                    events.push(raw(120, 3));
                    values.extend([u32::MAX; 23]);
                }
                expected.push(values);
                events
            })
            .collect();
        let artifacts: Vec<_> = channels
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
            .collect();
        for map in maps() {
            let count = *map.iter().max().unwrap() as usize + 1;
            let tables: Vec<_> = (0..count)
                .map(|cluster| {
                    let merged = std::array::from_fn(|symbol| {
                        counts
                            .iter()
                            .enumerate()
                            .filter(|(context, _)| map[*context] as usize == cluster)
                            .map(|(_, counts)| counts[symbol])
                            .sum()
                    });
                    hybrid::HybridCode {
                        config: Default::default(),
                        code: AnsCode::from_counts(&merged).unwrap(),
                    }
                })
                .collect();
            // Reverse labels as well, so the distance table is not assumed to be table zero.
            for reverse in [false, true] {
                let code = EntropyCode::Ans(Box::new(AnsCodebook {
                    tables: if reverse {
                        tables.iter().rev().cloned().collect()
                    } else {
                        tables.clone()
                    },
                    context_map: map.map(|cluster| {
                        if reverse {
                            count as u8 - 1 - cluster
                        } else {
                            cluster
                        }
                    }),
                    mode,
                }));
                let (plan, words) =
                    gpu_fragment(&context, &pipeline, code.ans().unwrap(), &channels, None);
                let fragment =
                    validate_encoded_group(plan, bytemuck::cast_slice(&words), &artifacts).unwrap();
                independent_decode(&code, fragment, &expected);
            }
        }
    }
}

fn cluster(
    counts: &[[u64; ALPHABET]; CONTEXTS],
    copies: u64,
) -> Result<(Vec<hybrid::HybridCode>, ContextMap), EncodeError> {
    super::cluster(
        &[hybrid::HybridCounts {
            config: Default::default(),
            counts: *counts,
            extra_bits: [0; CONTEXTS],
        }],
        copies,
    )
}

#[test]
fn joint_hybrid_partition_search_matches_exhaustive_f64_costs() {
    for seed in 0..4 {
        let profiles: Vec<_> = hybrid::HybridConfig::candidates()
            .iter()
            .enumerate()
            .map(|(index, &config)| hybrid::HybridCounts {
                config,
                counts: std::array::from_fn(|context| {
                    std::array::from_fn(|symbol| {
                        if symbol % (index + 2) == context % (index + 2) {
                            1 << ((symbol + context + seed) % 18)
                        } else {
                            0
                        }
                    })
                }),
                extra_bits: std::array::from_fn(|context| {
                    (index * 19 + context * 7 + seed) as u128 * 113
                }),
            })
            .collect();
        for copies in [1, 1000] {
            // Independent F64 rates for all 31 unions and 37 hybrid choices;
            // Cartesian partition enumeration is independent of the production recursion.
            let mut rates = [[0.0; hybrid::PROFILES]; 31];
            for subset in 1..32 {
                for (profile_index, profile) in profiles.iter().enumerate() {
                    let merged = std::array::from_fn(|symbol| {
                        (0..CONTEXTS)
                            .filter(|&context| subset & (1 << context) != 0)
                            .map(|context| profile.counts[context][symbol])
                            .sum()
                    });
                    let code = AnsCode::from_counts(&merged).unwrap();
                    let mut header = BitWriter::new();
                    code.write_histogram(&mut header).unwrap();
                    profile.config.write(&mut header).unwrap();
                    let rate = &mut rates[subset - 1][profile_index];
                    *rate = header.bit_len() as f64 * copies as f64;
                    for context in 0..CONTEXTS {
                        if subset & (1 << context) != 0 {
                            *rate += profile.extra_bits[context] as f64;
                        }
                    }
                    for (&count, &frequency) in merged.iter().zip(code.gpu_words()) {
                        if count != 0 {
                            *rate += count as f64 * (12.0 - f64::from(frequency).log2());
                        }
                    }
                }
            }
            let map_rate = |map: ContextMap, configurations: Option<&[hybrid::HybridCode]>| {
                let clusters = *map.iter().max().unwrap() as usize + 1;
                let width = if clusters == 1 {
                    0
                } else {
                    (clusters - 1).ilog2() + 1
                };
                let mut cost = (3 + CONTEXTS as u64 * u64::from(width)) as f64 * copies as f64;
                for cluster in 0..clusters {
                    let subset: usize = (0..CONTEXTS)
                        .filter(|&context| map[context] as usize == cluster)
                        .map(|context| 1 << context)
                        .sum();
                    cost += if let Some(tables) = configurations {
                        let index = profiles
                            .iter()
                            .position(|profile| profile.config == tables[cluster].config)
                            .unwrap();
                        rates[subset - 1][index]
                    } else {
                        rates[subset - 1].into_iter().fold(f64::INFINITY, f64::min)
                    };
                }
                cost
            };
            let (tables, map) = super::cluster(&profiles, copies).unwrap();
            let minimum = maps()
                .into_iter()
                .map(|map| map_rate(map, None))
                .fold(f64::INFINITY, f64::min);
            let allowance = profiles
                .iter()
                .map(|profile| profile.counts.iter().flatten().sum::<u64>())
                .max()
                .unwrap() as f64
                * 2f64.powi(-19);
            assert!(map_rate(map, Some(&tables)) - minimum <= allowance);
            let (repeated, next_map) = super::cluster(&profiles, copies).unwrap();
            assert_eq!(map, next_map);
            for (first, second) in tables.iter().zip(repeated) {
                assert_eq!(first.config, second.config);
                assert_eq!(first.code.gpu_words(), second.code.gpu_words());
            }
        }
    }
    assert!(super::cluster(&[], 1).is_err());
}
