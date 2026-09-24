use super::super::super::types::{ModularArtifactHeader, ModularEvent};
use super::super::tests::{gpu_fragment, length, raw};
use super::*;

#[test]
fn symbol_plans_cover_every_full_domain_boundary_and_write_independent_headers() {
    let plans = CodingPlan::candidates();
    assert_eq!(plans.len(), 94);
    assert_eq!(
        plans
            .iter()
            .map(|plan| plan.alphabet.log_size())
            .collect::<std::collections::BTreeSet<_>>(),
        [6, 7, 8].into()
    );
    for plan in plans {
        let mut writer = BitWriter::new();
        plan.write_lz77(&mut writer).unwrap();
        let header_bits = writer.bit_len();
        writer.align_to_byte().unwrap();
        let bytes = writer.into_bytes();
        let mut bits = jxl_bitstream::Bitstream::new(&bytes);
        assert!(bits.read_bool().unwrap());
        assert_eq!(
            bits.read_u32(224, 512, 4096, 8 + jxl_bitstream::U(15))
                .unwrap() as usize,
            plan.min_symbol
        );
        assert_eq!(
            bits.read_u32(3, 4, 5 + jxl_bitstream::U(2), 9 + jxl_bitstream::U(8))
                .unwrap(),
            7
        );
        assert_eq!(bits.read_bits(4).unwrap(), plan.length.packed());
        let split = plan.length.packed();
        let width = (32 - split.leading_zeros()) as usize;
        assert_eq!(bits.read_bits(width).unwrap(), 0);
        assert_eq!(bits.read_bits(width).unwrap(), 0);
        assert_eq!(bits.num_read_bits(), header_bits);
        assert!(plan.min_symbol + plan.length.symbols() <= plan.alphabet.symbols());
        for &config in hybrid::HybridConfig::candidates() {
            assert_eq!(plan.supports(config), config.symbols(32) <= plan.min_symbol);
        }
    }
    let length = length::LengthCoding::canonical();
    for invalid in [0, 7, 225, usize::MAX] {
        assert!(CodingPlan::new(invalid, length).is_err());
    }
    assert!(AnsAlphabet::for_symbols(0).is_err());
    assert!(AnsAlphabet::for_symbols(257).is_err());
    for symbols in [1, 32, 33, 64, 65, 128, 129, 256] {
        let alphabet = AnsAlphabet::for_symbols(symbols).unwrap();
        assert!(alphabet.symbols() >= symbols);
        if alphabet.symbols() < ALPHABET {
            let mut invalid = [0; ALPHABET];
            invalid[alphabet.symbols()] = 1;
            assert!(AnsHistogram::from_counts(&invalid, alphabet).is_err());
        }
    }
}

#[test]
fn omitted_thresholds_cannot_beat_their_nearest_configuration_boundary() {
    // Exhaust every legal threshold for each full-u32 configuration and length split.
    // Moving length symbols upward preserves populations and inserts only empty bins.
    // Check sparse, two-symbol and general histograms, including the 224 wire exception.
    for &config in hybrid::HybridConfig::candidates() {
        for length in length::LengthCoding::candidates() {
            let minimum = config.symbols(32);
            let Ok(base) = CodingPlan::new(minimum, length) else {
                continue;
            };
            for shape in 0..3 {
                let cost = |plan: CodingPlan| {
                    let mut counts = [0; ALPHABET];
                    counts[minimum - 1] = 1000;
                    if shape != 0 {
                        counts[0] = 500;
                    }
                    for token in 0..length.symbols() {
                        if shape == 2 || token == length.symbols() - 1 {
                            counts[plan.min_symbol + token] = 1 + (token % 7) as u64;
                        }
                    }
                    let histogram = AnsHistogram::from_counts(&counts, plan.alphabet).unwrap();
                    let mut header = BitWriter::new();
                    plan.write_lz77(&mut header).unwrap();
                    config.write(&mut header, plan.alphabet).unwrap();
                    histogram.write_histogram(&mut header).unwrap();
                    histogram.estimated_data_bits(&counts).unwrap()
                        + ((header.bit_len() as u128) << 20)
                };
                let base_cost = cost(base);
                for threshold in minimum..=ALPHABET - length.symbols() {
                    if threshold == 224 {
                        continue;
                    }
                    let plan = CodingPlan::new(threshold, length).unwrap();
                    assert!(cost(plan) >= base_cost, "{config:?} {plan:?} shape={shape}");
                }
            }
        }
    }
}

#[test]
fn gpu_symbol_domains_reject_raw_collisions_and_length_overflow_without_a_fragment() {
    let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let pipeline = pipeline(&context);
    let coding = CodingPlan::new(33, length::LengthCoding::candidates()[0]).unwrap();
    let counts = std::array::from_fn(|symbol| u64::from(symbol < coding.alphabet.symbols()));
    let mut codebook = AnsCodebook {
        tables: vec![hybrid::HybridCode {
            config: Default::default(),
            code: AnsHistogram::from_counts(&counts, coding.alphabet)
                .unwrap()
                .compile()
                .unwrap(),
        }],
        context_map: [0; 5],
        mode: LosslessModularLz77::ZeroRuns,
        coding,
    };
    // Inject inconsistent GPU metadata after planning: one raw token collides with
    // LZ77, or a valid length exceeds the declared alias alphabet. Neither may publish.
    for (threshold, event) in [(32, raw(u32::MAX, 0)), (50, length((1 << 20) - 1, 1))] {
        codebook.coding.min_symbol = threshold;
        let channels = [vec![event]];
        let artifacts = [ValidatedModularArtifact {
            header: ModularArtifactHeader {
                event_count: 1,
                raw_counts: [0; RAW_SYMBOLS],
                lz77_counts: [0; LZ77_SYMBOLS],
                distance_counts: [0; RAW_SYMBOLS],
            },
            events: &channels[0],
            palette_counts: None,
        }];
        let (plan, words) = gpu_fragment(&context, &pipeline, &codebook, &channels, None);
        assert_eq!(words[plan.byte_offset as usize / 4], 2);
        assert!(validate_encoded_group(plan, bytemuck::cast_slice(&words), &artifacts).is_err());
    }
    codebook.coding = CodingPlan::canonical();
    assert!(codebook.write_histograms(&mut BitWriter::new()).is_err());
}

struct Rates {
    coding: CodingPlan,
    configurations: Vec<hybrid::HybridConfig>,
    unions: Vec<Vec<f64>>,
    header_bits: usize,
}

impl Rates {
    fn for_map(&self, map: [u8; 5], tables: Option<&[hybrid::HybridCode]>, copies: u64) -> f64 {
        let clusters = *map.iter().max().unwrap() as usize + 1;
        let width = if clusters == 1 {
            0
        } else {
            (clusters - 1).ilog2() as usize + 1
        };
        let mut cost = (self.header_bits + 3 + 5 * width) as f64 * copies as f64;
        for cluster in 0..clusters {
            let subset: usize = (0..5)
                .filter(|&context| map[context] as usize == cluster)
                .map(|context| 1 << context)
                .sum();
            cost += if let Some(tables) = tables {
                let index = self
                    .configurations
                    .iter()
                    .position(|&config| config == tables[cluster].config)
                    .unwrap();
                self.unions[subset - 1][index]
            } else {
                self.unions[subset - 1]
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min)
            };
        }
        cost
    }
}

#[test]
fn global_alphabet_threshold_length_and_hybrid_choices_match_independent_joint_rates() {
    let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let pipelines = AnsPipelines::new(&context);
    for mode in [LosslessModularLz77::ZeroRuns, LosslessModularLz77::Greedy] {
        for seed in 0..2 {
            let channels: Vec<Vec<ModularEvent>> = (0..6)
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
                let mut max_samples = 0;
                for coding in CodingPlan::candidates() {
                    let profiles = histograms.candidates(mode, coding).unwrap();
                    let mut header = BitWriter::new();
                    coding.write_lz77(&mut header).unwrap();
                    let mut entry = Rates {
                        coding,
                        configurations: profiles.iter().map(|profile| profile.config).collect(),
                        unions: vec![vec![0.0; profiles.len()]; 31],
                        header_bits: header.bit_len(),
                    };
                    for subset in 1..32 {
                        for (index, profile) in profiles.iter().enumerate() {
                            let merged = std::array::from_fn(|symbol| {
                                (0..5)
                                    .filter(|&context| subset & (1 << context) != 0)
                                    .map(|context| profile.counts[context][symbol])
                                    .sum()
                            });
                            max_samples = max_samples.max(merged.iter().sum::<u64>());
                            let histogram =
                                AnsHistogram::from_counts(&merged, coding.alphabet).unwrap();
                            let mut header = BitWriter::new();
                            histogram.write_histogram(&mut header).unwrap();
                            profile.config.write(&mut header, coding.alphabet).unwrap();
                            let rate = &mut entry.unions[subset - 1][index];
                            *rate = header.bit_len() as f64 * copies as f64;
                            for context in 0..5 {
                                if subset & (1 << context) != 0 {
                                    *rate += profile.extra_bits[context] as f64;
                                }
                            }
                            for (&count, &frequency) in merged.iter().zip(histogram.frequencies()) {
                                if count != 0 {
                                    *rate += count as f64 * (12.0 - f64::from(frequency).log2());
                                }
                            }
                        }
                    }
                    rates.push(entry);
                }
                let maps = clustering::tests::maps();
                let minimum = rates
                    .iter()
                    .flat_map(|rate| maps.iter().map(move |&map| rate.for_map(map, None, copies)))
                    .fold(f64::INFINITY, f64::min);
                let selected = AnsCodebook::new(mode, &histograms, copies).unwrap();
                let rate = rates
                    .iter()
                    .find(|rate| rate.coding == selected.coding)
                    .unwrap();
                let actual = rate.for_map(selected.context_map, Some(&selected.tables), copies);
                assert!(
                    actual - minimum <= max_samples as f64 * 2f64.powi(-19),
                    "mode={mode:?} seed={seed} copies={copies}: {actual} > {minimum}"
                );
                let repeated = AnsCodebook::new(mode, &histograms, copies).unwrap();
                assert_eq!(selected.coding, repeated.coding);
                assert_eq!(selected.context_map, repeated.context_map);
                assert!(
                    selected
                        .tables
                        .iter()
                        .zip(repeated.tables)
                        .all(|(a, b)| a.config == b.config
                            && a.code.gpu_words() == b.code.gpu_words())
                );
            }
        }
    }
}
