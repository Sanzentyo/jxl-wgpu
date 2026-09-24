use super::super::super::types::{ModularArtifactHeader, ModularEvent};
use super::super::tests::{dispatch_gpu, gpu_fragment, independent_decode, length, raw};
use super::*;

fn gpu_profiles(
    context: &WgpuContext,
    pipeline: &wgpu::ComputePipeline,
    mode: LosslessModularLz77,
    channels: &[Vec<ModularEvent>],
) -> (FrameHistograms, Vec<u32>, u64) {
    let mut histograms = FrameHistograms::default();
    let profiles_start = 8 + 4 * channels.len();
    let mut metadata = vec![0u32; profiles_start];
    let mut words = Vec::new();
    for (channel, events) in channels.iter().enumerate() {
        let context = channel.min(3);
        let base = words.len();
        words.resize(base + 100, 0);
        words[base] = events.len() as u32;
        words.extend_from_slice(bytemuck::cast_slice(events));
        metadata[8 + channel * 4..12 + channel * 4].copy_from_slice(&[
            base as u32 + 100,
            base as u32,
            events.len() as u32,
            context as u32 + 1,
        ]);
        for event in events {
            match event.kind {
                0 => histograms.raw[context][event.token as usize] += 1,
                1 => {
                    histograms.raw[context][0] += 1;
                    histograms.lz77[context][event.token as usize] += 1;
                }
                2 => histograms.lz77[context][event.token as usize] += 1,
                3 => histograms.distance[event.token as usize] += 1,
                _ => unreachable!(),
            }
        }
    }
    let offset = words.len();
    metadata[..8].copy_from_slice(&[
        0,
        0,
        u32::from(mode == LosslessModularLz77::Greedy),
        0,
        8,
        offset as u32,
        profiles_start as u32,
        channels.len() as u32,
    ]);
    metadata.extend(
        HybridConfig::candidates()
            .iter()
            .map(|config| config.packed()),
    );
    words.resize(offset + PROFILE_BYTES as usize / 4 + 1, 0);
    *words.last_mut().unwrap() = 0xfeed_cafe;
    let result = dispatch_gpu(
        context,
        pipeline,
        &words,
        &metadata,
        [channels.len() as u32, PROFILES as u32, 1],
    );
    (histograms, result, offset as u64 * 4)
}

#[test]
fn hybrid_profiling_shader_is_portable_and_configurations_cover_the_admitted_alphabet() {
    let module =
        naga::front::wgsl::parse_str(&shader_source(include_str!("../profile.wgsl"))).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let candidates = HybridConfig::candidates();
    assert_eq!(candidates.len(), 37);
    for split in 0..=8 {
        for msb in 0..=split {
            for lsb in 0..=split - msb {
                let config = HybridConfig { split, msb, lsb };
                assert_eq!(candidates.contains(&config), config.max_token() < 224);
            }
        }
    }
    assert_eq!(PROFILE_BYTES, 165_768);
}

#[test]
fn all_hybrid_profiles_decode_gpu_words_and_reject_unvalidated_histograms() {
    let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let pipelines = AnsPipelines::new(&context);
    for mode in [LosslessModularLz77::ZeroRuns, LosslessModularLz77::Greedy] {
        // Dense small values exercise every split, mantissa and low-bit boundary;
        // wide values cover all 32 widths, including signed residual extrema.
        let mut values: Vec<u32> = (0..512).collect();
        for bit in 9..32 {
            for delta in [0, 1, 3, 7, 31, 255] {
                values.push((1 << bit) - delta);
                values.push((1 << bit) + delta);
                values.push(u32::MAX - delta);
            }
        }
        let mut expected = vec![values.clone(); 6];
        let channels: Vec<_> = (0..6)
            .map(|channel| {
                if channel == 1 {
                    expected[channel].clear();
                    return Vec::new();
                }
                let mut events: Vec<_> = values.iter().map(|&value| raw(value, 0)).collect();
                if mode == LosslessModularLz77::ZeroRuns {
                    events.push(length(23, 1));
                    expected[channel].extend([0; 24]);
                } else {
                    events.push(length(23, 2));
                    events.push(raw(120, 3));
                    expected[channel].extend([*values.last().unwrap(); 23]);
                    events.push(length(17, 2));
                    events.push(raw(values.len() as u32 + 23 + 119, 3));
                    expected[channel].extend_from_slice(&values[..17]);
                }
                events
            })
            .collect();
        let (mut histograms, words, offset) =
            gpu_profiles(&context, &pipelines.profile, mode, &channels);
        let bytes = bytemuck::cast_slice(&words);
        histograms
            .read_profiles(mode, offset, channels.len(), bytes)
            .unwrap();
        let candidates = histograms.candidates(mode).unwrap();
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
        for candidate in &candidates {
            let tables = candidate
                .counts
                .iter()
                .map(|counts| HybridCode {
                    config: candidate.config,
                    code: AnsCode::from_counts(counts).unwrap(),
                })
                .collect();
            let code = EntropyCode::Ans(Box::new(AnsCodebook {
                tables,
                context_map: [0, 1, 2, 3, 4],
                mode,
            }));
            let (plan, result) = gpu_fragment(
                &context,
                &pipelines.encode,
                code.ans().unwrap(),
                &channels,
                None,
            );
            let fragment =
                validate_encoded_group(plan, bytemuck::cast_slice(&result), &artifacts).unwrap();
            independent_decode(&code, fragment, &expected);
        }
        let base = offset as usize / 4;
        for (index, value) in [
            (base, 0),
            (base, words[base] - 1),
            (base + 1, 1),
            (base + 2 + 223, 1),
            (base + 2 + 224, words[base + 2 + 224] + 1),
        ] {
            let mut corrupt = words.clone();
            corrupt[index] = value;
            assert!(
                histograms
                    .read_profiles(mode, offset, channels.len(), bytemuck::cast_slice(&corrupt))
                    .is_err()
            );
        }
        assert!(
            histograms
                .read_profiles(mode, offset, channels.len(), &bytes[..bytes.len() - 5])
                .is_err()
        );
        assert!(
            histograms
                .read_profiles(mode, u64::MAX, channels.len(), bytes)
                .is_err()
        );
    }
}

#[test]
fn adaptive_hybrid_selection_uses_mantissa_and_low_bits_with_exact_extra_costs() {
    let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let pipelines = AnsPipelines::new(&context);
    for low_bits in [false, true] {
        let values: Vec<u32> = (0..4096)
            .map(|index| {
                (1 << 20)
                    + if low_bits {
                        (index % 256) * 4 + ((index / 256 % 4) << 18)
                    } else {
                        (index % 4) + (((index / 4 % 2) * 3) << 18)
                    }
            })
            .collect();
        let channels = [values.iter().map(|&value| raw(value, 0)).collect()];
        let mode = LosslessModularLz77::ZeroRuns;
        let (mut histograms, result, offset) =
            gpu_profiles(&context, &pipelines.profile, mode, &channels);
        histograms
            .read_profiles(mode, offset, 1, bytemuck::cast_slice(&result))
            .unwrap();
        for profile in histograms.candidates(mode).unwrap() {
            let expected: u128 = values
                .iter()
                .map(|&value| {
                    if value < 1 << profile.config.split {
                        0
                    } else {
                        u128::from(
                            value.ilog2() - u32::from(profile.config.msb + profile.config.lsb),
                        )
                    }
                })
                .sum();
            assert_eq!(profile.extra_bits[1], expected);
        }
        let codebook = AnsCodebook::new(mode, &histograms, 1).unwrap();
        let selected = codebook.tables[codebook.context_map[1] as usize].config;
        assert_eq!(
            if low_bits { selected.lsb } else { selected.msb },
            2,
            "{selected:?}"
        );
        let code = EntropyCode::Ans(Box::new(codebook));
        let (plan, result) = gpu_fragment(
            &context,
            &pipelines.encode,
            code.ans().unwrap(),
            &channels,
            None,
        );
        let artifacts = [ValidatedModularArtifact {
            header: ModularArtifactHeader {
                event_count: channels[0].len() as u32,
                raw_counts: [0; RAW_SYMBOLS],
                lz77_counts: [0; LZ77_SYMBOLS],
                distance_counts: [0; RAW_SYMBOLS],
            },
            events: &channels[0],
            palette_counts: None,
        }];
        independent_decode(
            &code,
            validate_encoded_group(plan, bytemuck::cast_slice(&result), &artifacts).unwrap(),
            &[values],
        );
    }
    let missing = FrameHistograms::default();
    assert!(AnsCodebook::new(LosslessModularLz77::ZeroRuns, &missing, 1).is_err());
    let mut overflow = FrameHistograms::default();
    overflow.raw[0][0] = u64::MAX;
    let mut next = FrameHistograms::default();
    next.raw[0][0] = 1;
    assert!(overflow.accumulate(next).is_err());
}
