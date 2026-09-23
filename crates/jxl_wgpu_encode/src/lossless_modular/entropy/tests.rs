use super::super::types::{ModularArtifactHeader, ModularEvent};
use super::*;

#[test]
fn ans_shader_is_portable_and_capacity_checks_bit_addressing() {
    let module = naga::front::wgsl::parse_str(include_str!("../entropy.wgsl")).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(EntropyArtifactPlan::for_events(0, 0).unwrap().bytes(), 20);
    assert!(EntropyArtifactPlan::for_events(0, u64::MAX).is_err());
    assert!(EntropyArtifactPlan::for_events(0, u64::from(u32::MAX) / 80 + 1).is_err());
}

#[test]
fn unvalidated_ans_fragments_cannot_gain_packet_authority() {
    let plan = EntropyArtifactPlan::for_events(8, 2).unwrap();
    let header = ModularArtifactHeader {
        event_count: 2,
        raw_counts: [0; RAW_SYMBOLS],
        lz77_counts: [0; LZ77_SYMBOLS],
        distance_counts: [0; RAW_SYMBOLS],
    };
    let events = [
        ModularEvent {
            kind: 0,
            token: 0,
            extra_bit_count: 0,
            extra_bits: 0,
        },
        ModularEvent {
            kind: 1,
            token: 0,
            extra_bit_count: 0,
            extra_bits: 0,
        },
    ];
    let artifacts = [ValidatedModularArtifact {
        header,
        events: &events,
        palette_counts: None,
    }];
    let mut words = vec![0u32; 2 + plan.bytes() as usize / 4];
    words[2..6].copy_from_slice(&[1, 33, 4, 0]);
    let result = validate_encoded_group(plan, bytemuck::cast_slice(&words), &artifacts).unwrap();
    assert_eq!(result.bit_len, 33);
    for (index, invalid) in [
        (2, 0),
        (2, 2),
        (3, 0),
        (3, 31),
        (3, plan.capacity_words * 32 + 1),
        (4, 3),
        (4, 5),
        (5, 1),
        (7, 2),
    ] {
        let mut changed = words.clone();
        changed[index] = invalid;
        assert!(
            validate_encoded_group(plan, bytemuck::cast_slice(&changed), &artifacts).is_err(),
            "{index}/{invalid}"
        );
    }
    let bytes: &[u8] = bytemuck::cast_slice(&words);
    assert!(validate_encoded_group(plan, &bytes[..bytes.len() - 1], &artifacts).is_err());
    assert!(
        validate_encoded_group(
            EntropyArtifactPlan {
                byte_offset: u64::MAX,
                ..plan
            },
            bytes,
            &artifacts
        )
        .is_err()
    );
    assert!(
        EntropyCode::Ans(Box::new(
            AnsCodebook::new(
                LosslessModularLz77::ZeroRuns,
                &[[0; RAW_SYMBOLS]; 4],
                &[[0; LZ77_SYMBOLS]; 4],
                &[0; RAW_SYMBOLS]
            )
            .unwrap()
        ))
        .write_stream(&mut BitWriter::new(), &artifacts, None)
        .is_err()
    );
}

fn raw(value: u32, kind: u32) -> ModularEvent {
    let token = 32 - value.leading_zeros();
    let extra_bit_count = token.saturating_sub(1);
    ModularEvent {
        kind,
        token,
        extra_bit_count,
        extra_bits: if value == 0 {
            0
        } else {
            value - (1 << extra_bit_count)
        },
    }
}

fn length(copied: u32, kind: u32) -> ModularEvent {
    let value = copied - 7;
    let extra_bit_count = if value < 16 { 0 } else { value.ilog2() };
    ModularEvent {
        kind,
        token: if value < 16 {
            value
        } else {
            extra_bit_count + 12
        },
        extra_bit_count,
        extra_bits: if value < 16 {
            0
        } else {
            value - (1 << extra_bit_count)
        },
    }
}

fn gpu_fragment(
    context: &WgpuContext,
    pipeline: &wgpu::ComputePipeline,
    codebook: &AnsCodebook,
    channels: &[Vec<ModularEvent>],
    limit: Option<u32>,
) -> (EntropyArtifactPlan, Vec<u32>) {
    use wgpu::util::DeviceExt;
    let mut words = Vec::new();
    let table_start = 8 + 4 * channels.len();
    let mut metadata = vec![
        1,
        table_start as u32,
        u32::from(codebook.mode == LosslessModularLz77::Greedy),
        0,
    ];
    metadata.resize(table_start, 0);
    for (channel, events) in channels.iter().enumerate() {
        let base = words.len();
        words.resize(base + 100, 0);
        words[base] = events.len() as u32;
        words.extend_from_slice(bytemuck::cast_slice(events));
        metadata[8 + 4 * channel..12 + 4 * channel].copy_from_slice(&[
            base as u32 + 100,
            base as u32,
            events.len() as u32,
            channel.min(3) as u32 + 1,
        ]);
    }
    let output = words.len();
    let mut plan = EntropyArtifactPlan::for_events(
        4 * output as u64,
        channels.iter().map(|events| events.len() as u64).sum(),
    )
    .unwrap();
    if let Some(limit) = limit {
        plan.capacity_words = limit;
    }
    metadata[4..8].copy_from_slice(&[8, channels.len() as u32, output as u32, plan.capacity_words]);
    // A sentinel beyond the admitted output detects writes past the declared capacity.
    words.resize(output + plan.bytes() as usize / 4 + 1, 0);
    *words.last_mut().unwrap() = 0xfeed_cafe;
    for table in &codebook.tables {
        metadata.extend_from_slice(table.gpu_words());
    }
    let storage = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANS test events/output"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let params = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANS test tables"),
            contents: bytemuck::cast_slice(&metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let readback = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("ANS test readback"),
        size: storage.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bindings = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });
    let mut command = context.device().create_command_encoder(&Default::default());
    {
        let mut pass = command.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    command.copy_buffer_to_buffer(&storage, 0, &readback, 0, storage.size());
    context.queue().submit([command.finish()]);
    let (send, receive) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).unwrap()
        });
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receive.recv().unwrap().unwrap();
    let mapped = readback.slice(..).get_mapped_range().unwrap();
    let result: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
    assert_eq!(result.last(), Some(&0xfeed_cafe));
    drop(mapped);
    readback.unmap();
    (plan, result)
}

fn independent_decode(code: &EntropyCode, fragment: EncodedGroup<'_>, expected: &[Vec<u32>]) {
    let mut writer = BitWriter::new();
    super::super::serializer::write_ma_config(
        &mut writer,
        code,
        super::super::predictor::LosslessModularPredictor::Zero,
    )
    .unwrap();
    code.write_stream(&mut writer, &[], Some(fragment)).unwrap();
    let bit_len = writer.bit_len();
    writer.align_to_byte().unwrap();
    let bytes = writer.into_bytes();
    for multiplier in [1, 17, 1024] {
        let mut bits = jxl_bitstream::Bitstream::new(&bytes);
        let mut tree = jxl_coding::Decoder::parse(&mut bits, 6).unwrap();
        tree.begin(&mut bits).unwrap();
        let mut nodes = 1;
        while nodes != 0 {
            nodes -= 1;
            if tree.read_varint(&mut bits, 1).unwrap() == 0 {
                for context in 2..6 {
                    tree.read_varint(&mut bits, context).unwrap();
                }
            } else {
                tree.read_varint(&mut bits, 0).unwrap();
                nodes += 2;
            }
        }
        tree.finalize().unwrap();
        let mut decoder = jxl_coding::Decoder::parse(&mut bits, 4).unwrap();
        let start = bits.num_read_bits();
        decoder.begin(&mut bits).unwrap();
        assert_eq!(bits.num_read_bits(), start + 32, "actual ANS initial state");
        for (channel, values) in expected.iter().enumerate() {
            for &value in values {
                assert_eq!(
                    decoder
                        .read_varint_with_multiplier(
                            &mut bits,
                            3 - channel.min(3) as u32,
                            multiplier
                        )
                        .unwrap(),
                    value
                );
            }
        }
        decoder.finalize().unwrap();
        assert_eq!(bits.num_read_bits(), bit_len);
    }
}

#[test]
fn gpu_ans_preserves_shared_state_all_extra_widths_long_matches_and_capacity_failures() {
    let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let pipeline = pipeline(&context);
    for mode in [LosslessModularLz77::ZeroRuns, LosslessModularLz77::Greedy] {
        let raw_counts = [[1; RAW_SYMBOLS]; 4];
        let lz_counts = std::array::from_fn(|_| std::array::from_fn(|index| u64::from(index < 32)));
        let distance = if mode == LosslessModularLz77::ZeroRuns {
            [0; RAW_SYMBOLS]
        } else {
            [1; RAW_SYMBOLS]
        };
        let code = EntropyCode::Ans(Box::new(
            AnsCodebook::new(mode, &raw_counts, &lz_counts, &distance).unwrap(),
        ));
        let mut channels = Vec::new();
        let mut expected = Vec::new();
        for channel in 0..5 {
            let values: Vec<_> = (0..=32)
                .map(|token| {
                    if token == 0 {
                        0
                    } else {
                        u32::MAX >> (32 - token)
                    }
                })
                .collect();
            let mut events: Vec<_> = values.iter().copied().map(|value| raw(value, 0)).collect();
            let mut values = values;
            if channel == 1 {
                events.clear();
                values.clear();
            } else {
                for copied in [7, 22, 23, (1 << 19) + 7] {
                    if mode == LosslessModularLz77::ZeroRuns {
                        events.push(length(copied, 1));
                        values.extend(std::iter::repeat_n(0, copied as usize + 1));
                    } else {
                        events.push(length(copied, 2));
                        events.push(raw(120, 3));
                        values.extend(std::iter::repeat_n(
                            *values.last().unwrap(),
                            copied as usize,
                        ));
                    }
                }
            }
            channels.push(events);
            expected.push(values);
        }
        if mode == LosslessModularLz77::Greedy {
            let values = vec![1, 3, 7, 15, 31, 63, 127];
            let mut events: Vec<_> = values.iter().copied().map(|value| raw(value, 0)).collect();
            events.push(raw(0, 0));
            events.push(length((1 << 20) - 15, 2));
            events.push(raw(120, 3));
            events.push(length(7, 2));
            events.push(raw((1 << 20) - 7 + 119, 3));
            let mut values = values;
            values.resize((1 << 20) - 7, 0);
            values.extend([1, 3, 7, 15, 31, 63, 127]);
            channels.push(events);
            expected.push(values);
        }
        let (plan, words) = gpu_fragment(&context, &pipeline, code.ans().unwrap(), &channels, None);
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
        let fragment =
            validate_encoded_group(plan, bytemuck::cast_slice(&words), &artifacts).unwrap();
        independent_decode(&code, fragment, &expected);
        let (short, words) =
            gpu_fragment(&context, &pipeline, code.ans().unwrap(), &channels, Some(1));
        assert_eq!(words[short.byte_offset as usize / 4], 2);
        assert!(validate_encoded_group(short, bytemuck::cast_slice(&words), &artifacts).is_err());
        let missing = AnsCodebook::new(
            mode,
            &[[0; RAW_SYMBOLS]; 4],
            &[[0; LZ77_SYMBOLS]; 4],
            &[0; RAW_SYMBOLS],
        )
        .unwrap();
        let (plan, words) = gpu_fragment(&context, &pipeline, &missing, &channels, None);
        assert_eq!(words[plan.byte_offset as usize / 4], 2);
        assert!(validate_encoded_group(plan, bytemuck::cast_slice(&words), &artifacts).is_err());
    }
}
