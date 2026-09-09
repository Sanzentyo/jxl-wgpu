use super::*;
use crate::modular_side_image::ModularSideImagePlan;
use crate::modular_transform::ModularChannelTopology;
use crate::modular_tree::{
    AnsBucketIr, AnsHistogramIr, EntropyCoderIr, EntropyDecoderIr, HybridIntegerConfigIr,
    MaConfigIr, MaTreeNodeIr, PrefixHistogramIr,
};
use crate::wgpu_engine::ModularSideImagePipeline;
use jxl_gpu_bitstream::{BitReader, BitWriter, PrefixCodeEntry};
use jxl_wgpu::{KernelVariant, WgpuBackend};
use wgpu::util::DeviceExt;

#[test]
fn continuation_shader_matches_the_160_byte_parameter_abi() {
    let module = naga::front::wgsl::parse_str(&shader_source()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let (_, params) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("Params"))
        .unwrap();
    let naga::TypeInner::Struct { members, span } = &params.inner else {
        panic!("Params struct");
    };
    assert_eq!(*span, 160);
    assert_eq!(members.last().unwrap().name.as_deref(), Some("stream_end"));
    assert_eq!(members.last().unwrap().offset, 156);
}

#[test]
fn continuation_status_requires_host_bounds_and_the_expected_group() {
    let status = GpuHfCoefficientStatus {
        error_code: 1,
        group_index: 7,
        bit_cursor: 101,
        token_end: 143,
        ..Zeroable::zeroed()
    };
    assert_eq!(status.validate_cursor(7, 97, 143).unwrap(), 101);
    assert!(status.validate(7).is_err());
    for (group, start, end) in [
        (6, 97, 143),
        (7, 102, 143),
        (7, 97, 100),
        (7, 97, 144),
        (7, 150, 143),
    ] {
        assert!(status.validate_cursor(group, start, end).is_err());
    }
    for code in (0..=64)
        .chain(std::iter::once(u32::MAX))
        .filter(|&code| code != 1)
    {
        assert!(
            GpuHfCoefficientStatus {
                error_code: code,
                ..status
            }
            .validate_cursor(7, 97, 143)
            .is_err()
        );
    }
    let end = GpuHfCoefficientStatus {
        bit_cursor: 143,
        ..status
    };
    end.validate(7).unwrap();
    assert_eq!(end.validate_cursor(7, 97, 143).unwrap(), 143);
}

fn multilf_plan(stream_limit: u64) -> HfCoefficientExecutionPlan {
    use crate::vardct_artifact::{HfMetadataArtifactConfig, VarDctArtifactDeviceLimits};
    let digits = include_str!("../../test-data/testsrc_vardct_progressive_multilf.jxl.hex")
        .split_whitespace()
        .collect::<String>();
    let bytes = digits
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect::<Vec<_>>();
    let parsed = jxl_gpu_bitstream::parse(&bytes, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let packet = BoundedVarDctPacketPlan::parse(parsed.codestream(), &inventory).unwrap();
    let artifacts = packet
        .groups
        .iter()
        .map(|group| {
            let [width, height] = group.block_extent();
            VarDctArtifactLayout::plan(
                &HfMetadataArtifactConfig {
                    blocks_width: width,
                    blocks_height: height,
                    block_info_entries: group.task_capacity,
                    strategy_offset_words: 0,
                    hf_mul_offset_words: group.task_capacity,
                    raw_metadata_words: u64::from(group.task_capacity) * 2,
                    pass_group_dim_blocks: packet.profile.group_dimension / 8,
                    lf_stride: packet.block_extent()[0],
                    correlation_stride: packet.block_extent()[0].div_ceil(8),
                    correlation_width: width.div_ceil(8),
                    correlation_height: height.div_ceil(8),
                    destination_origin: [group.rect.x, group.rect.y],
                    afv_basis_offset: 0,
                    quant_offset: 0,
                    correlation_offset: 0,
                    global_scale: packet.global_scale,
                    channel_shifts: packet.profile.channel_shifts,
                    lf_offsets: [0; 3],
                    lf_strides: [packet.block_extent()[0]; 3],
                    matrix_offsets: [0; 27],
                },
                VarDctArtifactDeviceLimits {
                    max_buffer_size: u64::MAX,
                    max_storage_buffer_binding_size: u64::MAX,
                    storage_binding_alignment: 256,
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    HfCoefficientExecutionPlan::new(
        &packet,
        packet.hf_coefficients.as_ref().unwrap(),
        &artifacts,
        parsed.codestream().len() as u64,
        stream_limit,
    )
    .unwrap()
}

#[test]
fn a_pass_termination_change_reaches_every_window_without_changing_other_passes() {
    let mut plan = multilf_plan(128);
    assert!(plan.groups.len() > 1);
    let selected = plan.groups[0].params[0].global_group_index;
    let derived_params = |group: &HfCoefficientGroupExecutionPlan| {
        group
            .passes
            .iter()
            .enumerate()
            .flat_map(|(pass_index, pass)| {
                pass.streams
                    .batches()
                    .flat_map(|batch| batch.segments().to_vec())
                    .map(move |segment| group.params_for_segment(pass_index, segment).unwrap())
            })
            .collect::<Vec<_>>()
    };
    assert!(
        derived_params(&plan.groups[0])
            .iter()
            .filter(|p| p.global_group_index == selected)
            .count()
            > 1
    );
    plan.set_stream_end(selected, HfCoefficientStreamEnd::Continuation)
        .unwrap();
    assert!(matches!(
        plan.set_stream_end(u32::MAX, HfCoefficientStreamEnd::Continuation),
        Err(HfCoefficientPlanError::MissingPassGroup { .. })
    ));
    for group in &plan.groups {
        let segments = derived_params(group);
        for params in group.params.iter().chain(&segments) {
            assert_eq!(
                params.stream_end,
                u32::from(params.global_group_index == selected)
            );
        }
    }
    plan.set_stream_end(selected, HfCoefficientStreamEnd::Packet)
        .unwrap();
    assert!(plan.groups.iter().all(|group| {
        let segments = derived_params(group);
        group
            .params
            .iter()
            .chain(&segments)
            .all(|p| p.stream_end == 0)
    }));
}

#[test]
fn image_wide_pass_barriers_keep_state_and_validation_offsets_in_every_window() {
    let whole = multilf_plan(u64::MAX);
    let windowed = multilf_plan(128);
    assert_eq!(whole.pass_count(), 3);
    assert_eq!(windowed.pass_count(), whole.pass_count());
    assert!(whole.groups.len() > 1);
    assert!(!whole.uses_bounded_stream_windows());
    assert!(windowed.uses_bounded_stream_windows());
    assert_eq!(whole.status_bytes(), windowed.status_bytes());
    let spatial_groups = whole
        .groups
        .iter()
        .map(|g| g.passes[0].parameter_range.len())
        .sum::<usize>();
    for plan in [&whole, &windowed] {
        let mut seen = std::collections::BTreeSet::new();
        let mut batches = 0;
        for pass_index in 0..plan.pass_count() {
            let mut pass_groups = std::collections::BTreeSet::new();
            for (lf, group) in plan.groups.iter().enumerate() {
                let pass = &group.passes[pass_index];
                let lane_count = group.passes[0].parameter_range.len();
                assert_eq!(
                    pass.parameter_range,
                    pass_index * lane_count..(pass_index + 1) * lane_count
                );
                for batch in pass.streams.batches() {
                    batches += 1;
                    for &segment in batch.segments() {
                        let derived = group.params_for_segment(pass_index, segment).unwrap();
                        let original_index = pass.parameter_range.start + segment.group_index;
                        let original = &whole.groups[lf].params[original_index];
                        assert_eq!(
                            derived.global_group_index as usize / spatial_groups,
                            pass_index
                        );
                        assert_eq!(derived.status_index as usize, original_index);
                        assert_eq!(derived.global_group_index, original.global_group_index);
                        assert_eq!(
                            derived.execution_state_base_words,
                            original.execution_state_base_words
                        );
                        assert_eq!(
                            derived.lz77_window_base_words,
                            original.lz77_window_base_words
                        );
                        assert_eq!(derived.metadata_base_words, original.metadata_base_words);
                        assert_eq!(derived.order_base_words, original.order_base_words);
                        assert_eq!(derived.coeff_shift, original.coeff_shift);
                        pass_groups.insert(derived.global_group_index);
                        seen.insert(derived.global_group_index);
                        let mut outside = segment;
                        outside.group_index = lane_count;
                        assert!(group.params_for_segment(pass_index, outside).is_none());
                        assert!(
                            group
                                .params_for_segment(plan.pass_count(), segment)
                                .is_none()
                        );
                    }
                }
            }
            assert_eq!(pass_groups.len(), spatial_groups);
        }
        assert_eq!(seen.len(), spatial_groups * plan.pass_count());
        assert_eq!(batches, plan.stream_batch_count());
    }
}

fn descriptor(kind: u32) -> EntropyDecoderIr {
    let coder = match kind {
        0 => EntropyCoderIr::Prefix(vec![PrefixHistogramIr {
            entries: vec![PrefixCodeEntry::default()],
            single_symbol: Some(0),
        }]),
        1 => EntropyCoderIr::Prefix(vec![PrefixHistogramIr {
            entries: vec![
                PrefixCodeEntry {
                    bit_len: 1,
                    bits: 0,
                },
                PrefixCodeEntry {
                    bit_len: 1,
                    bits: 1,
                },
            ],
            single_symbol: None,
        }]),
        2 => EntropyCoderIr::Ans {
            log_alphabet_size: 5,
            histograms: vec![AnsHistogramIr {
                buckets: (0..32)
                    .map(|index| {
                        let distribution = if index == 0 { 4096 } else { 0 };
                        AnsBucketIr {
                            symbol_cutoff_dist: distribution << 16,
                            offset_dist_xor: (index * 128) | ((distribution ^ 4096) << 16),
                        }
                    })
                    .collect(),
                log_bucket_size: 7,
                single_symbol: Some(0),
            }],
        },
        _ => unreachable!(),
    };
    EntropyDecoderIr {
        lz77: None,
        context_to_cluster: vec![0],
        configs: vec![HybridIntegerConfigIr {
            split_exponent: 0,
            msb_in_token: 0,
            lsb_in_token: 0,
        }],
        coder,
    }
}

fn bytes_after_submission(
    backend: &WgpuBackend,
    staging: &wgpu::Buffer,
    submission: wgpu::SubmissionIndex,
) -> Vec<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
    staging.unmap();
    bytes
}

struct Probe {
    pipeline: HfCoefficientPipeline,
    stream: wgpu::Buffer,
    metadata: wgpu::Buffer,
    reconstruction: wgpu::Buffer,
    artifact: wgpu::Buffer,
    orders: wgpu::Buffer,
    coefficients: wgpu::Buffer,
    sink: wgpu::Buffer,
    status: wgpu::Buffer,
    staging: wgpu::Buffer,
    params: HfCoefficientPassParams,
}

impl Probe {
    fn new(
        backend: &WgpuBackend,
        data: &[u8],
        descriptor: &EntropyDecoderIr,
        start: u32,
        end: u32,
    ) -> Self {
        let device = backend.device();
        let buffer = |label, words: &[u32], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(words),
                usage,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let mut bytes = data.to_vec();
        bytes.resize(bytes.len().div_ceil(4) * 4 + 4, 0);
        let stream = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("AC then Modular bitstream"),
            contents: &bytes,
            usage: storage | wgpu::BufferUsages::COPY_DST,
        });
        let mut metadata = descriptor.pack_gpu_metadata().unwrap().words;
        let block_context =
            append_block_context_tables(&mut metadata, &[0; 39], &[], &[vec![], vec![], vec![]])
                .unwrap();
        let context_map_offset_words = metadata.len() as u32;
        metadata.resize(metadata.len() + 2 * 495, 0);
        let mut artifact = vec![0; 13];
        artifact[0] = 1; // One DCT8 task at block (0, 0).
        artifact[1 + 4] = 1;
        artifact[1 + 5] = 1;
        artifact[1 + 6] = 1;
        artifact[1 + 9] = 192;
        artifact[1 + 11] = 7 << 8;
        let sink = HfCoefficientSinkParams {
            task_metadata_offset_words: 1,
            task_count: 1,
            coefficient_words: 192,
            ..Zeroable::zeroed()
        };
        let params = HfCoefficientPassParams {
            entropy: EntropyStreamParams {
                token_start: start,
                token_end: end,
                lz77_window_mask: 0,
            },
            stream_token_end: end,
            window_yield_end: end,
            window_flags: 3,
            execution_state_base_words: 3,
            block_width: 1,
            block_height: 1,
            blocks_per_row: 1,
            num_hf_presets: 2,
            num_block_clusters: 1,
            context_map_offset_words,
            lf_plane_stride_words: 1,
            global_group_index: 7,
            block_context,
            stream_end: HfCoefficientStreamEnd::Continuation as u32,
            ..Zeroable::zeroed()
        };
        Self {
            pipeline: HfCoefficientPipeline::new(device),
            stream,
            metadata: buffer("AC descriptors", &metadata, storage),
            reconstruction: buffer("AC state and LF zeros", &[0; 3 + 116], storage),
            artifact: buffer("AC one task", &artifact, storage),
            orders: buffer("unused zero AC orders", &[0; 4], storage),
            coefficients: buffer("zero coefficients", &[0; 192], storage),
            sink: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("AC sink"),
                contents: bytemuck::bytes_of(&sink),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
            status: buffer(
                "AC cursor status",
                &[0; 8],
                storage | wgpu::BufferUsages::COPY_SRC,
            ),
            staging: buffer(
                "AC cursor readback",
                &[0; 8],
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            params,
        }
    }

    fn run(&self, backend: &WgpuBackend) -> GpuHfCoefficientStatus {
        let device = backend.device();
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("AC cursor params"),
            contents: bytemuck::bytes_of(&self.params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        self.pipeline.encode(
            device,
            &mut encoder,
            HfCoefficientBuffers {
                codestream: &self.stream,
                entropy_bundle: &self.metadata,
                reconstruction: &self.reconstruction,
                params: &params,
                status: &self.status,
                artifact: &self.artifact,
                order_table: &self.orders,
                coefficients: &self.coefficients,
                sink_params: &self.sink,
            },
            1,
        );
        encoder.copy_buffer_to_buffer(&self.status, 0, &self.staging, 0, 32);
        let bytes = bytes_after_submission(
            backend,
            &self.staging,
            backend.queue().submit([encoder.finish()]),
        );
        bytemuck::pod_read_unaligned(&bytes)
    }
}

#[test]
fn gpu_ac_cursor_hands_unaligned_streams_to_modular_and_rejects_corrupt_ans() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let modular = ModularSideImagePipeline::new(&backend, KernelVariant::Lanes64);
    for kind in 0..3 {
        let entropy = descriptor(kind);
        let ma = MaConfigIr {
            nodes: vec![MaTreeNodeIr::Leaf {
                cluster: 0,
                predictor: 0,
                offset: 17,
                multiplier: 1,
            }],
            max_depth: 0,
            entropy: entropy.clone(),
        };
        for start in 0..8 {
            let mut writer = BitWriter::new();
            writer.write_bits((1 << start) - 1, start as u8).unwrap();
            writer.write_bits(1, 1).unwrap(); // Select preset 1.
            if kind == 2 {
                writer.write_bits(0x13_0000, 32).unwrap();
            }
            if kind == 1 {
                writer.write_bits(0, 3).unwrap();
            } // Three zero nonzero-counts.
            let cursor = writer.bit_len() as u32;
            writer.write_bits(1, 1).unwrap(); // Global MA tree for the following Modular image.
            writer.write_bits(1, 1).unwrap(); // Default WP.
            writer.write_bits(0, 2).unwrap(); // No transforms.
            if kind == 2 {
                writer.write_bits(0x13_0000, 32).unwrap();
            }
            if kind == 1 {
                writer.write_bits(0, 6).unwrap();
            }
            let modular_end = writer.bit_len() as u32;
            writer.write_bits(0x1a35, 13).unwrap(); // Another unaligned descriptor follows Modular.
            let packet_end = writer.bit_len() as u32;
            let mut probe = Probe::new(&backend, writer.as_bytes(), &entropy, start, packet_end);
            let status = probe.run(&backend);
            assert_eq!(
                status.validate_cursor(7, start, packet_end).unwrap(),
                cursor,
                "kind {kind} start {start}"
            );
            // This field counts LZ history entries; these descriptors have no LZ77 stream.
            assert_eq!(status.decoded_symbols, 0);
            assert_eq!(status.selected_preset, 1);
            assert_eq!(status.nonzero_coefficients, 0);
            let mut reader = BitReader::new(writer.as_bytes());
            reader.skip_bits(u64::from(cursor)).unwrap();
            let plan = ModularSideImagePlan::parse(
                &mut reader,
                ModularChannelTopology::full_resolution(3, 2, 8, 1, Default::default()).unwrap(),
                8,
                9,
                Some(&ma),
            )
            .unwrap();
            let source = crate::GpuCodestream::from_shared(
                writer.as_bytes().to_vec().into(),
                0..writer.as_bytes().len(),
                false,
            )
            .unwrap();
            let stream = modular
                .plan_source(&source, &plan, packet_end, 1024)
                .unwrap();
            let recording = modular
                .record_source(&backend, &source, &plan, &stream)
                .unwrap();
            let pixels = backend.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("six Modular samples"),
                size: 24,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut job = recording.finish();
            let mut copies = backend.device().create_command_encoder(&Default::default());
            copies.copy_buffer_to_buffer(job.arena(), 0, &pixels, 0, 24);
            let submission = backend
                .queue()
                .submit([job.take_commands().unwrap(), copies.finish()]);
            let bytes = bytes_after_submission(&backend, &pixels, submission.clone());
            assert_eq!(bytemuck::cast_slice::<u8, i32>(&bytes), &[17; 6]);
            job.mark_status_mapped();
            let (tx, rx) = std::sync::mpsc::channel();
            job.status_staging()
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            backend
                .device()
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let modular_status = job.finish_status().unwrap();
            assert!(modular_status.is_ok());
            assert_eq!(modular_status.cursor, modular_end);

            probe.params.stream_end = HfCoefficientStreamEnd::Packet as u32;
            assert!(matches!(
                probe.run(&backend).validate(7),
                Err(GpuHfCoefficientError::TrailingBits { .. })
            ));
            probe.params.stream_end = HfCoefficientStreamEnd::Continuation as u32;
            probe.params.window_flags = 1; // A valid end may precede the last input window.
            assert_eq!(
                probe
                    .run(&backend)
                    .validate_cursor(7, start, packet_end)
                    .unwrap(),
                cursor
            );
            if kind == 1 {
                probe.params.window_yield_end = start + 2;
                assert_eq!(probe.run(&backend).error_code, 14);
                probe.params.window_flags = 2;
                probe.params.window_yield_end = packet_end;
                assert_eq!(
                    probe
                        .run(&backend)
                        .validate_cursor(7, start, packet_end)
                        .unwrap(),
                    cursor
                );
            }
            if kind == 2 {
                probe.params.window_flags = 3;
                let mut corrupt = writer.as_bytes().to_vec();
                let bit = start + 1;
                corrupt[bit as usize / 8] ^= 1 << (bit % 8);
                corrupt.resize(corrupt.len().div_ceil(4) * 4, 0);
                backend.queue().write_buffer(&probe.stream, 0, &corrupt);
                assert!(matches!(
                    probe.run(&backend).validate_cursor(7, start, packet_end),
                    Err(GpuHfCoefficientError::AnsState { .. })
                ));
                probe.params.entropy.token_end = start + 32;
                assert!(matches!(
                    probe.run(&backend).validate_cursor(7, start, packet_end),
                    Err(GpuHfCoefficientError::TruncatedBits { .. })
                ));
            }
        }
    }
}
