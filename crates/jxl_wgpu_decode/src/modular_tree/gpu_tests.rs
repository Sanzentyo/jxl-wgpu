use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{BitReader, BitWriter};
use wgpu::util::DeviceExt;

use super::{
    ANS_SIGNATURE, AnsHistogramIr, EntropyCoderIr, EntropyDecoderIr, HybridIntegerConfigIr, Lz77Ir,
    MetadataEntropyCursor, PrefixHistogramIr, canonical_entries,
};
use crate::entropy::EntropyStreamParams;

const PROBE_TEMPLATE: &str = include_str!("../modular_entropy_probe.wgsl");
const ENTROPY_ABI: &str = include_str!("../modular_entropy_abi.wgsl");
const ENTROPY: &str = include_str!("../modular_entropy.wgsl");
const ENTROPY_ABI_MARKER: &str = "/*__JXL_MODULAR_ENTROPY_ABI__*/";
const ENTROPY_MARKER: &str = "/*__JXL_MODULAR_ENTROPY__*/";

const ERROR_TRUNCATED_BITS: u32 = 2;
const ERROR_TRAILING_BITS: u32 = 7;
const ERROR_ANS_STATE: u32 = 10;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ProbeParams {
    entropy: EntropyStreamParams,
    symbol_count: u32,
    distance_multiplier: u32,
    _reserved: [u32; 3],
}

#[derive(Debug)]
struct ProbeOutput {
    values: Vec<u32>,
    status: [u32; 4],
}

struct ProbeInput<'a> {
    descriptor: &'a EntropyDecoderIr,
    stream: &'a [u8],
    token_end: u32,
    contexts: &'a [u32],
    lz77_window_mask: u32,
    distance_multiplier: u32,
}

fn shader_source() -> String {
    PROBE_TEMPLATE
        .replace(ENTROPY_ABI_MARKER, ENTROPY_ABI)
        .replace(ENTROPY_MARKER, ENTROPY)
}

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("jxl-wgpu common entropy differential test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()
}

fn run_probe(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    input: ProbeInput<'_>,
) -> ProbeOutput {
    let ProbeInput {
        descriptor,
        stream,
        token_end,
        contexts,
        lz77_window_mask,
        distance_multiplier,
    } = input;
    let metadata = descriptor.pack_gpu_metadata().unwrap().words;
    let mut stream_upload = stream.to_vec();
    stream_upload.resize(stream_upload.len().max(8).div_ceil(4) * 4, 0);
    let stream_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("common entropy probe stream"),
        contents: &stream_upload,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("common entropy probe metadata"),
        contents: bytemuck::cast_slice(&metadata),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let context_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("common entropy probe contexts"),
        contents: bytemuck::cast_slice(contexts),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let symbol_count = u32::try_from(contexts.len()).unwrap();
    let params = ProbeParams {
        entropy: EntropyStreamParams {
            token_start: 0,
            token_end,
            lz77_window_mask,
        },
        symbol_count,
        distance_multiplier,
        _reserved: [0; 3],
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("common entropy probe params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let reconstruction_words = symbol_count
        .checked_add(lz77_window_mask.saturating_add(1))
        .unwrap()
        .max(1);
    let reconstruction = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("common entropy probe reconstruction"),
        size: u64::from(reconstruction_words) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let status = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("common entropy probe status"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let value_bytes = u64::from(symbol_count.max(1)) * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("common entropy probe readback"),
        size: value_bytes + 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("common entropy probe bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: stream_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: metadata_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: context_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: reconstruction.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: status.as_entire_binding(),
            },
        ],
    });
    let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("common entropy probe"),
    });
    commands.clear_buffer(&reconstruction, 0, None);
    commands.clear_buffer(&status, 0, None);
    {
        let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("common entropy probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    commands.copy_buffer_to_buffer(&reconstruction, 0, &staging, 0, value_bytes);
    commands.copy_buffer_to_buffer(&status, 0, &staging, value_bytes, 16);
    let submission = queue.submit([commands.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let values =
        bytemuck::cast_slice::<u8, u32>(&mapped[..value_bytes as usize])[..contexts.len()].to_vec();
    let status =
        *bytemuck::from_bytes::<[u32; 4]>(&mapped[value_bytes as usize..value_bytes as usize + 16]);
    drop(mapped);
    staging.unmap();
    ProbeOutput { values, status }
}

fn scalar_decode(
    descriptor: &EntropyDecoderIr,
    stream: &[u8],
    token_end: u32,
    contexts: &[usize],
    distance_multiplier: u32,
) -> Vec<u32> {
    let mut reader = BitReader::new(stream);
    let mut cursor = MetadataEntropyCursor::new(descriptor, contexts.len());
    cursor.begin(&mut reader).unwrap();
    let values = contexts
        .iter()
        .map(|&context| {
            cursor
                .read_varint(&mut reader, context, distance_multiplier)
                .unwrap_or_else(|error| {
                    panic!(
                        "scalar entropy context {context} failed at bit {}: {error:?}",
                        reader.bit_offset()
                    )
                })
        })
        .collect::<Vec<_>>();
    cursor.finalize().unwrap();
    let consumed = u32::try_from(reader.bit_offset()).unwrap();
    assert!(consumed <= token_end);
    let remaining = token_end - consumed;
    assert!(remaining <= 7);
    assert_eq!(reader.read_bits(remaining as u8).unwrap(), 0);
    values
}

fn write_ans_u8(writer: &mut BitWriter, value: u8) {
    if value == 0 {
        writer.write_bits(0, 1).unwrap();
        return;
    }
    writer.write_bits(1, 1).unwrap();
    let bits = u8::try_from(u8::BITS - 1 - value.leading_zeros()).unwrap();
    writer.write_bits(u64::from(bits), 3).unwrap();
    writer
        .write_bits(u64::from(value - (1 << bits)), bits)
        .unwrap();
}

fn parse_ans_histogram(writer: BitWriter) -> AnsHistogramIr {
    let mut reader = BitReader::new(writer.as_bytes());
    AnsHistogramIr::parse(&mut reader, 5).unwrap()
}

fn unary_histogram(symbol: u8) -> AnsHistogramIr {
    let mut writer = BitWriter::new();
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(0, 1).unwrap();
    write_ans_u8(&mut writer, symbol);
    parse_ans_histogram(writer)
}

fn binary_histogram() -> AnsHistogramIr {
    let mut writer = BitWriter::new();
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(1, 1).unwrap();
    write_ans_u8(&mut writer, 0);
    write_ans_u8(&mut writer, 1);
    writer.write_bits(1024, 12).unwrap();
    parse_ans_histogram(writer)
}

fn flat_histogram(alphabet_size: u8) -> AnsHistogramIr {
    let mut writer = BitWriter::new();
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(1, 1).unwrap();
    write_ans_u8(&mut writer, alphabet_size - 1);
    parse_ans_histogram(writer)
}

fn compressed_histogram() -> AnsHistogramIr {
    let mut writer = BitWriter::new();
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(0, 1).unwrap();
    write_ans_u8(&mut writer, 0);
    for _ in 0..2 {
        writer.write_bits(3, 3).unwrap();
        writer.write_bits(1, 1).unwrap();
    }
    writer.write_bits(7, 3).unwrap();
    writer.write_bits(1, 1).unwrap();
    parse_ans_histogram(writer)
}

fn ans_descriptor(histogram: AnsHistogramIr) -> EntropyDecoderIr {
    EntropyDecoderIr {
        lz77: None,
        context_to_cluster: vec![0],
        configs: vec![HybridIntegerConfigIr {
            split_exponent: 5,
            msb_in_token: 0,
            lsb_in_token: 0,
        }],
        coder: EntropyCoderIr::Ans {
            log_alphabet_size: 5,
            histograms: vec![histogram],
        },
    }
}

fn ans_stream_for_symbol(
    histogram: &AnsHistogramIr,
    symbol: u32,
    require_alias: Option<bool>,
    require_renormalization: Option<bool>,
) -> Option<(Vec<u8>, u32, bool)> {
    for index in 0..4096u32 {
        let bucket_index = usize::try_from(index >> histogram.log_bucket_size).ok()?;
        let position = index & ((1 << histogram.log_bucket_size) - 1);
        let (alias_symbol, cutoff, mut distribution, alias_offset, dist_xor) =
            histogram.buckets.get(bucket_index)?.fields();
        let map_alias = position >= cutoff;
        if require_alias.is_some_and(|required| required != map_alias) {
            continue;
        }
        let mapped_symbol = if map_alias {
            distribution ^= dist_xor;
            alias_symbol
        } else {
            u32::try_from(bucket_index).ok()?
        };
        if mapped_symbol != symbol || distribution == 0 {
            continue;
        }
        let offset = position + if map_alias { alias_offset } else { 0 };
        for (target, renormalized) in [(ANS_SIGNATURE, false), (ANS_SIGNATURE >> 16, true)] {
            if require_renormalization.is_some_and(|required| required != renormalized)
                || target < offset
                || !(target - offset).is_multiple_of(distribution)
            {
                continue;
            }
            let high = (target - offset) / distribution;
            if high >= 1 << 20 {
                continue;
            }
            let initial_state = (high << 12) | index;
            let mut writer = BitWriter::new();
            writer.write_bits(u64::from(initial_state), 32).unwrap();
            if renormalized {
                writer
                    .write_bits(u64::from(ANS_SIGNATURE & 0xffff), 16)
                    .unwrap();
            }
            return Some((
                writer.as_bytes().to_vec(),
                u32::try_from(writer.bit_len()).unwrap(),
                map_alias,
            ));
        }
    }
    None
}

fn prefix_hybrid_case() -> (EntropyDecoderIr, Vec<u8>, u32, Vec<usize>) {
    let entries = canonical_entries(&[1, 2, 3, 3]).unwrap();
    let descriptor = EntropyDecoderIr {
        lz77: None,
        context_to_cluster: vec![0],
        configs: vec![HybridIntegerConfigIr {
            split_exponent: 1,
            msb_in_token: 0,
            lsb_in_token: 0,
        }],
        coder: EntropyCoderIr::Prefix(vec![PrefixHistogramIr {
            entries: entries.clone(),
            single_symbol: None,
        }]),
    };
    let mut writer = BitWriter::new();
    for (token, extra, extra_bits) in [(0, 0, 0), (1, 0, 0), (2, 1, 1), (3, 2, 2)] {
        let entry = entries[token];
        writer
            .write_bits(u64::from(entry.bits), entry.bit_len)
            .unwrap();
        writer.write_bits(extra, extra_bits).unwrap();
    }
    writer.align_to_byte().unwrap();
    let token_end = u32::try_from(writer.bit_len()).unwrap();
    (descriptor, writer.into_bytes(), token_end, vec![0; 4])
}

fn lz77_case() -> (EntropyDecoderIr, Vec<usize>) {
    let descriptor = EntropyDecoderIr {
        lz77: Some(Lz77Ir {
            min_symbol: 8,
            min_length: 3,
            length_config: HybridIntegerConfigIr {
                split_exponent: 0,
                msb_in_token: 0,
                lsb_in_token: 0,
            },
        }),
        context_to_cluster: vec![0, 1, 2],
        configs: vec![
            HybridIntegerConfigIr {
                split_exponent: 4,
                msb_in_token: 0,
                lsb_in_token: 0,
            },
            HybridIntegerConfigIr {
                split_exponent: 4,
                msb_in_token: 0,
                lsb_in_token: 0,
            },
            HybridIntegerConfigIr {
                split_exponent: 0,
                msb_in_token: 0,
                lsb_in_token: 0,
            },
        ],
        coder: EntropyCoderIr::Prefix(vec![
            PrefixHistogramIr::single(5).unwrap(),
            PrefixHistogramIr::single(8).unwrap(),
            PrefixHistogramIr::single(0).unwrap(),
        ]),
    };
    (descriptor, vec![0, 1, 1, 1])
}

#[test]
fn common_entropy_executor_matches_scalar_for_all_distribution_and_termination_forms() {
    let source = shader_source();
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();

    let Some((device, queue)) = device() else {
        eprintln!("skipping common entropy GPU differential test: no adapter");
        return;
    };
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("jxl-wgpu common entropy probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("jxl-wgpu common entropy probe"),
        layout: None,
        module: &module,
        entry_point: Some("probe_entropy"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let (prefix, stream, token_end, contexts) = prefix_hybrid_case();
    let expected = scalar_decode(&prefix, &stream, token_end, &contexts, 0);
    assert_eq!(expected, [0, 1, 3, 6]);
    let actual = run_probe(
        &device,
        &queue,
        &pipeline,
        ProbeInput {
            descriptor: &prefix,
            stream: &stream,
            token_end,
            contexts: &contexts
                .iter()
                .map(|&value| value as u32)
                .collect::<Vec<_>>(),
            lz77_window_mask: 0,
            distance_multiplier: 0,
        },
    );
    assert_eq!(actual.values, expected);
    assert_eq!(actual.status, [0, token_end, 0, 0]);

    let ans_cases = [
        ("unary", unary_histogram(7), 7, Some(true)),
        ("binary", binary_histogram(), 1, None),
        ("flat", flat_histogram(5), 4, None),
        ("compressed", compressed_histogram(), 0, Some(false)),
    ];
    let mut observed_direct = false;
    let mut observed_alias = false;
    for (name, histogram, symbol, require_alias) in ans_cases {
        let (stream, token_end, used_alias) =
            ans_stream_for_symbol(&histogram, symbol, require_alias, None)
                .unwrap_or_else(|| panic!("{name} ANS form has no reversible probe state"));
        observed_alias |= used_alias;
        observed_direct |= !used_alias;
        let descriptor = ans_descriptor(histogram);
        let expected = scalar_decode(&descriptor, &stream, token_end, &[0], 0);
        assert_eq!(expected, [symbol], "{name} scalar symbol");
        let actual = run_probe(
            &device,
            &queue,
            &pipeline,
            ProbeInput {
                descriptor: &descriptor,
                stream: &stream,
                token_end,
                contexts: &[0],
                lz77_window_mask: 0,
                distance_multiplier: 0,
            },
        );
        assert_eq!(actual.values, expected, "{name} GPU symbol");
        assert_eq!(
            actual.status,
            [0, token_end, ANS_SIGNATURE, 0],
            "{name} GPU termination",
        );
    }
    assert!(observed_direct && observed_alias);

    let (lz77, contexts) = lz77_case();
    let expected = scalar_decode(&lz77, &[], 0, &contexts, 0);
    assert_eq!(expected, [5, 5, 5, 5]);
    let actual = run_probe(
        &device,
        &queue,
        &pipeline,
        ProbeInput {
            descriptor: &lz77,
            stream: &[],
            token_end: 0,
            contexts: &contexts
                .iter()
                .map(|&value| value as u32)
                .collect::<Vec<_>>(),
            lz77_window_mask: 0,
            distance_multiplier: 0,
        },
    );
    assert_eq!(actual.values, expected);
    assert_eq!(actual.status, [0, 0, 0, 4]);

    let single = EntropyDecoderIr {
        lz77: None,
        context_to_cluster: vec![0],
        configs: vec![HybridIntegerConfigIr {
            split_exponent: 0,
            msb_in_token: 0,
            lsb_in_token: 0,
        }],
        coder: EntropyCoderIr::Prefix(vec![PrefixHistogramIr::single(0).unwrap()]),
    };
    for (stream, token_end) in [([1_u8].as_slice(), 1_u32), ([0_u8].as_slice(), 8_u32)] {
        let actual = run_probe(
            &device,
            &queue,
            &pipeline,
            ProbeInput {
                descriptor: &single,
                stream,
                token_end,
                contexts: &[0],
                lz77_window_mask: 0,
                distance_multiplier: 0,
            },
        );
        assert_eq!(actual.status[0], ERROR_TRAILING_BITS);
    }

    let unary = unary_histogram(7);
    let invalid_state = ANS_SIGNATURE + 1;
    let mut invalid_writer = BitWriter::new();
    invalid_writer
        .write_bits(u64::from(invalid_state), 32)
        .unwrap();
    let invalid = run_probe(
        &device,
        &queue,
        &pipeline,
        ProbeInput {
            descriptor: &ans_descriptor(unary),
            stream: invalid_writer.as_bytes(),
            token_end: 32,
            contexts: &[0],
            lz77_window_mask: 0,
            distance_multiplier: 0,
        },
    );
    assert_eq!(invalid.status[0], ERROR_ANS_STATE);
    assert_eq!(invalid.status[2], invalid_state);

    let flat = flat_histogram(5);
    let (renormalized, _, _) = ans_stream_for_symbol(&flat, 0, None, Some(true)).unwrap();
    let truncated = run_probe(
        &device,
        &queue,
        &pipeline,
        ProbeInput {
            descriptor: &ans_descriptor(flat),
            stream: &renormalized,
            token_end: 32,
            contexts: &[0],
            lz77_window_mask: 0,
            distance_multiplier: 0,
        },
    );
    assert_eq!(truncated.status[0], ERROR_TRUNCATED_BITS);
}

const _: () = {
    assert!(std::mem::size_of::<ProbeParams>() == 32);
    assert!(std::mem::align_of::<ProbeParams>() == 16);
    assert!(std::mem::offset_of!(ProbeParams, entropy) == 0);
    assert!(std::mem::offset_of!(ProbeParams, symbol_count) == 12);
    assert!(std::mem::offset_of!(ProbeParams, distance_multiplier) == 16);
};
