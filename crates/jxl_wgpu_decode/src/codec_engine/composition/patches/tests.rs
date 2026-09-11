use super::*;
use jxl_gpu_bitstream::{BitRange, BitWriter};

fn backend() -> WgpuBackend {
    pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap()
}

// Independent prefix stream: one 1x1 replacement in reference slot 0. The LZ77 variant copies
// seven zero values from distance one, across source geometry, occurrence count and destination.
fn prefix(lz77: bool, unary: bool) -> BitWriter {
    let mut bits = BitWriter::new();
    bits.write_bits(u64::from(lz77), 1).unwrap();
    if lz77 {
        bits.write_bits(0, 2).unwrap(); // Minimum symbol 224.
        bits.write_bits(0, 2).unwrap(); // Minimum length 3.
        bits.write_bits(8, 4).unwrap(); // Direct 8-bit lengths.
    }
    bits.write_bits(1, 1).unwrap(); // Simple context map.
    bits.write_bits(0, 2).unwrap(); // One cluster, including the distance context.
    bits.write_bits(1, 1).unwrap(); // Prefix.
    bits.write_bits(15, 4).unwrap(); // Direct prefix tokens.
    if unary {
        bits.write_bits(0, 1).unwrap(); // Alphabet {0}, no histogram or payload bits.
    } else {
        bits.write_bits(1, 1).unwrap();
        bits.write_bits(if lz77 { 7 } else { 0 }, 4).unwrap();
        if lz77 {
            bits.write_bits(100, 7).unwrap();
        } // Alphabet 229.
        bits.write_bits(1, 2).unwrap(); // Simple histogram.
        bits.write_bits(if lz77 { 2 } else { 1 }, 2).unwrap();
        let width = if lz77 { 8 } else { 1 };
        bits.write_bits(0, width).unwrap();
        bits.write_bits(1, width).unwrap();
        if lz77 {
            bits.write_bits(228, width).unwrap();
            for (code, width) in [(1, 2), (0, 1), (3, 2), (0, 1), (1, 2)] {
                bits.write_bits(code, width).unwrap();
            }
        } else {
            for value in [1, 0, 0, 0, 0, 0, 0, 0, 0, 1] {
                bits.write_bits(value, 1).unwrap();
            }
        }
    }
    bits
}

fn plan(bits: BitWriter) -> (Plan, Arc<GpuCodestream>) {
    let end = bits.bit_len() as u64;
    let bytes: Arc<[u8]> = bits.into_bytes().into();
    let source =
        Arc::new(GpuCodestream::from_shared(bytes.clone(), 0..bytes.len(), false).unwrap());
    let compact = include_str!("../../../../test-data/testsrc_modular_orientation_rgb_1.jxl.hex")
        .split_whitespace()
        .collect::<String>();
    let fixture: Vec<_> = compact
        .as_bytes()
        .chunks_exact(2)
        .map(|bytes| u8::from_str_radix(std::str::from_utf8(bytes).unwrap(), 16).unwrap())
        .collect();
    let parsed = jxl_gpu_bitstream::parse(&fixture, Default::default()).unwrap();
    let mut frame = parsed
        .codestream_inventory(Default::default())
        .unwrap()
        .frames
        .remove(0);
    frame.sections.truncate(1);
    frame.sections[0].kind = FrameSectionKind::Single;
    frame.sections[0].bits = BitRange {
        offset: 0,
        length: end,
    };
    frame.width = 1;
    frame.height = 1;
    (
        Plan::new(&source, &frame, &[], [[1; 4]; 4], 40).unwrap(),
        source,
    )
}

fn words(backend: &WgpuBackend, dictionary: &Dictionary) -> Vec<u32> {
    let layout = jxl_gpu_formats::ImageLayout::packed(
        jxl_gpu_protocol::Extent2d::new(dictionary.count * dictionary.stride, 1),
        jxl_gpu_formats::PixelFormat::non_color(
            jxl_gpu_formats::SampleKind::Unsigned,
            32,
            &[jxl_gpu_formats::Channel::X],
        ),
    )
    .unwrap();
    let frame = jxl_wgpu::GpuImageFrame {
        token: jxl_gpu_protocol::SubmissionToken(1),
        outputs: vec![jxl_wgpu::GpuImageOutput {
            id: jxl_gpu_protocol::OutputId(0),
            layout,
            buffer: dictionary.commands.clone(),
        }],
        changed: Default::default(),
    };
    jxl_wgpu::ImageReadbackPipeline::new(backend)
        .submit(&frame)
        .unwrap()
        .wait()
        .unwrap()
        .frame
        .outputs[0]
        .bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn drained(backend: &WgpuBackend) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while backend.transient_memory_budget().snapshot().reserved_bytes != 0
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
}

#[test]
fn prefix_unary_and_lz77_produce_exact_resident_commands_and_cursors() {
    let backend = backend();
    for (lz77, unary) in [(false, false), (true, false), (false, true)] {
        let bits = prefix(lz77, unary);
        let end = bits.bit_len() as u64;
        let (plan, source) = plan(bits);
        assert_eq!(plan.history_words != 0, lz77);
        let (dictionary, submissions) = plan
            .submit(backend.clone(), source)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(dictionary.end, end);
        assert_eq!(dictionary.count, u32::from(!unary));
        assert_eq!(submissions, if unary { 1 } else { 2 });
        if !unary {
            assert_eq!(
                words(&backend, &dictionary),
                [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
            );
        }
        drop(dictionary);
        drained(&backend);
    }
}

#[test]
fn dictionary_count_and_replay_are_accounted_retryable_and_cancellation_owned() {
    let backend = backend();
    let (plan, source) = plan(prefix(true, false));
    let memory = backend.transient_memory_budget();
    let bytes = (STATE_WORDS + u64::from(plan.history_words)) * 4
        + plan.windows.stream_bytes()
        + plan.metadata.len() as u64 * 4
        + 128
        + STATUS_BYTES
        + 4;
    let blocker = memory
        .try_reserve(memory.snapshot().available_bytes - (bytes - 1))
        .unwrap();
    assert!(matches!(
        plan.clone().submit(backend.clone(), source.clone()),
        Err(Error::MemoryBackpressure(_))
    ));
    assert_eq!(memory.snapshot().available_bytes, bytes - 1);
    drop(blocker);
    for phase in 0..3 {
        let mut pending = plan
            .clone()
            .submit(backend.clone(), source.clone())
            .unwrap();
        if phase != 0 {
            pending.completion.wait().unwrap();
            if phase == 2 {
                let blocker = memory
                    .try_reserve(memory.snapshot().available_bytes - 43)
                    .unwrap();
                assert!(matches!(
                    pending.advance(),
                    Err(Error::MemoryBackpressure(_))
                ));
                drop(blocker);
            } else {
                assert!(pending.advance().unwrap().is_none());
                assert_eq!(pending.submissions, 2);
            }
        }
        drop(pending);
        drained(&backend);
    }
    let (dictionary, _) = plan
        .submit(backend.clone(), source)
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(memory.snapshot().reserved_bytes, 44);
    drop(dictionary);
    drained(&backend);
}
