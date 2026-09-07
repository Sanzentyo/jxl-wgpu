use super::*;
use crate::modular_transform::{ModularChannelTopology, ModularTransformLimits};
use crate::modular_tree::{MaTreeLimits, parse_ma_config};
use crate::vardct_frontend::{LfGlobalPrefix, parse_lf_group_header_reader};
use jxl_gpu_bitstream::{
    BitReader, FrameEncoding, FrameSectionKind, InventoryLimits, SampleBitDepth,
};

#[path = "../../../../tests/common/extra_channel_oracle.rs"]
mod oracle;

fn fixtures() -> [(&'static str, &'static str); 6] {
    [
        (
            "data_only",
            include_str!("../../../../test-data/vardct_extras_data_only.jxl.hex"),
        ),
        (
            "rgb12",
            include_str!("../../../../test-data/vardct_extras_rgb12.jxl.hex"),
        ),
        (
            "gray8",
            include_str!("../../../../test-data/vardct_extras_gray8.jxl.hex"),
        ),
        (
            "gray_alpha",
            include_str!("../../../../test-data/vardct_extras_gray_alpha.jxl.hex"),
        ),
        (
            "rgba",
            include_str!("../../../../test-data/vardct_extras_rgba.jxl.hex"),
        ),
        (
            "transformed",
            include_str!("../../../../test-data/vardct_extras_transformed.jxl.hex"),
        ),
    ]
}

#[test]
fn vardct_global_extra_planes_and_the_following_lf_cursor_are_reconstructed_on_gpu() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let pipeline = ModularSideImagePipeline::new(&backend, KernelVariant::Lanes64);
    let device = backend.device();
    for (name, hex) in fixtures() {
        let hex = hex.split_whitespace().collect::<String>();
        let encoded = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let parsed = jxl_gpu_bitstream::parse(&encoded, Default::default()).unwrap();
        let inventory = parsed
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let image = &inventory.image_header;
        let frame = &inventory.frames[0];
        assert_eq!(frame.encoding, FrameEncoding::VarDct);
        assert_eq!(frame.sections.len(), 1);
        assert_eq!(frame.sections[0].kind, FrameSectionKind::Single);
        let packet = frame.sections[0].bits;
        let end = u32::try_from(packet.end().unwrap()).unwrap();
        let prefix = LfGlobalPrefix::parse(parsed.codestream(), packet).unwrap();
        let mut reader = BitReader::new(parsed.codestream());
        reader.skip_bits(prefix.suffix_bit_offset).unwrap();
        let global_ma = prefix
            .global_ma_tree_bit_offset
            .map(|_| parse_ma_config(&mut reader, MaTreeLimits::default()).unwrap());
        let SampleBitDepth::Integer { bits_per_sample } = image.bit_depth else {
            panic!("integer");
        };
        let topology = ModularChannelTopology::full_resolution(
            frame.width,
            frame.height,
            bits_per_sample,
            image.extra_channel_count,
            ModularTransformLimits::default(),
        )
        .unwrap();
        let plan = ModularSideImagePlan::parse(
            &mut reader,
            topology,
            bits_per_sample,
            0,
            global_ma.as_ref(),
        )
        .unwrap();
        assert_eq!(plan.final_planes.len(), image.extra_channels.len());
        assert!(
            plan.channel_metadata
                .channels
                .iter()
                .enumerate()
                .all(|(index, c)| index < plan.meta_channel_count
                    || (c.width <= 256 && c.height <= 256))
        );
        if name == "transformed" {
            assert!(!plan.inverse_plan.jobs().is_empty());
        }
        let mut codestream = parsed.codestream().to_vec();
        codestream.resize(codestream.len().next_multiple_of(4), 0);
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VarDCT extra substream codestream"),
            contents: &codestream,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let bytes = pipeline.memory_bytes(&plan, end).unwrap();
        let permit = backend
            .transient_memory_budget()
            .try_reserve(bytes)
            .unwrap();
        let mut recording = pipeline.record(&backend, &input, &plan, end).unwrap();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VarDCT extra substream oracle readback"),
            size: plan.inverse_plan.arena_bytes(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        recording.encoder.copy_buffer_to_buffer(
            &recording.job.arena,
            0,
            &staging,
            0,
            staging.size(),
        );
        let mut job = recording.finish();
        assert_eq!(job.memory_bytes(), bytes);
        let submission = backend.queue().submit([job.take_commands().unwrap()]);
        let (tx, rx) = std::sync::mpsc::channel();
        job.mark_status_mapped();
        job.status_staging()
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
        let (tx, pixels_rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        pixels_rx.recv().unwrap().unwrap();
        let status = job.finish_status().unwrap();
        assert_eq!(
            status.code,
            crate::wgpu_engine::types::STATUS_OK,
            "{name}: {status:?}"
        );
        assert_eq!(status.decoded_samples, plan.decoded_words);
        eprintln!(
            "{name}: token={} end={} samples={} meta={}",
            plan.token_bit_offset, status.cursor, status.decoded_samples, plan.meta_channel_count
        );
        assert_eq!(status.expected_cursor, end);
        assert!((plan.token_bit_offset..end).contains(&status.cursor));
        // The side stream has no byte padding before the following LF image header.
        let mut suffix = BitReader::new(parsed.codestream());
        suffix.skip_bits(u64::from(status.cursor)).unwrap();
        let lf = parse_lf_group_header_reader(&mut suffix, u64::from(end)).unwrap();
        assert!(lf.modular.tree_or_token_bit_offset > u64::from(status.cursor));
        let reference = oracle::libjxl_planes(
            &encoded,
            (frame.width * frame.height) as usize,
            plan.final_planes.len(),
        );
        let rust_reference = oracle::rust_planes(&encoded);
        let mapped = staging.slice(..).get_mapped_range().unwrap();
        let words = bytemuck::cast_slice::<u8, i32>(&mapped);
        for (index, plane) in plan.final_planes.iter().enumerate() {
            let SampleBitDepth::Integer {
                bits_per_sample: depth,
            } = image.extra_channels[index].bit_depth
            else {
                panic!("integer");
            };
            let mask = (1u32 << depth) - 1;
            for y in 0..frame.height {
                for x in 0..frame.width {
                    let c = if image.grayscale { 1 } else { 3 } + index as u32;
                    let expected = match x % 11 {
                        0 => 0,
                        1 => mask,
                        _ => (193 * x + 317 * y + 97 * c + (x ^ y) * (23 + c)) & mask,
                    };
                    let value =
                        words[(plane.word_offset + y * plane.row_stride_words + x) as usize];
                    assert_eq!(value, expected as i32, "{name} channel {index} at {x},{y}");
                    for (_, extras) in std::iter::once(&rust_reference).chain(reference.iter()) {
                        let w = frame.width;
                        let h = frame.height;
                        let (ox, oy) = match image.orientation {
                            1 => (x, y),
                            2 => (w - 1 - x, y),
                            3 => (w - 1 - x, h - 1 - y),
                            4 => (x, h - 1 - y),
                            5 => (y, x),
                            6 => (h - 1 - y, x),
                            7 => (h - 1 - y, w - 1 - x),
                            8 => (y, w - 1 - x),
                            _ => unreachable!(),
                        };
                        let stride = if image.orientation >= 5 { h } else { w };
                        assert!(
                            (value as f32 / mask as f32
                                - extras[index][(oy * stride + ox) as usize])
                                .abs()
                                < 2e-7,
                            "{name} reference channel {index}"
                        );
                    }
                }
            }
        }
        drop(mapped);
        staging.unmap();
        drop(job);
        drop(permit);
        assert_eq!(
            backend.transient_memory_budget().snapshot().reserved_bytes,
            0
        );
    }
}

#[test]
fn side_image_stream_window_is_word_aligned_and_cursor_rebased() {
    let window = stream_window_geometry(35, 100).unwrap();
    assert_eq!(window.source_offset, 4);
    assert_eq!(window.bytes, 12);
    assert_eq!(window.cursor_base_bits, 32);
    assert_eq!(window.token_start, 3);
    assert_eq!(window.token_end, 68);
    assert!(stream_window_geometry(100, 100).is_err());
    assert!(stream_window_geometry(101, 100).is_err());
}
