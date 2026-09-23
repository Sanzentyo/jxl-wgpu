use super::*;
use crate::lossless_modular::serializer::parse_group_artifact;
use crate::{
    AnimationHeader, Determinism, EncodeProfile, FrameOptions, GpuEncodeBackend, GpuFrameSource,
    LosslessModularConfig, LosslessModularFormat, LosslessModularGroupSize, LosslessModularLz77,
    LosslessModularPredictor, ProgressivePlan,
};
use wgpu::util::DeviceExt;

#[test]
fn gpu_lz77_artifacts_include_overlapping_and_near_window_limit_matches() {
    let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let encoder = LosslessModularBackend::with_config(
        &context,
        LosslessModularConfig {
            lz77: LosslessModularLz77::Greedy,
            predictor: LosslessModularPredictor::Zero,
            group_size: LosslessModularGroupSize::Pixels1024,
            ..Default::default()
        },
    );
    let count = 1024 * 1024;
    let mut pixels = vec![0u8; count];
    pixels[..7].copy_from_slice(&[1, 3, 7, 15, 31, 63, 127]);
    pixels.copy_within(..7, count - 7);
    let buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("general LZ77 artifact source"),
            contents: &pixels,
            usage: wgpu::BufferUsages::STORAGE,
        });
    let source = crate::BufferImageSource::new(
        Arc::new(buffer),
        jxl_gpu_formats::ImageLayout::packed(
            jxl_gpu_protocol::Extent2d::new(1024, 1024),
            LosslessModularFormat::Gray.pixel_format(8).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let request = FrameEncodeRequest {
        frame_index: FrameIndex::new(0),
        is_last: true,
        profile: EncodeProfile::ModularLossless {
            sample_bit_depth: jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample: 8 },
        },
        progressive: ProgressivePlan::single(),
        minimum_determinism: Determinism::Assembly,
        animation: AnimationHeader::Still,
        canvas_width: 1024,
        canvas_height: 1024,
        options: FrameOptions::default(),
    };
    let mut job = encoder
        .submit(&context, GpuFrameSource::Buffer(source), &request)
        .unwrap();
    let LosslessModularJobState::Resident(resident) = &mut job.state else {
        panic!("one resident group");
    };
    resident.completion.wait().unwrap();
    {
        let lifetime = resident.lifetime.as_ref().unwrap();
        let mapped = lifetime
            .buffer_lease
            .buffers()
            .readback
            .slice(0..resident.output_size)
            .get_mapped_range()
            .unwrap();
        let group = &resident.groups[0];
        let artifact = parse_group_artifact(
            1024,
            1024,
            group.max_events,
            &mapped[..group.output_size as usize],
        )
        .unwrap();
        assert!(artifact.header.event_count < 32);
        assert!(artifact.header.lz77_counts[31] > 0);
        let distances: Vec<_> = artifact
            .events
            .iter()
            .filter(|event| event.kind == 3)
            .map(|event| ((1u32 << event.extra_bit_count) + event.extra_bits) - 119)
            .collect();
        assert!(distances.contains(&1), "{distances:?}");
        assert!(distances.contains(&(count as u32 - 7)), "{distances:?}");
    }
    resident.finish(Ok(())).unwrap();
    drop(job);
    assert_eq!(context.memory_budget().snapshot().reserved_bytes, 0);
}
