use super::*;
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction};
use jxl_gpu_protocol::RgbColorSpace;
use jxl_test_support::gpu::planes;
use wgpu::util::DeviceExt;

fn input(
    backend: &WgpuBackend,
    format: PixelFormat,
    extent: Extent2d,
    values: &[f32],
) -> GpuImageFrame {
    let layout = ImageLayout::packed(extent, format).unwrap();
    let bytes = bytemuck::cast_slice(values);
    let buffer = GpuBufferLease::from_tracked(
        backend
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gain-map lifetime test input"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            }),
        backend
            .transient_memory_budget()
            .try_reserve(bytes.len() as u64)
            .unwrap(),
    );
    GpuImageFrame {
        token: SubmissionToken(0),
        outputs: vec![GpuImageOutput {
            id: OutputId(0),
            layout,
            buffer,
        }],
        changed: Default::default(),
    }
}

#[test]
fn completed_cancelled_and_rejected_gain_submissions_release_all_reservations() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let memory = backend.transient_memory_budget();
    let raw = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test-data/gain_map/generated/case_0.jxl"),
    )
    .unwrap();
    let mut header = jxl_gpu_bitstream::parse(&raw, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .image_header;
    header.width = 2;
    header.height = 2;
    let ColorSpecification::Defined(mut fields) = crate::vardct_rgb8_format().color_spec else {
        unreachable!()
    };
    fields.transfer = TransferFunction::Linear;
    let color = ColorSpecification::Defined(fields);
    let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        color.clone(),
    ))
    .unwrap();
    let plan = Plan::new(
        &backend,
        &request,
        &header,
        RgbColorEncoding {
            space: RgbColorSpace::Bt709,
            transfer: jxl_gpu_protocol::TransferFunction::Linear,
        },
        &GainMapMetadata::default(),
        1.0,
        DisplayIntensity::new(203.0).unwrap(),
    )
    .unwrap();
    for cancel in [false, true] {
        let mut samples = [0.25_f32; 16];
        samples[12..].fill(0.5);
        let base = input(
            &backend,
            PixelFormat::rgb_f32(RgbChannelOrder::Rgba, true, color.clone()),
            Extent2d::new(2, 2),
            &samples,
        );
        let map = input(
            &backend,
            PixelFormat::gray_f32(false, true, color.clone()),
            Extent2d::new(3, 1),
            &[0.5; 3],
        );
        let resident = memory.snapshot().reserved_bytes;
        let output = plan.output_plan.memory.output_storage_bytes;
        let transient = (size_of::<Params>() + size_of::<ImageOutputParams>()) as u64
            + completion_fence_bytes();
        for (remaining, needed) in [(output - 1, output), (output + transient - 1, transient)] {
            let blocker = memory
                .try_reserve(memory.snapshot().available_bytes - remaining)
                .unwrap();
            let before = memory.snapshot().reserved_bytes;
            assert!(matches!(plan.submit(&backend, &base, &map),
                Err(crate::Error::MemoryBackpressure(jxl_wgpu::MemoryBudgetError::Exhausted { requested_bytes, .. })) if requested_bytes == needed));
            assert_eq!(memory.snapshot().reserved_bytes, before);
            drop(blocker);
            assert_eq!(memory.snapshot().reserved_bytes, resident);
        }
        let work = plan.submit(&backend, &base, &map).unwrap();
        drop((base, map));
        if cancel {
            // Drop the consumer of already-submitted work without polling/waiting for it.
            // Completion owns every input, output and uniform reservation independently.
            drop(work);
        } else {
            let frame = plan.frame(SubmissionToken(0), work.wait().unwrap());
            assert_eq!(memory.snapshot().reserved_bytes, output);
            for pixel in planes::read(&backend, &frame.outputs[0]).as_chunks::<4>().0 {
                for word in &pixel[..3] {
                    assert!(
                        (f64::from(f32::from_bits(*word)) - 0.25 * 2.0_f64.sqrt()).abs() < 2e-6
                    );
                }
                assert_eq!(f32::from_bits(pixel[3]), 0.5);
            }
            drop(frame);
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while memory.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}
