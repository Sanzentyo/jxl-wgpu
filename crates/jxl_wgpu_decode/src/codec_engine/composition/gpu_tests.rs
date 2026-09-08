use super::*;

#[test]
fn spot_metadata_admission_is_exact_retryable_and_completion_owned() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let hex = include_str!("../../../test-data/extras_spots_thin.jxl.hex");
    let compact = hex.split_whitespace().collect::<String>();
    let data = compact
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect::<Vec<_>>();
    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let image = &inventory.image_header;
    let memory = backend.transient_memory_budget();
    for policy in [
        crate::SpotColorPolicy::Render,
        crate::SpotColorPolicy::Preserve,
    ] {
        let request = GpuOutputRequest::color(FrameSurfaceEncoding::Srgb.format())
            .unwrap()
            .with_spot_color_policy(policy);
        let compositor = Compositor::new(
            backend.clone(),
            Extent2d::new(image.width, image.height),
            &image.extra_channels,
            false,
            image.bit_depth,
            OutputOrientation::from_exif_value(1).unwrap(),
            &request,
        )
        .unwrap();
        let size = compositor.surface.storage_bytes;
        let buffer = GpuBufferLease::from_tracked(
            backend.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("spot accounting test surface"),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            memory.try_reserve(size).unwrap(),
        );
        let source = compositor.completed_surface(buffer);
        let output_size = aligned(compositor.layout.logical_size).unwrap();
        let transient = 192
            + if policy == crate::SpotColorPolicy::Render {
                5 * 32
            } else {
                0
            };
        let held = memory
            .try_reserve(memory.snapshot().available_bytes - (output_size + transient - 1))
            .unwrap();
        let reserved = memory.snapshot().reserved_bytes;
        for _ in 0..2 {
            assert!(
                matches!(compositor.pack(&source), Err(Error::MemoryBackpressure(
                jxl_wgpu::MemoryBudgetError::Exhausted { requested_bytes, .. })) if requested_bytes == transient)
            );
            assert_eq!(
                memory.snapshot().reserved_bytes,
                reserved,
                "output admission rolled back"
            );
        }
        drop(held);
        let work = compositor.pack(&source).unwrap();
        let output = work.unvalidated().unwrap();
        drop(work);
        drop(source);
        drop(compositor);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while memory.snapshot().reserved_bytes != output_size
            && std::time::Instant::now() < deadline
        {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(memory.snapshot().reserved_bytes, output_size);
        drop(output);
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}
