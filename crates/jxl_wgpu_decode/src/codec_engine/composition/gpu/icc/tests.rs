use super::super::{ColorUsage, Compositor};
use super::*;
use jxl_gpu_protocol::Extent2d;

#[test]
fn icc_program_admission_is_exact_reusable_retryable_and_completion_owned() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let memory = backend.transient_memory_budget();
    for case in jxl_test_support::fixtures::embedded_icc::cases()
        .filter(|case| !case.xyb && case.encoding == jxl_gpu_bitstream::FrameEncoding::Modular)
    {
        let bytes = case.bytes();
        let image = jxl_gpu_bitstream::parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap()
            .image_header;
        let request = GpuOutputRequest::color(
            FrameSurfaceEncoding::Rgb(jxl_gpu_protocol::RgbColorEncoding::SRGB_BT709).format(),
        )
        .unwrap()
        .with_alpha_output_policy(crate::AlphaOutputPolicy::Preserve);
        let compositor = Compositor::new(
            backend.clone(),
            Extent2d::new(image.width, image.height),
            &image,
            &request,
            ColorUsage::ORIGINAL,
        )
        .unwrap();
        let super::super::Packing::Icc {
            original: Some(presentation),
            ..
        } = &compositor.packing
        else {
            panic!("ICC presentation plan")
        };
        let transform = presentation.transform.as_ref().unwrap();
        assert!(super::super::super::lock(&transform.uploaded).is_none());
        assert_eq!(memory.snapshot().reserved_bytes, 0);
        let size = compositor.surface.storage_bytes;
        let buffer = GpuBufferLease::from_tracked(
            backend.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("ICC admission source"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            memory.try_reserve(size).unwrap(),
        );
        let source = compositor.completed_surface(buffer);
        let output_size = aligned(compositor.layout.logical_size).unwrap();
        let transient =
            208 + transform.memory.dispatch_uniform_bytes + presentation.working.storage_bytes;
        let program_bytes = transform.memory.program_bytes;
        for (available, requested) in [
            (output_size - 1, output_size),
            (output_size + transient - 1, transient),
            (output_size + transient + program_bytes - 1, program_bytes),
        ] {
            let held = memory
                .try_reserve(memory.snapshot().available_bytes - available)
                .unwrap();
            let before = memory.snapshot().reserved_bytes;
            for _ in 0..2 {
                assert!(
                    matches!(compositor.pack(&source), Err(Error::MemoryBackpressure(jxl_wgpu::MemoryBudgetError::Exhausted { requested_bytes, .. })) if requested_bytes == requested)
                );
                assert_eq!(memory.snapshot().reserved_bytes, before);
                assert!(super::super::super::lock(&transform.uploaded).is_none());
            }
            drop(held);
        }
        let held = memory
            .try_reserve(
                memory.snapshot().available_bytes - output_size - transient - program_bytes,
            )
            .unwrap();
        let output = compositor.pack(&source).unwrap().wait().unwrap();
        drop(held);
        assert_eq!(
            memory.snapshot().reserved_bytes,
            size + program_bytes + output_size
        );
        drop(output);
        // The second dispatch fits without reserving another program. Abandon its work and
        // image context while the completion callback keeps every submitted allocation alive.
        let held = memory
            .try_reserve(memory.snapshot().available_bytes - output_size - transient)
            .unwrap();
        let work = compositor.pack(&source).unwrap();
        let output = work.unvalidated().unwrap();
        drop(work);
        drop(held);
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
