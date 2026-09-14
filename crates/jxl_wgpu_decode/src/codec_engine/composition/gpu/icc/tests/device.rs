use super::*;
use crate::codec_engine::composition::{gpu::Packing, lock};
use jxl_gpu_formats::{ColorSample, ColorStorage, PixelFormat};

#[test]
fn device_output_admits_exact_bytes_retries_and_retains_cancelled_resources() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = jxl_gpu_protocol::icc::IccProfile::parse(
        std::fs::read(root.join("../jxl_wgpu/test-data/icc/lut/ab_channels15.icc"))
            .unwrap()
            .into(),
        Default::default(),
    )
    .unwrap();
    let mut prepared = 0;
    for (path, same) in [
        ("test-data/cmyk/generated/lut8_lab_4_0.jxl", true),
        ("test-data/cmyk/generated/ab_xyz_4_1.jxl", true),
        ("test-data/cmyk/generated/lut16_xyz_4_2.jxl", false),
        (
            "../jxl_wgpu/test-data/icc/black/decoder/lut16_lab_3_modular.jxl",
            false,
        ),
    ] {
        let bytes = std::fs::read(root.join(path)).unwrap();
        let image = jxl_gpu_bitstream::parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap()
            .image_header;
        let profile = if same {
            jxl_gpu_protocol::icc::IccProfile::parse(
                image.embedded_icc.as_ref().unwrap().profile.clone(),
                Default::default(),
            )
            .unwrap()
        } else {
            target.clone()
        };
        let format =
            PixelFormat::icc_device(profile, ColorSample::F32, ColorStorage::Planar, true).unwrap();
        let request = GpuOutputRequest::color(format)
            .unwrap()
            .with_icc_rendering_intent(jxl_gpu_protocol::icc::IccRenderingIntent::Perceptual);
        let compositor = Compositor::new(
            backend.clone(),
            Extent2d::new(image.width, image.height),
            &image,
            &request,
            ColorUsage::ORIGINAL,
        )
        .unwrap();
        let Packing::Icc(presentations) = &compositor.packing else {
            panic!("ICC presentation")
        };
        assert_eq!(presentations.len(), 1);
        let presentation = &presentations[0];
        assert_eq!(presentation.params.len(), 320);
        assert_eq!(presentation.transform.is_none(), same);
        if same {
            assert_eq!(
                presentation.working.storage_bytes,
                compositor.surface.storage_bytes
            );
        } else {
            assert_eq!(presentation.working.color.planes.len(), 15);
        }
        prepared += usize::from(
            presentation
                .transform
                .as_ref()
                .is_some_and(|t| t.memory.validation_bytes != 0),
        );
        exercise(&backend, compositor);
    }
    assert_eq!(prepared, 1);
}

fn exercise(backend: &WgpuBackend, compositor: Compositor) {
    let Packing::Icc(presentations) = &compositor.packing else {
        unreachable!()
    };
    let presentation = &presentations[0];
    let memory = backend.transient_memory_budget();
    assert_eq!(memory.snapshot().reserved_bytes, 0);
    let size = compositor.surface.storage_bytes;
    let source = compositor.completed_surface(GpuBufferLease::from_tracked(
        backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("device output admission source"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        memory.try_reserve(size).unwrap(),
    ));
    let output_size = aligned(compositor.layout.logical_size).unwrap();
    let program = presentation
        .transform
        .as_ref()
        .map_or(0, |t| t.memory.program_bytes);
    let transient = 320
        + completion_fence_bytes()
        + presentation
            .spots
            .as_ref()
            .map_or(0, Rendering::memory_bytes)
        + presentation.transform.as_ref().map_or(0, |t| {
            presentation.working.storage_bytes + t.memory.transient_bytes()
        });
    let mut limits = vec![
        (output_size - 1, output_size),
        (output_size + transient - 1, transient),
    ];
    if program != 0 {
        limits.push((output_size + transient + program - 1, program));
    }
    for (available, requested) in limits {
        let held = memory
            .try_reserve(memory.snapshot().available_bytes - available)
            .unwrap();
        let before = memory.snapshot().reserved_bytes;
        for _ in 0..2 {
            assert!(
                matches!(compositor.pack(&source), Err(Error::MemoryBackpressure(jxl_wgpu::MemoryBudgetError::Exhausted { requested_bytes, .. })) if requested_bytes == requested)
            );
            assert_eq!(memory.snapshot().reserved_bytes, before);
            if let Some(t) = &presentation.transform {
                assert!(lock(&t.uploaded).is_none());
            }
        }
        drop(held);
    }
    let held = memory
        .try_reserve(memory.snapshot().available_bytes - output_size - transient - program)
        .unwrap();
    let first = compositor.pack(&source).unwrap().wait().unwrap();
    drop(held);
    assert_eq!(
        memory.snapshot().reserved_bytes,
        size + program + output_size
    );
    drop(first);
    let first = compositor.pack(&source).unwrap();
    let second = compositor.pack(&source).unwrap();
    drop(second.wait().unwrap());
    drop(first.wait().unwrap());
    assert_eq!(memory.snapshot().reserved_bytes, size + program);
    let held = memory
        .try_reserve(memory.snapshot().available_bytes - output_size - transient)
        .unwrap();
    let pending = compositor.pack(&source).unwrap();
    let output = pending.unvalidated().unwrap();
    drop(pending);
    drop(held);
    drop(source);
    drop(compositor);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while memory.snapshot().reserved_bytes != output_size && std::time::Instant::now() < deadline {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(memory.snapshot().reserved_bytes, output_size);
    drop(output);
    assert_eq!(memory.snapshot().reserved_bytes, 0);
}
