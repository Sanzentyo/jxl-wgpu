use super::super::{ColorUsage, Compositor};
use super::*;
use jxl_gpu_protocol::Extent2d;

mod device;

#[test]
fn icc_program_admission_is_exact_reusable_retryable_and_completion_owned() {
    admission(false);
    admission(true);
}

fn admission(tone_mapping: bool) {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let memory = backend.transient_memory_budget();
    let image_header = |bytes: &[u8]| {
        jxl_gpu_bitstream::parse(bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap()
            .image_header
    };
    let matrix = jxl_test_support::fixtures::embedded_icc::cases()
        .filter(|case| !case.xyb && case.encoding == jxl_gpu_bitstream::FrameEncoding::Modular)
        .map(|case| case.bytes());
    let black = ["lut8_xyz_1", "lut16_lab_3", "low_1", "bright_3"].map(|name| {
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../jxl_wgpu/test-data/icc/black/decoder")
                .join(format!("{name}_modular.jxl")),
        )
        .unwrap()
    });
    let enumerated = jxl_test_support::fixtures::original_color::cases()
        .into_iter()
        .chain(jxl_test_support::fixtures::original_color::analytic_cases())
        .filter(|case| {
            !case.sequence
                && !case.mode.xyb()
                && !case.mode.ycbcr()
                && case.mode.encoding() == jxl_gpu_bitstream::FrameEncoding::Modular
        })
        .map(|case| (case.bytes(), true));
    let target = jxl_gpu_protocol::icc::IccProfile::parse(
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../jxl_wgpu/test-data/icc/gray.icc"),
        )
        .unwrap()
        .into(),
        Default::default(),
    )
    .unwrap();
    let mut dynamic = 0;
    let mut rgb_transfers = 0;
    let mut spot_programs = 0;
    let spots = jxl_test_support::fixtures::icc_spots::cases()
        .into_iter()
        .filter(|case| case.original && case.modular && !case.sequence)
        .map(|case| (image_header(&case.bytes()), !case.icc))
        .collect::<Vec<_>>();
    let ink = spots[0]
        .0
        .extra_channels
        .iter()
        .find(|extra| {
            matches!(
                extra.channel_type,
                jxl_gpu_bitstream::ExtraChannelTypeInventory::SpotColour { .. }
            )
        })
        .unwrap();
    // These compositor-only headers combine a real ink declaration with each dynamic
    // black-validation profile, covering the map callback's ownership of spot resources.
    let dynamic_spots = black
        .iter()
        .map(|bytes| {
            let mut image = image_header(bytes);
            image.extra_channels.push(ink.clone());
            image.extra_channel_count += 1;
            (image, false)
        })
        .collect::<Vec<_>>();
    for (image, to_icc) in matrix
        .chain(black)
        .map(|bytes| (bytes, false))
        .chain(enumerated)
        .map(|(bytes, to_icc)| (image_header(&bytes), to_icc))
        .chain(spots)
        .chain(dynamic_spots)
    {
        let ink_count = image
            .extra_channels
            .iter()
            .filter(|extra| {
                matches!(
                    extra.channel_type,
                    jxl_gpu_bitstream::ExtraChannelTypeInventory::SpotColour { .. }
                )
            })
            .count() as u64;
        let request = GpuOutputRequest::color(
            if to_icc {
                FrameSurfaceEncoding::Icc(target.clone())
            } else {
                FrameSurfaceEncoding::Rgb(jxl_gpu_protocol::RgbColorEncoding::SRGB_BT709)
            }
            .format(),
        )
        .unwrap()
        .with_alpha_output_policy(crate::AlphaOutputPolicy::Preserve)
        .with_icc_rendering_intent(jxl_gpu_protocol::icc::IccRenderingIntent::Perceptual);
        let request = if tone_mapping {
            request.with_tone_mapping(jxl_gpu_protocol::LuminanceRange::new(0.0, 80.0).unwrap())
        } else {
            request
        };
        let compositor = Compositor::new(
            backend.clone(),
            Extent2d::new(image.width, image.height),
            &image,
            &request,
            ColorUsage::ORIGINAL,
        )
        .unwrap();
        let super::super::Packing::Icc(presentations) = &compositor.packing else {
            panic!("ICC presentation plan")
        };
        assert_eq!(presentations.len(), 1);
        let presentation = &presentations[0];
        let transform = presentation.transform.as_ref().unwrap();
        rgb_transfers += usize::from(matches!(presentation.source_encoding,
            FrameSurfaceEncoding::Rgb(encoding) if encoding.transfer != jxl_gpu_protocol::TransferFunction::Linear));
        dynamic += usize::from(transform.memory.validation_bytes == 4);
        spot_programs += usize::from(presentation.spots.is_some());
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
        let transient = 288
            + transform.memory.transient_bytes()
            + presentation.working.storage_bytes
            + if presentation.spots.is_some() {
                size + ink_count * 32
            } else {
                0
            };
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
        // Concurrent submissions share immutable metadata, but own their status and dispatch
        // allocations. Completing in reverse consumer order must release both reservations.
        let first = compositor.pack(&source).unwrap();
        let second = compositor.pack(&source).unwrap();
        drop(second.wait().unwrap());
        drop(first.wait().unwrap());
        assert_eq!(memory.snapshot().reserved_bytes, size + program_bytes);
        // The next dispatch fits without reserving another program. Abandon its work and
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
    assert_eq!(dynamic, 8);
    assert!(rgb_transfers >= 10);
    assert_eq!(spot_programs, 8);
}
