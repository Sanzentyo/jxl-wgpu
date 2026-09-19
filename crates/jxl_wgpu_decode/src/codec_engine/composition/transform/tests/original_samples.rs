use super::*;
use jxl_gpu_formats::{Channel, PixelFormat, SampleKind};
use jxl_gpu_protocol::icc::IccError;

fn image(gray: bool) -> (ImageHeaderInventory, FrameInventory) {
    let case = jxl_test_support::fixtures::embedded_icc::cases()
        .find(|case| !case.xyb && case.gray == gray)
        .unwrap();
    let data = case.bytes();
    let mut inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let image = &mut inventory.image_header;
    image.width = 3;
    image.height = 2;
    // Retain opaque profile metadata without granting it ICC color authority.
    let icc = image.embedded_icc.as_mut().unwrap();
    Arc::make_mut(&mut icc.profile)[76..80].copy_from_slice(&0xd32b_u32.to_be_bytes());
    assert!(matches!(
        crate::image_color::original_domain(image),
        Err(Error::Icc(IccError::Invalid {
            field: "PCS D50 illuminant",
            offset: 68
        }))
    ));
    (inventory.image_header, inventory.frames.remove(0))
}

fn compositor(backend: &WgpuBackend, image: &ImageHeaderInventory) -> Compositor {
    let request = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
        crate::NumericSampleMapping::NativeFloat,
    )
    .unwrap()
    .with_color_channel(0)
    .unwrap();
    Compositor::new(
        backend.clone(),
        jxl_gpu_protocol::Extent2d::new(image.width, image.height),
        image,
        &request,
        ColorUsage::ORIGINAL,
    )
    .unwrap()
}

fn words(backend: &WgpuBackend, surface: &Surface) -> Vec<u32> {
    jxl_test_support::gpu::planes::read(
        backend,
        &jxl_wgpu::GpuImageOutput {
            id: jxl_gpu_protocol::OutputId(0),
            layout: surface.layout.layouts().last().unwrap().clone(),
            buffer: surface.buffer.clone(),
        },
    )
}

#[test]
fn raw_original_copy_admits_exact_budget_and_retains_words_through_cancellation() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let memory = backend.transient_memory_budget();
    let patterns = [
        0x8000_0000,
        1,
        0x807f_ffff,
        0x7f80_0000,
        0xffc4_5678,
        0x7f80_0042,
    ];
    for gray in [false, true] {
        let (image, mut frame) = image(gray);
        frame.do_ycbcr = false;
        let metadata = image.embedded_icc.clone();
        let compositor = compositor(&backend, &image);
        let encoding = compositor.original.clone();
        assert_eq!(
            encoding,
            FrameSurfaceEncoding::OriginalSamples { grayscale: gray }
        );
        assert!(encoding.icc_profile().is_none() && encoding.rgb_encoding().is_none());
        assert!(FrameSurfaceEncoding::from_format(&encoding.format()).is_none());
        let source = source(&backend, &image);
        let mut storage = vec![0xdead_beef_u32; source.layout.storage_bytes as usize / 4];
        for layout in source.layout.layouts() {
            for plane in &layout.planes {
                let offset = plane.offset as usize / 4;
                storage[offset..offset + patterns.len()].copy_from_slice(&patterns);
            }
        }
        backend.queue().write_buffer(
            source.buffer.as_wgpu_buffer(),
            0,
            bytemuck::cast_slice(&storage),
        );
        let source_bytes = source.layout.storage_bytes;
        let output_bytes = FrameSurfaceLayout::with_encoding(
            source.extent(),
            image.extra_channels.len(),
            encoding.clone(),
            &backend.device().limits(),
        )
        .unwrap()
        .storage_bytes;
        let transient = completion_fence_bytes();
        let convert = || {
            super::super::convert(
                &backend,
                &source,
                &image,
                &frame,
                &compositor,
                encoding.clone(),
            )
        };
        let mut shortages = vec![(output_bytes - 1, output_bytes)];
        if transient != 0 {
            shortages.push((output_bytes + transient - 1, transient));
        }
        for (available, requested) in shortages {
            let held = memory
                .try_reserve(memory.snapshot().available_bytes - available)
                .unwrap();
            let before = memory.snapshot().reserved_bytes;
            for _ in 0..2 {
                assert!(matches!(convert(), Err(Error::MemoryBackpressure(
                    jxl_wgpu::MemoryBudgetError::Exhausted { requested_bytes, .. }
                )) if requested_bytes == requested));
                assert_eq!(memory.snapshot().reserved_bytes, before);
            }
            drop(held);
        }
        let held = memory
            .try_reserve(memory.snapshot().available_bytes - output_bytes - transient)
            .unwrap();
        let output = convert().unwrap().wait().unwrap();
        drop(held);
        assert_eq!(
            memory.snapshot().reserved_bytes,
            source_bytes + output_bytes
        );
        let actual = words(&backend, &output);
        for layout in output.layout.layouts() {
            for plane in &layout.planes {
                let offset = plane.offset as usize / 4;
                assert_eq!(&actual[offset..offset + patterns.len()], &patterns);
            }
        }
        // Domain confusion is a typed rejection, even though the non-color formats can match.
        assert!(matches!(
            compositor.pack(&source),
            Err(Error::EngineContract(_))
        ));
        let mut wrong = output.clone();
        wrong.encoding = FrameSurfaceEncoding::Encoded;
        assert!(matches!(
            compositor.pack(&wrong),
            Err(Error::EngineContract(_))
        ));
        drop(wrong);
        let work = convert().unwrap();
        drop(work);
        drop(source);
        drop(compositor);
        drain(&backend, output_bytes);
        assert_eq!(words(&backend, &output), actual);
        assert_eq!(image.embedded_icc, metadata);
        drop(output);
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}

#[test]
fn raw_gray_and_rgb_ycbcr_reconstruction_keeps_original_components() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for gray in [false, true] {
        let (image, mut frame) = image(gray);
        frame.do_ycbcr = true;
        let compositor = compositor(&backend, &image);
        let source = source(&backend, &image);
        let mut storage = vec![0_u32; source.layout.storage_bytes as usize / 4];
        for (plane, value) in source
            .layout
            .color
            .planes
            .iter()
            .zip([0.125_f32, 0.25, -0.0625])
        {
            let offset = plane.offset as usize / 4;
            storage[offset..offset + 6].fill(value.to_bits());
        }
        backend.queue().write_buffer(
            source.buffer.as_wgpu_buffer(),
            0,
            bytemuck::cast_slice(&storage),
        );
        let output = super::super::convert(
            &backend,
            &source,
            &image,
            &frame,
            &compositor,
            compositor.original.clone(),
        )
        .unwrap()
        .wait()
        .unwrap();
        let actual = words(&backend, &output);
        // Independent JPEG YCbCr inverse; a one-component output retains the first component.
        let y = 0.25 + 128.0 / 255.0;
        let expected = [
            y - 1.402 * 0.0625,
            y - (0.114 * 1.772 / 0.587) * 0.125 + (0.299 * 1.402 / 0.587) * 0.0625,
            y + 1.772 * 0.125,
        ];
        assert_eq!(output.layout.color.planes.len(), if gray { 1 } else { 3 });
        for (plane, expected) in output.layout.color.planes.iter().zip(expected) {
            let offset = plane.offset as usize / 4;
            for &word in &actual[offset..offset + 6] {
                assert!((f64::from(f32::from_bits(word)) - expected).abs() <= 2e-7);
            }
        }
        drop((output, source, compositor));
        assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
    }
}
