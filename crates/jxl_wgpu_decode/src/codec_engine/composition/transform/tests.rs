use super::*;
use crate::GpuOutputRequest;
use crate::codec_engine::composition::gpu::ColorUsage;

fn source(backend: &WgpuBackend, image: &ImageHeaderInventory) -> Surface {
    let layout = FrameSurfaceLayout::with_encoding(
        jxl_gpu_protocol::Extent2d::new(image.width, image.height),
        image.extra_channels.len(),
        FrameSurfaceEncoding::Encoded,
        &backend.device().limits(),
    )
    .unwrap();
    let permit = backend
        .transient_memory_budget()
        .try_reserve(layout.storage_bytes)
        .unwrap();
    let buffer = GpuBufferLease::from_tracked(
        backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("XYB reconstruction admission source"),
            size: layout.storage_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        permit,
    );
    Surface {
        buffer,
        layout: Arc::new(layout),
        encoding: FrameSurfaceEncoding::Encoded,
    }
}

fn drain(backend: &WgpuBackend, expected: u64) {
    let memory = backend.transient_memory_budget();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while memory.snapshot().reserved_bytes != expected && std::time::Instant::now() < deadline {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(memory.snapshot().reserved_bytes, expected);
}

#[test]
fn original_icc_reconstruction_admits_exact_storage_retries_and_retains_cancelled_work() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let memory = backend.transient_memory_budget();
    for case in jxl_test_support::fixtures::embedded_icc::cases().filter(|case| case.xyb) {
        let data = case.bytes();
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let original = crate::image_color::original_domain(image).unwrap();
        let request = GpuOutputRequest::color(original.format()).unwrap();
        let compositor = Compositor::new(
            backend.clone(),
            jxl_gpu_protocol::Extent2d::new(image.width, image.height),
            image,
            &request,
            ColorUsage::ORIGINAL,
        )
        .unwrap();
        let connection = compositor.reconstruction.as_ref().unwrap();
        let source = source(&backend, image);
        let source_bytes = source.layout.storage_bytes;
        let output_bytes = FrameSurfaceLayout::with_encoding(
            source.extent(),
            image.extra_channels.len(),
            original.clone(),
            &backend.device().limits(),
        )
        .unwrap()
        .storage_bytes;
        let linear = FrameSurfaceLayout::with_encoding(
            source.extent(),
            0,
            compositor.linear_encoding(),
            &backend.device().limits(),
        )
        .unwrap();
        let transient = 368
            + completion_fence_bytes()
            + linear.storage_bytes
            + connection.memory.transient_bytes();
        let program_bytes = connection.memory.program_bytes;
        let convert = || {
            super::convert(
                &backend,
                &source,
                image,
                &inventory.frames[0],
                &compositor,
                original.clone(),
            )
        };
        assert!(super::super::lock(&connection.uploaded).is_none());
        for (available, requested) in [
            (output_bytes - 1, output_bytes),
            (output_bytes + transient - 1, transient),
            (output_bytes + transient + program_bytes - 1, program_bytes),
        ] {
            let held = memory
                .try_reserve(memory.snapshot().available_bytes - available)
                .unwrap();
            let before = memory.snapshot().reserved_bytes;
            for _ in 0..2 {
                assert!(matches!(convert(), Err(Error::MemoryBackpressure(
                    jxl_wgpu::MemoryBudgetError::Exhausted { requested_bytes, .. }
                )) if requested_bytes == requested));
                assert_eq!(memory.snapshot().reserved_bytes, before);
                assert!(super::super::lock(&connection.uploaded).is_none());
            }
            drop(held);
        }
        let held = memory
            .try_reserve(
                memory.snapshot().available_bytes - output_bytes - transient - program_bytes,
            )
            .unwrap();
        let output = convert().unwrap().wait().unwrap();
        drop(held);
        assert_eq!(output.encoding, original);
        assert_eq!(
            output.layout.color.planes.len(),
            if case.gray { 1 } else { 3 }
        );
        assert_eq!(
            memory.snapshot().reserved_bytes,
            source_bytes + output_bytes + program_bytes
        );
        let held = memory
            .try_reserve(memory.snapshot().available_bytes - output_bytes - transient)
            .unwrap();
        let work = convert().unwrap();
        drop(work);
        drop(held);
        drop(source);
        drop(compositor);
        drain(&backend, output_bytes);
        drop(output);
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}

#[test]
fn original_reconstruction_and_linear_presentation_share_one_program_admission() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let memory = backend.transient_memory_budget();
    for case in jxl_test_support::fixtures::embedded_icc::cases().filter(|case| case.xyb) {
        let inventory = jxl_gpu_bitstream::parse(&case.bytes(), Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let original = crate::image_color::original_domain(image).unwrap();
        let request = GpuOutputRequest::color(original.format())
            .unwrap()
            .with_alpha_output_policy(crate::AlphaOutputPolicy::Preserve);
        let compositor = Compositor::new(
            backend.clone(),
            jxl_gpu_protocol::Extent2d::new(image.width, image.height),
            image,
            &request,
            ColorUsage {
                original: true,
                linear: true,
                reconstruct_original: true,
            },
        )
        .unwrap();
        let source = source(&backend, image);
        let original_surface = convert(
            &backend,
            &source,
            image,
            &inventory.frames[0],
            &compositor,
            original,
        )
        .unwrap()
        .wait()
        .unwrap();
        let working_bytes = original_surface.layout.storage_bytes;
        drop(original_surface);
        let linear = convert(
            &backend,
            &source,
            image,
            &inventory.frames[0],
            &compositor,
            compositor.linear_encoding(),
        )
        .unwrap()
        .wait()
        .unwrap();
        drop(source);
        let connection = compositor.reconstruction.as_ref().unwrap();
        let program_bytes = connection.memory.program_bytes;
        assert_eq!(
            memory.snapshot().reserved_bytes,
            linear.layout.storage_bytes + program_bytes
        );
        let output_bytes = compositor.layout.logical_size;
        assert_eq!(output_bytes % 4, 0);
        let transient = std::mem::size_of::<jxl_wgpu::ImageOutputParams>() as u64
            + completion_fence_bytes()
            + working_bytes
            + connection.memory.transient_bytes();
        // Leave room only for this dispatch's output and scratch. A second admission of
        // the identical linear-to-original program would fail under this byte budget.
        let held = memory
            .try_reserve(memory.snapshot().available_bytes - output_bytes - transient)
            .unwrap();
        let work = compositor.pack(&linear).unwrap();
        let output = work.unvalidated().unwrap();
        drop(work);
        drop(held);
        drop(linear);
        drop(compositor);
        drain(&backend, output_bytes);
        drop(output);
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}

#[test]
fn direct_xyb_never_selects_an_unused_original_profile_intent() {
    use jxl_gpu_protocol::icc::{IccDirection, IccError, IccProfile, IccRenderingIntent};
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for case in jxl_test_support::fixtures::embedded_icc::cases().filter(|case| case.xyb) {
        let data = case.bytes();
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        for (code, intent) in [
            (0u32, IccRenderingIntent::Perceptual),
            (2, IccRenderingIntent::Saturation),
            (3, IccRenderingIntent::Absolute),
        ] {
            let mut image = inventory.image_header.clone();
            let mut profile = case.profile();
            profile[64..68].copy_from_slice(&code.to_be_bytes());
            let profile = IccProfile::parse(profile.into(), Default::default()).unwrap();
            let profile = jxl_test_support::fixtures::icc::with_nonfinite_matrix_mpe(
                &profile,
                IccDirection::PcsToDevice,
                intent,
            );
            image.embedded_icc.as_mut().unwrap().profile = profile.bytes().clone();
            let linear =
                FrameSurfaceEncoding::Rgb(jxl_gpu_protocol::RgbColorEncoding::LINEAR_BT709);
            let request = GpuOutputRequest::color(linear.format()).unwrap();
            let extent = jxl_gpu_protocol::Extent2d::new(image.width, image.height);
            let compositor = Compositor::new(
                backend.clone(),
                extent,
                &image,
                &request,
                ColorUsage::LINEAR,
            )
            .unwrap();
            assert!(compositor.reconstruction.is_none());
            let source = source(&backend, &image);
            let reconstructed = convert(
                &backend,
                &source,
                &image,
                &inventory.frames[0],
                &compositor,
                linear,
            )
            .unwrap()
            .wait()
            .unwrap();
            drop(compositor.pack(&reconstructed).unwrap().wait().unwrap());
            assert!(matches!(
                Compositor::new(
                    backend.clone(),
                    extent,
                    &image,
                    &request,
                    ColorUsage::ORIGINAL
                ),
                Err(Error::Icc(IccError::Invalid {
                    field: "non-finite float",
                    ..
                }))
            ));
            drop(reconstructed);
            drop(source);
            drop(compositor);
            assert_eq!(
                backend.transient_memory_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}

#[test]
fn linear_icc_gray_surfaces_select_numeric_extras_from_the_actual_plane_layout() {
    use crate::{NumericChannel, NumericSampleMapping};
    use jxl_gpu_formats::{Channel, ImageLayout, PixelFormat, SampleKind};
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let case = jxl_test_support::fixtures::embedded_icc::cases()
        .find(|case| case.gray && case.xyb)
        .unwrap();
    let data = case.bytes();
    let mut image = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .image_header;
    // An unused original CMS intent must not prevent exact extra-channel selection.
    let mut profile = case.profile();
    profile[64..68].copy_from_slice(&0u32.to_be_bytes());
    let profile =
        jxl_gpu_protocol::icc::IccProfile::parse(profile.into(), Default::default()).unwrap();
    let profile = jxl_test_support::fixtures::icc::with_nonfinite_matrix_mpe(
        &profile,
        jxl_gpu_protocol::icc::IccDirection::PcsToDevice,
        jxl_gpu_protocol::icc::IccRenderingIntent::Perceptual,
    );
    image.embedded_icc.as_mut().unwrap().profile = profile.bytes().clone();
    image.extra_channels.push(image.extra_channels[0].clone());
    let extent = jxl_gpu_protocol::Extent2d::new(image.width, image.height);
    let color_request = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
        NumericSampleMapping::NativeFloat,
    )
    .unwrap()
    .with_color_channel(0)
    .unwrap();
    assert!(matches!(
        Compositor::new(
            backend.clone(),
            extent,
            &image,
            &color_request,
            ColorUsage::LINEAR
        ),
        Err(Error::Icc(jxl_gpu_protocol::icc::IccError::Invalid {
            field: "non-finite float",
            ..
        }))
    ));
    let encoding = FrameSurfaceEncoding::Rgb(jxl_gpu_protocol::RgbColorEncoding::LINEAR_BT709);
    let layout =
        FrameSurfaceLayout::with_encoding(extent, 2, encoding.clone(), &backend.device().limits())
            .unwrap();
    let buffer = GpuBufferLease::from_tracked(
        backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gray ICC numeric preview with RGB planes"),
            size: layout.storage_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        backend
            .transient_memory_budget()
            .try_reserve(layout.storage_bytes)
            .unwrap(),
    );
    let values: [_; 2] = std::array::from_fn(|extra| {
        (0..153)
            .map(|pixel| {
                if extra == 0 {
                    (pixel % 17) as f32 / 16.0
                } else {
                    -0.25 - (pixel % 9) as f32 / 8.0
                }
            })
            .collect::<Vec<_>>()
    });
    for (extra, values) in layout.extras.iter().zip(&values) {
        backend.queue().write_buffer(
            buffer.as_wgpu_buffer(),
            extra.planes[0].offset,
            bytemuck::cast_slice(values),
        );
    }
    let source = Surface {
        buffer,
        layout: Arc::new(layout),
        encoding,
    };
    for (extra, expected) in values.iter().enumerate() {
        let request = GpuOutputRequest::numeric(
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
            NumericSampleMapping::NativeFloat,
        )
        .unwrap()
        .with_numeric_channel(NumericChannel::Extra(extra as u32))
        .unwrap();
        let compositor = Compositor::new(
            backend.clone(),
            extent,
            &image,
            &request,
            ColorUsage::LINEAR,
        )
        .unwrap();
        assert!(compositor.reconstruction.is_none());
        let buffer = compositor.pack(&source).unwrap().wait().unwrap();
        let layout = ImageLayout::packed(extent, request.format().clone()).unwrap();
        let outputs = crate::frame_surface::outputs(&layout, None, &buffer);
        let actual = jxl_test_support::gpu::planes::read(&backend, &outputs[0]);
        assert_eq!(
            actual,
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    drop(source);
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
}
