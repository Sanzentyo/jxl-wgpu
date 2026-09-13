use super::*;
use wgpu::util::DeviceExt;

#[test]
fn native_scalar_packing_preserves_gray_rgb_and_extra_ieee_words() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let memory = backend.transient_memory_budget();
    let words: [u32; 18] = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x8000_0001,
        0x807f_ffff,
        0x0080_0000,
        0x8080_0000,
        0x3e80_0000,
        0xc000_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0123,
        0xffc4_5678,
        0x7f80_0042,
        0xff80_0246,
    ];
    let extent = Extent2d::new(3, 2);
    for case in jxl_test_support::fixtures::embedded_icc::cases()
        .filter(|case| !case.xyb && case.encoding == jxl_gpu_bitstream::FrameEncoding::Modular)
    {
        let bytes = case.bytes();
        let mut image = jxl_gpu_bitstream::parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap()
            .image_header;
        image.width = extent.width;
        image.height = extent.height;
        image.bit_depth = SampleBitDepth::Float {
            bits_per_sample: 32,
            exponent_bits_per_sample: 8,
        };
        image.extra_channels = [
            ExtraChannelTypeInventory::Alpha { associated: false },
            ExtraChannelTypeInventory::Depth,
        ]
        .into_iter()
        .map(|channel_type| ExtraChannelInventory {
            channel_type,
            bit_depth: image.bit_depth,
            dimension_shift: 0,
            name_bytes: Vec::new(),
        })
        .collect();
        image.extra_channel_count = 2;
        let encoding = crate::image_color::original_domain(&image).unwrap();
        let layout = FrameSurfaceLayout::with_encoding(
            extent,
            2,
            encoding.clone(),
            &backend.device().limits(),
        )
        .unwrap();
        let color_count = layout.color.planes.len();
        let stride = layout.color_plane_bytes as usize / 4;
        // Poison the alignment gaps independently of all valid color/extra samples.
        let mut storage = vec![0xdeaf_beef_u32; layout.storage_bytes as usize / 4];
        for channel in 0..color_count + 2 {
            for pixel in 0..6 {
                storage[channel * stride + pixel] = words[(channel * 6 + pixel) % words.len()];
            }
        }
        let source = Surface {
            buffer: GpuBufferLease::from_tracked(
                backend
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("native scalar IEEE word source"),
                        contents: bytemuck::cast_slice(&storage),
                        usage: wgpu::BufferUsages::STORAGE,
                    }),
                memory.try_reserve(layout.storage_bytes).unwrap(),
            ),
            layout: Arc::new(layout),
            encoding,
        };
        for (exif, pixels) in [
            (1, [0, 1, 2, 3, 4, 5]),
            (2, [2, 1, 0, 5, 4, 3]),
            (3, [5, 4, 3, 2, 1, 0]),
            (4, [3, 4, 5, 0, 1, 2]),
            (5, [0, 3, 1, 4, 2, 5]),
            (6, [3, 0, 4, 1, 5, 2]),
            (7, [5, 2, 4, 1, 3, 0]),
            (8, [2, 5, 1, 4, 0, 3]),
        ] {
            image.orientation = exif;
            for channel in 0..color_count + 2 {
                let request = GpuOutputRequest::numeric(
                    jxl_gpu_formats::vpi::VpiPitchLinearFormat::F32.pixel_format(),
                    crate::NumericSampleMapping::NativeFloat,
                )
                .unwrap()
                .with_alpha_output_policy(crate::AlphaOutputPolicy::Associated);
                let request = if channel < color_count {
                    request.with_color_channel(channel as u32)
                } else {
                    request.with_extra_channel((channel - color_count) as u32)
                }
                .unwrap();
                let compositor = Compositor::new(
                    backend.clone(),
                    extent,
                    &image,
                    &request,
                    ColorUsage::ORIGINAL,
                )
                .unwrap();
                let frame = jxl_wgpu::GpuImageFrame {
                    token: jxl_gpu_protocol::SubmissionToken(1),
                    outputs: vec![GpuImageOutput {
                        id: jxl_gpu_protocol::OutputId(0),
                        layout: compositor.layout.clone(),
                        buffer: compositor.pack(&source).unwrap().wait().unwrap(),
                    }],
                    changed: Default::default(),
                };
                let actual = jxl_wgpu::ImageReadbackPipeline::new(&backend)
                    .submit(&frame)
                    .unwrap()
                    .wait()
                    .unwrap()
                    .frame;
                let expected: Vec<_> = pixels
                    .into_iter()
                    .flat_map(|pixel| words[(channel * 6 + pixel) % words.len()].to_le_bytes())
                    .collect();
                assert_eq!(
                    actual.outputs[0].bytes, expected,
                    "Gray={} orientation={exif} channel={channel}",
                    image.grayscale
                );
            }
        }
        drop(source);
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}

#[test]
fn spot_metadata_admission_is_exact_retryable_and_completion_owned() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let hex = include_str!("../../../../test-data/extras_spots_thin.jxl.hex");
    let compact = hex.split_whitespace().collect::<String>();
    let data = compact
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
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
        let request = GpuOutputRequest::color(
            FrameSurfaceEncoding::Rgb(jxl_gpu_protocol::RgbColorEncoding::SRGB_BT709).format(),
        )
        .unwrap()
        .with_spot_color_policy(policy);
        let compositor = Compositor::new(
            backend.clone(),
            Extent2d::new(image.width, image.height),
            image,
            &request,
            ColorUsage::ORIGINAL,
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
        let transient = 208
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
