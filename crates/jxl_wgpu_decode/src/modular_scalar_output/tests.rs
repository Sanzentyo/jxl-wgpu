use super::*;

#[test]
fn scalar_output_shader_and_uniform_validate() {
    let module = naga::front::wgsl::parse_str(&shader()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(std::mem::size_of::<ScalarParams>(), 64);
    assert_eq!(std::mem::align_of::<ScalarParams>(), 16);
}

#[test]
fn scalar_plans_reject_mismatched_depth_overflow_and_insufficient_device_limits() {
    let config = ModularScalarOutputConfig {
        extent: Extent2d::new(17, 9),
        orientation: OutputOrientation::from_exif_value(6).unwrap(),
        bits: 12,
        mapping: NumericSampleMapping::NativeUnsigned,
    };
    let layout = ImageLayout::packed(
        config.orientation.map_extent(config.extent),
        crate::model::native_modular_pixel_format(ModularChannels::Gray, 12).unwrap(),
    )
    .unwrap();
    let limits = wgpu::Limits::default();
    let plan = |config, layout: &ImageLayout, limits: &wgpu::Limits| {
        ModularScalarOutputPlan::new(config, layout, limits, KernelVariant::Lanes64)
    };
    assert_eq!(plan(config, &layout, &limits).unwrap().storage_bytes, 308);
    assert!(matches!(
        plan(
            ModularScalarOutputConfig { bits: 8, ..config },
            &layout,
            &limits
        ),
        Err(ModularScalarOutputError::Invalid { .. })
    ));
    let mut bad_layout = layout.clone();
    bad_layout.planes[0].row_stride = u64::MAX;
    assert!(matches!(
        plan(config, &bad_layout, &limits),
        Err(ModularScalarOutputError::Invalid { .. })
    ));
    for bad_limits in [
        wgpu::Limits {
            max_storage_buffers_per_shader_stage: 2,
            ..limits.clone()
        },
        wgpu::Limits {
            max_storage_buffer_binding_size: 304,
            ..limits.clone()
        },
        wgpu::Limits {
            max_compute_workgroups_per_dimension: 1,
            ..limits.clone()
        },
        wgpu::Limits {
            max_uniform_buffer_binding_size: 63,
            ..limits.clone()
        },
    ] {
        assert!(matches!(
            plan(config, &layout, &bad_limits),
            Err(ModularScalarOutputError::Limit { .. })
        ));
    }
}

#[test]
fn resident_scalar_packing_preserves_signed_normalization_orientation_and_guard_bytes() {
    use crate::modular_finalize::{
        ModularFinalizeBindings, ModularFinalizeF64Path, ModularFinalizeOutput,
        ModularFinalizeParams, ModularFinalizePipeline, ModularFinalizeRegion,
    };
    use jxl_gpu_formats::{Channel, PitchLinearPlaneLayout, PixelFormat};
    use jxl_wgpu::WgpuBackend;
    use std::num::NonZeroU64;
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let device = backend.device();
    let variant = KernelVariant::Lanes64;
    let pipeline = ModularScalarOutputPipeline::new(device, variant);
    let finalize = ModularFinalizePipeline::with_variant(
        device,
        variant,
        ModularFinalizeF64Path::ExactF32Widening,
    )
    .unwrap();
    let status = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("validated scalar source status"),
        contents: bytemuck::cast_slice(&[1_u32, 0, 0, 0]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    for extent in [
        Extent2d::new(3, 2),
        Extent2d::new(1, 3),
        Extent2d::new(3, 1),
        Extent2d::new(129, 9),
    ] {
        for bits in [1, 5, 8, 12, 16] {
            let mask = (1_u32 << bits) - 1;
            let values = [-7, 0, mask as i32, mask as i32 + 13, (mask / 2) as i32, -1];
            let plane = GpuModularChannelLayout {
                word_offset: 2,
                row_stride_words: extent.width + 2,
                width: extent.width,
                height: extent.height,
                hshift: 0,
                vshift: 0,
                bit_depth: bits,
                reserved: 0,
            };
            let mut words = vec![
                123456_i32;
                (plane.word_offset + plane.row_stride_words * extent.height)
                    as usize
            ];
            for y in 0..extent.height {
                for x in 0..extent.width {
                    words[(plane.word_offset + y * plane.row_stride_words + x) as usize] =
                        values[((y * extent.width + x) % 6) as usize];
                }
            }
            let arena = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("signed scalar source with offset and stride"),
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::STORAGE,
            });
            for exif in 1..=8 {
                let orientation = OutputOrientation::from_exif_value(exif).unwrap();
                let oriented = orientation.map_extent(extent);
                for floating in [false, true] {
                    let (format, mapping, sample_bytes) = if floating {
                        (
                            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                            NumericSampleMapping::NormalizedUnsigned,
                            4,
                        )
                    } else {
                        (
                            crate::model::native_modular_pixel_format(
                                ModularChannels::Gray,
                                bits as u8,
                            )
                            .unwrap(),
                            NumericSampleMapping::NativeUnsigned,
                            bits.div_ceil(8),
                        )
                    };
                    let stride = u64::from((oriented.width + 2) * sample_bytes);
                    let layout = ImageLayout::from_planes(
                        oriented,
                        format,
                        vec![PitchLinearPlaneLayout {
                            plane_index: 0,
                            offset: 8,
                            row_stride: stride,
                            sample_extent: oriented,
                            row_bytes: u64::from(oriented.width * sample_bytes),
                        }],
                    )
                    .unwrap();
                    let mut limits = device.limits();
                    limits.max_compute_workgroups_per_dimension = 16;
                    let plan = ModularScalarOutputPlan::new(
                        ModularScalarOutputConfig {
                            extent,
                            orientation,
                            bits,
                            mapping,
                        },
                        &layout,
                        &limits,
                        variant,
                    )
                    .unwrap();
                    if extent.width == 129 && floating {
                        assert!(plan.dispatch[1] > 1);
                    }
                    let binding_offset =
                        u64::from(device.limits().min_storage_buffer_offset_alignment).max(4);
                    let buffer_size = binding_offset + plan.storage_bytes + 16;
                    let initial = vec![0x5a; buffer_size as usize];
                    let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("guarded scalar output"),
                        contents: &initial,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });
                    let legacy = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("guarded common Modular finalizer output"),
                        contents: &initial,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });
                    let target = |buffer| ResidentStorageBinding {
                        buffer,
                        offset: binding_offset,
                        size: NonZeroU64::new(plan.storage_bytes).unwrap(),
                    };
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    let scratch = pipeline
                        .encode(
                            device,
                            &mut encoder,
                            plan,
                            plane,
                            ResidentStorageBinding::entire(&arena).unwrap(),
                            target(&output),
                        )
                        .unwrap();
                    let _legacy_uniform = if floating {
                        encoder.clear_buffer(&legacy, binding_offset, Some(plan.storage_bytes));
                        let params = ModularFinalizeParams::new(
                            ModularFinalizeRegion {
                                source_extent: extent,
                                canvas_extent: extent,
                                orientation,
                                origin_x: 0,
                                origin_y: 0,
                                status_index: 0,
                            },
                            bits as u8,
                            &[plane],
                            words.len() as u32,
                            ModularFinalizeOutput {
                                kind: 8,
                                transfer: 0,
                                limited_range: false,
                                channels: 1,
                                order: 0,
                                bits: 32,
                                storage_bits: 32,
                                numeric_mapping: 4,
                                plane_offsets: [8, 0, 0, 0],
                                plane_strides: [stride as u32, 0, 0, 0],
                                logical_size: layout.logical_size as u32,
                                chroma_extent: Extent2d::new(0, 0),
                            },
                        )
                        .unwrap();
                        Some(
                            finalize
                                .encode(
                                    device,
                                    &mut encoder,
                                    ModularFinalizeBindings {
                                        arena: ResidentStorageBinding::entire(&arena).unwrap(),
                                        output_words: target(&legacy),
                                        output_f64: None,
                                        status: ResidentStorageBinding::entire(&status).unwrap(),
                                    },
                                    params,
                                )
                                .unwrap(),
                        )
                    } else {
                        None
                    };
                    let staging = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("scalar test readback"),
                        size: buffer_size * 2 + 20,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });
                    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, buffer_size);
                    encoder.copy_buffer_to_buffer(&legacy, 0, &staging, buffer_size, buffer_size);
                    encoder.copy_buffer_to_buffer(&status, 0, &staging, buffer_size * 2, 16);
                    encoder.copy_buffer_to_buffer(
                        &scratch.status,
                        0,
                        &staging,
                        buffer_size * 2 + 16,
                        4,
                    );
                    let submission = backend.queue().submit([encoder.finish()]);
                    let (tx, rx) = std::sync::mpsc::channel();
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
                    let mut expected = initial;
                    expected
                        [binding_offset as usize..(binding_offset + plan.storage_bytes) as usize]
                        .fill(0);
                    for y in 0..extent.height {
                        for x in 0..extent.width {
                            let [ox, oy] = match exif {
                                1 => [x, y],
                                2 => [extent.width - 1 - x, y],
                                3 => [extent.width - 1 - x, extent.height - 1 - y],
                                4 => [x, extent.height - 1 - y],
                                5 => [y, x],
                                6 => [extent.height - 1 - y, x],
                                7 => [extent.height - 1 - y, extent.width - 1 - x],
                                8 => [y, extent.width - 1 - x],
                                _ => unreachable!(),
                            };
                            let sample = words
                                [(plane.word_offset + y * plane.row_stride_words + x) as usize];
                            let code = if floating {
                                (sample as f32 / mask as f32).to_bits()
                            } else {
                                u32::try_from(sample)
                                    .ok()
                                    .filter(|&v| v <= mask)
                                    .unwrap_or(0)
                            };
                            let offset = (binding_offset
                                + 8
                                + u64::from(oy) * stride
                                + u64::from(ox * sample_bytes))
                                as usize;
                            expected[offset..offset + sample_bytes as usize]
                                .copy_from_slice(&code.to_le_bytes()[..sample_bytes as usize]);
                        }
                    }
                    let mapped = staging.slice(..).get_mapped_range().unwrap();
                    let assert_output = |actual: &[u8]| {
                        let mut rounded = expected.clone();
                        if floating {
                            for y in 0..oriented.height {
                                for x in 0..oriented.width {
                                    let offset = (binding_offset
                                        + 8
                                        + u64::from(y) * stride
                                        + u64::from(x) * 4)
                                        as usize;
                                    let got = u32::from_le_bytes(
                                        actual[offset..offset + 4].try_into().unwrap(),
                                    );
                                    let expected = u32::from_le_bytes(
                                        rounded[offset..offset + 4].try_into().unwrap(),
                                    );
                                    // WGSL division can use a rounded reciprocal on Metal. Check
                                    // two ULPs, while every padding/guard byte remains exact.
                                    assert!(
                                        got.abs_diff(expected) <= 2,
                                        "{extent:?}/{bits}/{exif} ({x},{y}): {} != {}",
                                        f32::from_bits(got),
                                        f32::from_bits(expected)
                                    );
                                    rounded[offset..offset + 4]
                                        .copy_from_slice(&actual[offset..offset + 4]);
                                }
                            }
                        }
                        assert_eq!(actual, &rounded, "{extent:?}/{bits}/{exif}/{floating}");
                    };
                    assert_output(&mapped[..buffer_size as usize]);
                    if floating {
                        assert_output(&mapped[buffer_size as usize..buffer_size as usize * 2]);
                    }
                    assert_eq!(
                        &mapped[buffer_size as usize * 2..buffer_size as usize * 2 + 16],
                        bytemuck::cast_slice::<u32, u8>(&[1_u32, 0, 0, 0]),
                        "common finalizer status {extent:?}/{bits}/{exif}"
                    );
                    let scalar_status = ModularScalarOutputScratch::validate_status(
                        &mapped[buffer_size as usize * 2 + 16..],
                    );
                    if floating {
                        scalar_status.unwrap();
                    } else {
                        assert!(matches!(
                            scalar_status,
                            Err(ModularScalarOutputError::SampleOutOfRange)
                        ));
                    }
                    drop(mapped);
                    staging.unmap();
                }
            }
        }
    }
}
