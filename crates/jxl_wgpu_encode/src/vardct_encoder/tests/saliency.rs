use super::*;
use crate::{
    FrameEncodeRequest, FrameIndex, GpuEncodeBackend, GpuEncodeJob, GpuFrameSource,
    VarDctGroupOrder,
};

/// Independent pixel-edge traversal, using u64 totals and no production geometry helper.
pub(super) fn oracle(width: usize, height: usize, pixels: &[[u8; 3]]) -> Vec<(u64, u64)> {
    let columns = width.div_ceil(256);
    let mut groups = vec![(0, 0); columns * height.div_ceil(256)];
    for y in 0..height {
        for x in 0..width {
            let group = &mut groups[y / 256 * columns + x / 256];
            for other in [
                x.checked_sub(1).map(|x| y * width + x),
                y.checked_sub(1).map(|y| y * width + x),
            ]
            .into_iter()
            .flatten()
            {
                group.0 += 1;
                for (&value, &neighbor) in pixels[y * width + x].iter().zip(&pixels[other]) {
                    group.1 += u64::from(value.abs_diff(neighbor));
                }
            }
        }
    }
    groups
}

pub(super) fn expected_order(width: usize, height: usize, pixels: &[[u8; 3]]) -> Vec<u32> {
    let groups = oracle(width, height, pixels);
    let mut order: Vec<_> = (0..groups.len() as u32).collect();
    // f64 distinguishes all possible bounded score fractions; production uses integer products.
    order.sort_by(|&a, &b| {
        let mean = |id: u32| {
            let (edges, sum) = groups[id as usize];
            sum as f64 / edges.max(1) as f64
        };
        mean(b).total_cmp(&mean(a)).then(a.cmp(&b))
    });
    order
}

fn request(width: usize, height: usize, config: &VarDctConfig) -> FrameEncodeRequest {
    FrameEncodeRequest {
        frame_index: FrameIndex::new(0),
        is_last: true,
        profile: crate::EncodeProfile::VarDct {
            quantization: config.quantization,
        },
        progressive: config.progressive.clone(),
        minimum_determinism: crate::Determinism::SameDevice,
        animation: crate::AnimationHeader::Still,
        canvas_width: width as u32,
        canvas_height: height as u32,
        options: crate::FrameOptions::default(),
    }
}

#[test]
fn gpu_saliency_statistics_match_independent_visible_pixel_edges_in_every_variant() {
    let (device, queue, info) = test_device().expect("actual GPU required for saliency");
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Lanes64,
        KernelVariant::Lanes128,
        KernelVariant::Lanes256,
    ] {
        let context = test_context_with_variants(
            &device,
            &queue,
            &info,
            &[(TILED_KERNEL_KEY, variant), (FORWARD_KERNEL_KEY, variant)],
        )
        .unwrap();
        let config = VarDctConfig {
            group_order: VarDctGroupOrder::saliency_first(),
            ..Default::default()
        };
        let backend = super::super::dispatch::VarDctBackend::new_tiled_dct8_with_config(
            &context,
            config.clone(),
        )
        .unwrap();
        assert!(
            backend
                .capabilities()
                .has_stage(crate::KernelStage::GroupOrderSelection)
        );
        for (width, height) in [
            (1, 1),
            (17, 1),
            (13, 21),
            (257, 17),
            (513, 257),
            (2057, 17),
            (16_384, 1),
            (1, 16_384),
        ] {
            let pixels = reference::pattern(width, height);
            let source = padded_rgb_source_sized(&context, width, height, &pixels);
            let (records, artifacts) = backend
                .submit(
                    &context,
                    GpuFrameSource::Buffer(source),
                    &request(width, height, &config),
                )
                .unwrap()
                .wait_with_saliency_for_test()
                .unwrap();
            let expected = oracle(width, height, &pixels);
            assert_eq!(records.len(), expected.len());
            for (record, (edges, sum)) in records.iter().zip(&expected) {
                assert_eq!(
                    (u64::from(record.edges), u64::from(record.contrast)),
                    (*edges, *sum),
                    "{variant:?}/{width}x{height}/{}",
                    record.group
                );
            }
            let actual: Vec<_> = artifacts
                .packets
                .packets_in_file_order()
                .filter_map(|packet| match packet.kind {
                    crate::GroupPacketKind::AcGroup { pass: 0, group } => Some(group),
                    _ => None,
                })
                .collect();
            if !artifacts.packets.layout.is_fused_single_group() {
                assert_eq!(actual, expected_order(width, height, &pixels));
            }
        }
        // Every edge attains the integer accumulator bound, including group boundaries.
        let (width, height) = (257, 257);
        let pixels: Vec<_> = (0..width * height)
            .map(|i| {
                [if (i % width + i / width) % 2 == 0 {
                    0
                } else {
                    255
                }; 3]
            })
            .collect();
        let source = padded_rgb_source_sized(&context, width, height, &pixels);
        let (records, _) = backend
            .submit(
                &context,
                GpuFrameSource::Buffer(source),
                &request(width, height, &config),
            )
            .unwrap()
            .wait_with_saliency_for_test()
            .unwrap();
        for record in records {
            assert_eq!(record.contrast, record.edges * 765);
        }
        // The general forward path uses exactly the same visible source grid, even when
        // replicated transforms cover padding outside odd image edges.
        for (width, height, all) in [(512, 512, true), (13, 21, false), (2057, 17, false)] {
            let map = mixed::packed_map(width as u32, height as u32, all);
            let backend = super::super::dispatch::VarDctBackend::new_with_strategy_map(
                &context,
                map,
                config.clone(),
            )
            .unwrap();
            let pixels = reference::pattern(width, height);
            let source = padded_rgb_source_sized(&context, width, height, &pixels);
            let (records, _) = backend
                .submit(
                    &context,
                    GpuFrameSource::Buffer(source),
                    &request(width, height, &config),
                )
                .unwrap()
                .wait_with_saliency_for_test()
                .unwrap();
            assert_eq!(
                records
                    .iter()
                    .map(|r| (u64::from(r.edges), u64::from(r.contrast)))
                    .collect::<Vec<_>>(),
                oracle(width, height, &pixels)
            );
        }
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
}

fn bytes_with_order(
    context: &WgpuContext,
    width: usize,
    height: usize,
    pixels: &[[u8; 3]],
    group_order: VarDctGroupOrder,
) -> Vec<u8> {
    let config = VarDctConfig {
        progressive: crate::ProgressivePlan::new(
            [1, 0]
                .map(|shift| crate::ProgressivePass {
                    coefficient_square: std::num::NonZeroU8::new(8).unwrap(),
                    shift,
                })
                .to_vec(),
        )
        .unwrap(),
        group_order,
        ..Default::default()
    };
    TiledVarDctEncoder::new_with_config(context.clone(), config)
        .unwrap()
        .encode(padded_rgb_source_sized(context, width, height, pixels))
        .unwrap()
}

#[test]
fn contrast_priority_improves_a_native_partial_group_image_without_changing_entropy() {
    use jxl_gpu_bitstream::FrameSectionKind;
    use jxl_test_support::oracles::progressive::{
        scalar_linear_prefix_updates, scalar_linear_updates,
    };
    let context = test_context().expect("actual GPU required for salient delivery");
    let (width, height) = (512, 256);
    let pixels: Vec<_> = (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            if x < 256 {
                [96; 3]
            } else {
                [if (x / 4 + y / 4) % 2 == 0 { 0 } else { 255 }; 3]
            }
        })
        .collect();
    assert_eq!(expected_order(width, height, &pixels), [1, 0]);
    let raster = bytes_with_order(&context, width, height, &pixels, Default::default());
    let salient = bytes_with_order(
        &context,
        width,
        height,
        &pixels,
        VarDctGroupOrder::saliency_first(),
    );
    let explicit = bytes_with_order(
        &context,
        width,
        height,
        &pixels,
        VarDctGroupOrder::explicit(vec![1, 0]).unwrap(),
    );
    assert_eq!(salient, explicit);
    let mut inventories = Vec::new();
    for bytes in [&raster, &salient] {
        inventories.push(
            jxl_gpu_bitstream::parse(bytes, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap(),
        );
    }
    for original in &inventories[0].frames[0].sections {
        let reordered = inventories[1].frames[0]
            .sections
            .iter()
            .find(|section| section.kind == original.kind)
            .unwrap();
        let slice = |bytes: &[u8], range: jxl_gpu_bitstream::ByteRange| {
            bytes[range.offset as usize..range.end().unwrap() as usize].to_vec()
        };
        assert_eq!(
            slice(&raster, original.bytes),
            slice(&salient, reordered.bytes)
        );
    }
    let full = scalar_linear_updates(&raster);
    let ordered_full = scalar_linear_updates(&salient);
    assert_eq!(full.len(), ordered_full.len());
    for (before, after) in full.iter().zip(&ordered_full) {
        assert_eq!(before.pixels, after.pixels);
    }
    let target = &full.last().unwrap().pixels;
    let mut errors = Vec::new();
    for (bytes, inventory) in [&raster, &salient].into_iter().zip(&inventories) {
        let first = inventory.frames[0]
            .sections
            .iter()
            .filter(|section| {
                matches!(
                    section.kind,
                    FrameSectionKind::PassGroup { pass_index: 0, .. }
                )
            })
            .min_by_key(|section| section.bitstream_index)
            .unwrap();
        let updates = scalar_linear_prefix_updates(&bytes[..first.bytes.end().unwrap() as usize]);
        let prefix = updates.last().unwrap();
        assert!(!prefix.complete);
        assert_eq!(prefix.pixels.len(), target.len());
        let error: f64 = prefix
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .zip(target.as_chunks::<4>().0)
            .map(|(a, b)| {
                (f64::from(f32::from_le_bytes(*a)) - f64::from(f32::from_le_bytes(*b))).powi(2)
            })
            .sum();
        errors.push(error);
    }
    assert!(
        errors[1] < errors[0],
        "one complete AC group, squared linear error: raster={}, salient={}",
        errors[0],
        errors[1]
    );
    eprintln!(
        "native one-group prefix squared linear error: raster={}, salient={}",
        errors[0], errors[1]
    );
    assert_eq!(
        quantization::assert_decoders_agree(&salient, width, height),
        decode_rgb8_sized(&raster, width, height)
    );
    // Flat groups have exact score ties and retain raster bytes, including clipped groups.
    let flat = vec![[31, 89, 167]; 513 * 17];
    assert_eq!(
        bytes_with_order(&context, 513, 17, &flat, VarDctGroupOrder::saliency_first()),
        bytes_with_order(&context, 513, 17, &flat, Default::default())
    );
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn saliency_layout_and_record_validation_reject_missing_forged_and_overflowed_metadata() {
    use super::super::saliency::{READY, Record, validate};
    let frame = VarDctFrameLayout::tiled_dct8(257, 17).unwrap();
    let code = fixed_prefix_code().unwrap();
    let hf = HfEntropyPlan::single_cluster_prefix().unwrap();
    let layout = ArtifactLayout::for_tiled_grid(frame, &code, &hf).unwrap();
    let largest = layout.with_saliency(4096).unwrap();
    assert_eq!(largest.artifact_words - largest.saliency_offset, 16_384);
    for count in [0, 4097, u32::MAX] {
        assert!(layout.with_saliency(count).is_err());
    }
    let layout = layout.with_passes(11).unwrap().with_saliency(2).unwrap();
    assert!(layout.with_saliency(2).is_err());
    assert!(layout.with_passes(1).is_err());
    let mut overflow = layout;
    overflow.saliency_groups = 0;
    overflow.artifact_words = u32::MAX - 1;
    assert!(overflow.with_saliency(2).is_err());
    let records: Vec<_> = oracle(257, 17, &reference::pattern(257, 17))
        .into_iter()
        .enumerate()
        .map(|(group, (edges, sum))| Record {
            status: READY,
            group: group as u32,
            edges: edges as u32,
            contrast: sum as u32,
        })
        .collect();
    validate(&records, frame).unwrap();
    assert!(validate(&records[..1], frame).is_err());
    for entry in 0..2 {
        for field in 0..4 {
            let mut corrupt = records.clone();
            match field {
                0 => corrupt[entry].status = 0,
                1 => corrupt[entry].group ^= 1,
                2 => corrupt[entry].edges += 1,
                _ => corrupt[entry].contrast = corrupt[entry].edges * 765 + 1,
            }
            assert!(validate(&corrupt, frame).is_err());
        }
    }
    let tiny = VarDctFrameLayout::tiled_dct8(1, 1).unwrap();
    assert!(
        validate(
            &[Record {
                status: READY,
                group: 0,
                edges: 0,
                contrast: 1
            }],
            tiny
        )
        .is_err()
    );
    assert!(
        VarDctGroupOrder::saliency_first()
            .validate_scores(frame, None)
            .is_err()
    );
    assert!(
        VarDctGroupOrder::default()
            .validate_scores(frame, Some(&records))
            .is_err()
    );
}

#[test]
fn saliency_artifact_tail_is_checked_against_device_limits_before_admission() {
    let frame = VarDctFrameLayout::tiled_dct8(8, 8).unwrap();
    let limit = ArtifactLayout::for_tiled_grid(
        frame,
        &fixed_prefix_code().unwrap(),
        &HfEntropyPlan::single_cluster_prefix().unwrap(),
    )
    .unwrap()
    .artifact_bytes();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let mut limits = wgpu::Limits::default().using_resolution(adapter.limits());
    limits.max_storage_buffer_binding_size = limit;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap();
    let context = WgpuContext::new(Arc::new(device), Arc::new(queue)).unwrap();
    let source = padded_rgb_source_sized(&context, 8, 8, &reference::pattern(8, 8));
    let raster = TiledVarDctEncoder::new(context.clone()).unwrap();
    assert_eq!(
        raster.memory_plan(&source).unwrap().artifact_storage_bytes,
        limit
    );
    raster.encode(source.clone()).unwrap();
    let salient = TiledVarDctEncoder::new_with_config(
        context.clone(),
        VarDctConfig {
            group_order: VarDctGroupOrder::saliency_first(),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        matches!(salient.submit(source), Err(EncodeError::Unsupported(UnsupportedFeature::DeviceLimit { name: "max_storage_buffer_binding_size", required, available })) if required == limit + 256 && available == limit)
    );
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn saliency_checks_both_dispatch_axes_before_memory_admission() {
    use super::super::dispatch::VarDctBackend;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let mut limits = wgpu::Limits::default().using_resolution(adapter.limits());
    limits.max_compute_workgroups_per_dimension = 32;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap();
    let context = test_context_with_variants(
        &Arc::new(device),
        &Arc::new(queue),
        &adapter.get_info(),
        &[(FORWARD_KERNEL_KEY, KernelVariant::Lanes256)],
    )
    .unwrap();
    for (width, height) in [(16_384, 1), (1, 16_384)] {
        let source =
            padded_rgb_source_sized(&context, width, height, &reference::pattern(width, height));
        let map = mixed::packed_map(width as u32, height as u32, false);
        // Linearized codec dispatches fit 32×32, but the 64-group saliency axis does not.
        let raster =
            VarDctBackend::new_with_strategy_map(&context, map.clone(), Default::default())
                .unwrap();
        raster
            .submit(
                &context,
                GpuFrameSource::Buffer(source.clone()),
                &request(width, height, &Default::default()),
            )
            .unwrap()
            .wait()
            .unwrap();
        let config = VarDctConfig {
            group_order: VarDctGroupOrder::saliency_first(),
            ..Default::default()
        };
        let salient = VarDctBackend::new_with_strategy_map(&context, map, config.clone()).unwrap();
        assert!(matches!(
            salient.submit(
                &context,
                GpuFrameSource::Buffer(source),
                &request(width, height, &config),
            ),
            Err(EncodeError::Unsupported(UnsupportedFeature::DeviceLimit {
                name: "max_compute_workgroups_per_dimension",
                required: 64,
                available: 32,
            }))
        ));
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
}
