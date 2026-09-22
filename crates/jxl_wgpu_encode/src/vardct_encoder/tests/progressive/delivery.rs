use super::*;

pub(super) fn endpoints(points: &[(u8, u8)]) -> Vec<ProgressiveDownsampling> {
    points
        .iter()
        .map(|&(factor, last_pass)| ProgressiveDownsampling { factor, last_pass })
        .collect()
}

pub(super) fn expected_ratio(progressive: &ProgressivePlan, completed: usize) -> u32 {
    if completed == progressive.passes().len() {
        return 1;
    }
    progressive
        .downsampling()
        .iter()
        .filter(|point| completed > usize::from(point.last_pass))
        .map(|point| u32::from(point.factor))
        .min()
        .unwrap_or(8)
}

pub(super) fn check_file_order_and_native_prefixes(
    bytes: &[u8],
    config: &VarDctConfig,
    inventory: &jxl_gpu_bitstream::FrameInventory,
    full: &[jxl_test_support::oracles::progressive::NativeUpdate],
) {
    use jxl_gpu_bitstream::FrameSectionKind;
    use jxl_test_support::oracles::progressive::scalar_linear_prefix_updates;
    let frame = VarDctFrameLayout::tiled_dct8(inventory.width, inventory.height).unwrap();
    let groups = config.group_order.resolve(frame).unwrap();
    assert_eq!(
        inventory.toc_permuted,
        !groups.iter().copied().eq(0..groups.len() as u32)
    );
    let mut physical = inventory.sections.clone();
    physical.sort_by_key(|section| section.bitstream_index);
    let actual = physical
        .iter()
        .filter_map(|section| match section.kind {
            FrameSectionKind::PassGroup {
                pass_index,
                group_index,
            } => Some((pass_index, group_index as u32)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let expected = (0..inventory.num_passes)
        .flat_map(|pass| groups.iter().map(move |&group| (pass, group)))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    if config.group_order == VarDctGroupOrder::default() {
        return;
    }
    // Only complete pass boundaries are advertised: all LF/HF metadata precedes AC.
    // The public GPU session is tested separately with complete input and bounded windows.
    for pass in 0..inventory.num_passes - 1 {
        let section = physical.iter().rev().find(|section| matches!(section.kind, FrameSectionKind::PassGroup { pass_index, .. } if pass_index == pass)).unwrap();
        let end = (section.bytes.offset + section.bytes.length) as usize;
        assert!(end < bytes.len());
        let prefix = scalar_linear_prefix_updates(&bytes[..end]);
        let last = prefix
            .last()
            .expect("native decoder must flush a complete AC pass");
        assert!(!last.complete);
        assert_eq!(
            last.ratio,
            expected_ratio(&config.progressive, pass as usize + 1)
        );
        assert_eq!(
            last.pixels,
            full[pass as usize + 1].pixels,
            "partial input after pass {pass}"
        );
    }
}

#[test]
fn resolution_stopping_points_reject_invalid_wire_syntax() {
    for invalid in [
        vec![(0, 0)],
        vec![(3, 0)],
        vec![(16, 0)],
        vec![(8, 8)],
        vec![(4, 0), (4, 1)],
        vec![(2, 0), (4, 1)],
        vec![(4, 1), (2, 1)],
        vec![(4, 2), (2, 1)],
        vec![(8, 0), (4, 1), (2, 2), (1, 3), (1, 4)],
    ] {
        assert!(
            maximum().with_downsampling(endpoints(&invalid)).is_err(),
            "{invalid:?}"
        );
    }
    assert!(
        ProgressivePlan::single()
            .with_downsampling(endpoints(&[(1, 0)]))
            .is_err()
    );
    assert!(
        plan(&[(4, 0), (8, 0)])
            .with_downsampling(endpoints(&[(2, 2)]))
            .is_err()
    );
    assert_eq!(
        ProgressivePlan::single().with_downsampling(vec![]).unwrap(),
        ProgressivePlan::single()
    );
    let points = endpoints(&[(8, 0), (4, 1), (2, 2), (1, 7)]);
    assert_eq!(
        maximum()
            .with_downsampling(points.clone())
            .unwrap()
            .downsampling(),
        points
    );
}

#[test]
fn group_order_is_bounded_and_uses_integer_geometry() {
    let frame = VarDctFrameLayout::tiled_dct8(768, 768).unwrap();
    assert_eq!(
        VarDctGroupOrder::center_first().resolve(frame).unwrap(),
        [4, 1, 3, 5, 7, 0, 2, 6, 8]
    );
    assert_eq!(
        VarDctGroupOrder::centered_at(767, 767)
            .resolve(frame)
            .unwrap(),
        [8, 5, 7, 4, 2, 6, 1, 3, 0]
    );
    assert_eq!(
        VarDctGroupOrder::default().resolve(frame).unwrap(),
        (0..9).collect::<Vec<_>>()
    );
    for (width, height) in [(1, 1), (1, 2057), (2057, 1), (513, 513), (16_384, 16_384)] {
        let frame = VarDctFrameLayout::tiled_dct8(width, height).unwrap();
        let order = VarDctGroupOrder::center_first().resolve(frame).unwrap();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..frame.ac_group_count().unwrap()).collect::<Vec<_>>()
        );
        assert_eq!(
            VarDctGroupOrder::explicit(order.clone())
                .unwrap()
                .resolve(frame)
                .unwrap(),
            order
        );
        assert!(
            VarDctGroupOrder::centered_at(width, 0)
                .resolve(frame)
                .is_err()
        );
        assert!(
            VarDctGroupOrder::centered_at(0, height)
                .resolve(frame)
                .is_err()
        );
        assert!(
            VarDctGroupOrder::centered_at(u32::MAX, u32::MAX)
                .resolve(frame)
                .is_err()
        );
        VarDctGroupOrder::centered_at(width - 1, height - 1)
            .resolve(frame)
            .unwrap();
    }
    for invalid in [
        vec![],
        vec![0, 0],
        vec![0, 2],
        vec![u32::MAX],
        (0..4097).collect(),
    ] {
        assert!(VarDctGroupOrder::explicit(invalid).is_err());
    }
    assert!(
        VarDctGroupOrder::explicit(vec![0, 1])
            .unwrap()
            .resolve(frame)
            .is_err()
    );
}

#[test]
fn every_resolution_endpoint_count_matches_native_pass_metadata() {
    use jxl_test_support::oracles::progressive::{native_updates, scalar_linear_updates};
    let context = test_context().expect("actual GPU required for progressive resolution hints");
    let source = padded_rgb_source_sized(&context, 17, 9, &reference::pattern(17, 9));
    let mut baseline = None;
    let points = endpoints(&[(8, 0), (4, 1), (2, 2), (1, 7)]);
    for count in 0..=4 {
        let progressive = maximum()
            .with_downsampling(points[..count].to_vec())
            .unwrap();
        let encoder = TiledVarDctEncoder::new_with_config(
            context.clone(),
            VarDctConfig {
                progressive: progressive.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        let bytes = encoder.encode(source.clone()).unwrap();
        let scalar = scalar_linear_updates(&bytes);
        let native = native_updates(&bytes, true).expect("native pass oracle required");
        assert_eq!(scalar.len(), 12);
        assert_eq!(native.len(), scalar.len());
        for (completed, (actual, native)) in scalar.iter().zip(&native).enumerate() {
            assert_eq!(actual.step, completed);
            assert_eq!(actual.complete, completed == 11);
            assert_eq!(actual.ratio, expected_ratio(&progressive, completed));
            assert_eq!(native.ratio, actual.ratio);
            assert_eq!(native.step, actual.step);
        }
        let images: Vec<_> = scalar.into_iter().map(|update| update.pixels).collect();
        if let Some(baseline) = &baseline {
            assert_eq!(
                &images, baseline,
                "resolution metadata cannot change reconstructed pixels"
            );
        } else {
            baseline = Some(images);
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn permuted_late_ac_rejects_before_publication_and_preserves_retained_previews() {
    use jxl_gpu_bitstream::FrameSectionKind;
    use jxl_wgpu_decode::{Error as DecodeError, VarDctDecodeError, WgpuDecodeEngine};
    let (device, queue, info) =
        test_device().expect("actual GPU required for permuted pass validation");
    let backend = WgpuBackend::from_device(
        device.as_ref().clone(),
        queue.as_ref().clone(),
        info,
        Default::default(),
    )
    .unwrap();
    let context = WgpuContext::from_backend(&backend);
    let source = padded_rgb_source_sized(&context, 257, 17, &reference::pattern(257, 17));
    let baseline = TiledVarDctEncoder::new(context.clone())
        .unwrap()
        .encode(source.clone())
        .unwrap();
    let identity = TiledVarDctEncoder::new_with_config(
        context.clone(),
        VarDctConfig {
            group_order: VarDctGroupOrder::explicit(vec![0, 1]).unwrap(),
            ..Default::default()
        },
    )
    .unwrap()
    .encode(source.clone())
    .unwrap();
    assert_eq!(identity, baseline);
    let reversed = TiledVarDctEncoder::new_with_config(
        context.clone(),
        VarDctConfig {
            group_order: VarDctGroupOrder::explicit(vec![1, 0]).unwrap(),
            ..Default::default()
        },
    )
    .unwrap()
    .encode(source.clone())
    .unwrap();
    assert_ne!(reversed, baseline);
    assert_eq!(
        quantization::assert_decoders_agree(&reversed, 257, 17),
        decode_rgb8_sized(&baseline, 257, 17)
    );
    let config = VarDctConfig {
        progressive: plan(&[(2, 0), (4, 0), (8, 0)])
            .with_downsampling(endpoints(&[(4, 0), (2, 1)]))
            .unwrap(),
        group_order: VarDctGroupOrder::explicit(vec![1, 0]).unwrap(),
        ..Default::default()
    };
    let bytes = TiledVarDctEncoder::new_with_config(context.clone(), config)
        .unwrap()
        .encode(source)
        .unwrap();
    let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert!(inventory.frames[0].toc_permuted);
    let packet = inventory.frames[0]
        .sections
        .iter()
        .find(|section| {
            section.kind
                == FrameSectionKind::PassGroup {
                    pass_index: 2,
                    group_index: 1,
                }
        })
        .unwrap();
    let mut damaged = bytes.clone();
    damaged[packet.bytes.offset as usize..packet.bytes.end().unwrap() as usize].fill(0xff);
    let readback = ImageReadbackPipeline::new(&backend);
    for cap in [u64::MAX, 256] {
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
        );
        let request = GpuOutputRequest::color(vardct_rgb8_format())
            .unwrap()
            .with_progressive_output(true);
        let mut session = decoder.open(&damaged, request).unwrap();
        let mut held = Vec::new();
        let mut pixels = Vec::new();
        for completed in 0..3 {
            let frame = session.next_update().unwrap().unwrap();
            assert!(!frame.is_complete());
            assert_eq!(
                frame.progression().unwrap().completed_passes().unwrap(),
                completed
            );
            pixels.push(
                readback
                    .submit(frame.output())
                    .unwrap()
                    .wait()
                    .unwrap()
                    .frame
                    .outputs[0]
                    .bytes
                    .clone(),
            );
            held.push(frame);
        }
        assert!(matches!(
            session.next_update(),
            Err(DecodeError::VarDct(VarDctDecodeError::HfCoefficientGpu(_)))
        ));
        assert!(matches!(
            session.next_update(),
            Err(DecodeError::SessionPoisoned)
        ));
        drop(session);
        for (frame, before) in held.iter().zip(&pixels) {
            assert_eq!(
                &readback
                    .submit(frame.output())
                    .unwrap()
                    .wait()
                    .unwrap()
                    .frame
                    .outputs[0]
                    .bytes,
                before
            );
        }
        drop(held);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
