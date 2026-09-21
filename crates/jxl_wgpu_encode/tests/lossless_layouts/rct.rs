use super::*;
use jxl_wgpu_encode::{
    LosslessModularColorTransform as Transform, LosslessModularConfig, LosslessModularGroupSize,
    LosslessModularRctType,
};

mod animation;
mod lifetime;

fn config(value: u32, local: bool, tree: usize) -> LosslessModularConfig {
    let rct = LosslessModularRctType::new(value).unwrap();
    LosslessModularConfig {
        group_size: LosslessModularGroupSize::ALL[value as usize % 4],
        tree_mode: TREES[tree],
        color_transform: if local {
            Transform::LocalRct(rct)
        } else {
            Transform::GlobalRct(rct)
        },
    }
}

fn matrix(kind: SampleKind, depths: &[u8]) {
    let rig = Rig::new();
    for value in 0..42 {
        for local in [false, true] {
            for tree in 0..2 {
                let config = config(value, local, tree);
                let edge = config.group_size.dimension();
                let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config);
                for (index, &bits) in depths.iter().enumerate() {
                    let case = Case {
                        format: if index % 2 == 0 {
                            LosslessModularFormat::Rgb
                        } else {
                            LosslessModularFormat::Rgba
                        },
                        bits,
                        kind,
                        storage: [Storage::Packed, Storage::Planar, Storage::Split]
                            [(value as usize + index) % 3],
                        reversed: true,
                        byte_order: ByteOrder::Big,
                        shifted: true,
                    };
                    // Cover fused packets and both tree choices in separate pass groups.
                    // The thin LF-boundary case also exercises the selected group's LF grid.
                    let extent = if tree == 0 && index % 2 == 0 {
                        Extent2d::new(17, 3)
                    } else if index == 0 {
                        Extent2d::new(1, edge * 8 + 1)
                    } else {
                        Extent2d::new(edge + 1, 3)
                    };
                    let expected = case.samples(extent);
                    let encoded = pollster::block_on(
                        encoder
                            .submit_container(upload(&rig.context, &case, extent, &expected, 4099))
                            .unwrap(),
                    )
                    .unwrap();
                    assert_eq!(
                        encoded,
                        encoder
                            .encode_container(upload(
                                &rig.context,
                                &case.canonical(),
                                extent,
                                &expected,
                                0,
                            ))
                            .unwrap(),
                        "RCT {value}, local={local}, tree={tree}, {case:?}"
                    );
                    let headers = modular_integer::local_rct_headers(&encoded, 0);
                    assert_eq!(
                        headers.len(),
                        if tree == 0 && index % 2 == 0 {
                            0
                        } else if index == 0 {
                            9
                        } else {
                            2
                        }
                    );
                    for header in headers {
                        assert_eq!(header, (tree == 0, local.then_some(value)));
                    }
                    check_oracles(&encoded, &expected, &case);
                    color::check_numeric(&rig, &encoded, &[expected], &case);
                    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
}

#[test]
fn every_rct_operation_and_permutation_preserves_integer_words() {
    matrix(SampleKind::Unsigned, &[1, 8, 16, 31]);
}

#[test]
fn every_rct_operation_and_permutation_preserves_ieee_words() {
    matrix(SampleKind::Float, &[16, 32]);
}

#[test]
fn explicit_none_and_identity_preserve_every_integer_depth_and_default_bytes() {
    let rig = Rig::new();
    let encoders: Vec<_> = [
        Transform::Auto,
        Transform::None,
        Transform::GlobalRct(LosslessModularRctType::IDENTITY),
        Transform::GlobalRct(LosslessModularRctType::YCOCG),
    ]
    .map(|color_transform| {
        LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                color_transform,
                ..Default::default()
            },
        )
    })
    .into_iter()
    .collect();
    for bits in 1..=31 {
        let case = Case {
            format: LosslessModularFormat::Rgba,
            bits,
            kind: SampleKind::Unsigned,
            storage: Storage::Planar,
            reversed: true,
            byte_order: ByteOrder::Big,
            shifted: true,
        };
        let extent = Extent2d::new(257, 3);
        let expected = case.samples(extent);
        let input = upload(&rig.context, &case, extent, &expected, 4099);
        let streams: Vec<_> = encoders
            .iter()
            .map(|encoder| encoder.encode(input.clone()).unwrap())
            .collect();
        assert_eq!(streams[0], streams[3]);
        // An explicit identity still declares its transform; None omits the transform entirely.
        assert_ne!(streams[1], streams[2]);
        for encoded in &streams[1..3] {
            check_oracles(encoded, &expected, &case);
            color::check_numeric(&rig, encoded, std::slice::from_ref(&expected), &case);
        }
    }
}
