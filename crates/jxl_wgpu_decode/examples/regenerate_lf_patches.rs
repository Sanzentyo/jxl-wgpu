//! Assemble separate-LF patch consumers and freeze native final/scalar LF presentation oracles.
use std::path::Path;

#[allow(dead_code)]
#[path = "../tests/support/lf_oracle.rs"]
mod lf_oracle;
#[allow(dead_code)]
#[path = "../tests/common/extra_channel_oracle.rs"]
mod native;
#[allow(dead_code)]
#[path = "support/offline.rs"]
mod offline;
#[path = "support/patch_oracle.rs"]
mod patch_oracle;
#[path = "../tests/support/patches.rs"]
#[allow(dead_code)]
mod patches;

fn parse_inventory(data: &[u8]) -> jxl_gpu_bitstream::CodestreamInventory {
    jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
}

fn native(data: &[u8]) -> Vec<f32> {
    native::libjxl_output(
        data,
        &["--linear", "--preserve-alpha", "--keep-orientation"],
    )
    .expect("native libjxl is required to regenerate LF patch references")
}

fn components(data: &[u8]) -> lf_oracle::Planes {
    let image = parse_inventory(data).image_header;
    let scaled = native::libjxl_output(data, &["--xyb", "--preserve-alpha", "--keep-orientation"])
        .expect("native libjxl XYB output is required");
    let pixels = image.width as usize * image.height as usize;
    assert_eq!(scaled.len(), pixels * (4 + image.extra_channels.len()));
    // libjxl v0.12.0 cms/opsin_params.h and render_pipeline/stage_xyb.cc. These
    // affine scale/offset inverses avoid the ill-conditioned RGB -> cube-root round trip.
    let scale = [
        f32::from_bits(0x41b7f760),
        f32::from_bits(0x3f976c8c),
        f32::from_bits(0x3fc0462b),
    ]
    .map(f64::from);
    let offset = [f32::from_bits(0x3c7c1620), 0.0, f32::from_bits(0x3e8e2f4c)].map(f64::from);
    let mut channels = vec![vec![0.0; pixels]; 3 + image.extra_channels.len()];
    for i in 0..pixels {
        for c in 0..3 {
            channels[c][i] = f64::from(scaled[i * 4 + c]) / scale[c] - offset[c];
        }
        channels[2][i] += channels[1][i];
        for c in 0..image.extra_channels.len() {
            channels[3 + c][i] = f64::from(scaled[(4 + c) * pixels + i]);
        }
    }
    lf_oracle::Planes {
        width: image.width as usize,
        height: image.height as usize,
        channels,
    }
}

fn main() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = data.join("patches/lf");
    std::fs::create_dir_all(&output).unwrap();
    for name in [
        "vardct_gab0",
        "modular_gab1",
        "nested_vardct_gab1",
        "nested_modular_gab1",
    ] {
        let source = offline::unhex(
            &std::fs::read_to_string(data.join(format!("lf_extra_channels/{name}.jxl.hex")))
                .unwrap(),
        );
        let source_inventory = parse_inventory(&source);
        let image = &source_inventory.image_header;
        assert!(image.xyb_encoded && !image.grayscale && image.orientation == 1);
        let frame = source_inventory.frames.last().unwrap();
        let raw_reference = components(&source);
        for (suffix, count) in [("", 16), ("_empty", 0)] {
            let values = patches::values(frame, image.extra_channels.len(), count);
            let encoded = patches::assemble_shared_lf(&source, &values);
            let inventory = parse_inventory(&encoded);
            assert_eq!(inventory.frames.len(), source_inventory.frames.len() + 1);
            let consumer = inventory.frames.last().unwrap();
            let reference = &inventory.frames[inventory.frames.len() - 2];
            assert_eq!(reference.lf_source_frame, consumer.lf_source_frame);
            assert_eq!(reference.save_as_reference, 3);
            assert!(reference.save_before_color_transform && consumer.flags & 2 != 0);
            let final_image = native(&encoded);
            // Establish that the independent component-domain patch algebra agrees with a
            // complete native image before using it as the LF presentation oracle.
            let mut raw_final = components(&source);
            patch_oracle::apply(&mut raw_final, &raw_reference, image, &values);
            let reconstructed = patch_oracle::packed(raw_final, image);
            assert_eq!(reconstructed.len(), final_image.len());
            let worst = reconstructed
                .iter()
                .zip(&final_image)
                .map(|(&a, &b)| {
                    assert!(a.is_finite() && b.is_finite());
                    (a - b).abs() / (1.0 + b.abs())
                })
                .fold(0.0f32, f32::max);
            assert!(worst < 2e-6, "{name}{suffix} scalar final mismatch {worst}");
            let mut snapshots = Vec::<f32>::new();
            for lf in &source_inventory.frames[..source_inventory.frames.len() - 1] {
                let level = lf.lf_level as u8;
                let small = offline::unhex(
                    &std::fs::read_to_string(
                        data.join(format!("lf_extra_channels/{name}.lf{level}.jxl.hex")),
                    )
                    .unwrap(),
                );
                let small_inventory = parse_inventory(&small);
                let small_image = &small_inventory.image_header;
                assert_eq!(image.extra_channels, small_image.extra_channels);
                assert_eq!(image.bit_depth, small_image.bit_depth);
                assert_eq!(image.opsin_inverse_matrix, small_image.opsin_inverse_matrix);
                for (original, standalone) in
                    source_inventory.frames.iter().zip(&small_inventory.frames)
                {
                    assert_eq!(
                        original.color_sample_extent(),
                        standalone.color_sample_extent()
                    );
                    assert_eq!(original.encoding, standalone.encoding);
                    assert_eq!(original.restoration_filter, standalone.restoration_filter);
                    assert_eq!(
                        original.extra_channel_upsampling,
                        standalone.extra_channel_upsampling
                    );
                    assert_eq!(original.sections.len(), standalone.sections.len());
                    for (a, b) in original.sections.iter().zip(&standalone.sections) {
                        assert_eq!(a.kind, b.kind);
                        assert_eq!(
                            &source[a.bytes.offset as usize..a.bytes.end().unwrap() as usize],
                            &small[b.bytes.offset as usize..b.bytes.end().unwrap() as usize]
                        );
                    }
                }
                let mut preview = components(&small);
                let native_small = native(&small);
                let restored = patch_oracle::packed(components(&small), image);
                assert_eq!(native_small.len(), restored.len());
                assert!(
                    native_small
                        .iter()
                        .zip(&restored)
                        .all(|(a, b)| (a - b).abs() < 0.0001 * (1.0 + a.abs())),
                    "{name}/LF{level} native producer"
                );
                preview.expand(image, [image.width, image.height], level);
                patch_oracle::apply(&mut preview, &raw_reference, image, &values);
                snapshots.extend(patch_oracle::packed(preview, image));
            }
            snapshots.extend(final_image);
            let name = format!("{name}{suffix}");
            std::fs::write(
                output.join(format!("{name}.jxl.hex")),
                offline::hex(&encoded),
            )
            .unwrap();
            let bytes: Vec<_> = snapshots.into_iter().flat_map(f32::to_le_bytes).collect();
            std::fs::write(
                output.join(format!("{name}.f32.hex")),
                offline::float_hex(&bytes),
            )
            .unwrap();
            eprintln!(
                "{name}: {} LF levels, {}x{}, {} bytes; scalar/native final error {worst}",
                source_inventory.frames.len() - 1,
                image.width,
                image.height,
                encoded.len()
            );
        }
    }
}
