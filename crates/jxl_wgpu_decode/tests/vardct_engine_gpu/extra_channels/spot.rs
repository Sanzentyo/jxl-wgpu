use super::*;
use jxl_wgpu_decode::AlphaOutputPolicy;

#[path = "spot_output.rs"]
mod output;

fn cases() -> [(&'static str, &'static str); 11] {
    [
        (
            "modular",
            include_str!("../../../test-data/extras_rgb12.jxl.hex"),
        ),
        (
            "vardct",
            include_str!("../../../test-data/vardct_extras_rgb12.jxl.hex"),
        ),
        (
            "modular_gray",
            include_str!("../../../test-data/extras_gray8.jxl.hex"),
        ),
        (
            "vardct_gray",
            include_str!("../../../test-data/vardct_extras_gray8.jxl.hex"),
        ),
        (
            "composition",
            include_str!("../../../test-data/composition_extras_rgb.jxl.hex"),
        ),
        (
            "composition_vardct",
            include_str!("../../../test-data/composition_extras_vardct.jxl.hex"),
        ),
        (
            "composition_gray",
            include_str!("../../../test-data/composition_extras_gray.jxl.hex"),
        ),
        (
            "composition_distributed",
            include_str!("../../../test-data/composition_extras_distributed.jxl.hex"),
        ),
        (
            "composition_resampled",
            include_str!("../../../test-data/composition_extras_resampled.jxl.hex"),
        ),
        (
            "composition_vardct_resampled",
            include_str!("../../../test-data/composition_extras_vardct_resampled.jxl.hex"),
        ),
        (
            "composition_data",
            include_str!("../../../test-data/composition_extras_data.jxl.hex"),
        ),
    ]
}

fn compare(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    let maximum = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (&a, &b))| {
            let error = (a - b).abs() / b.abs().max(1.0);
            assert!(
                a.is_finite() && b.is_finite() && error < tolerance,
                "{name}/{index}: {a} vs {b}, error {error}"
            );
            error
        })
        .fold(0.0_f32, f32::max);
    eprintln!("{name}: maximum relative error {maximum}");
}

#[test]
fn gpu_spots_follow_the_reference_color_domain_and_presentation_order() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for (name, hex) in cases() {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let pixels = image.width as usize * image.height as usize;
        for (linear, keep) in [(false, false), (true, true)] {
            let mut flags = vec!["--render-spots", "--preserve-alpha"];
            if linear {
                flags.push("--linear");
            }
            if keep {
                flags.push("--keep-orientation");
            }
            // Gray CMS discards colored spots, and its approximate transfer for extended
            // composed values differs from the exact sRGB curve. Transform RGB independently.
            let analytic_linear = linear && (image.grayscale || inventory.frames.len() > 1);
            if analytic_linear {
                flags.retain(|flag| *flag != "--linear");
            }
            let expected = oracle::libjxl_output(&data, &flags).map(|mut values| {
                if analytic_linear {
                    for frame in values.chunks_exact_mut(pixels * (4 + image.extra_channels.len()))
                    {
                        associated::linearize(&mut frame[..pixels * 4]);
                    }
                }
                values
            });
            let request = associated::floating_request(AlphaOutputPolicy::Preserve, linear, keep)
                .with_spot_color_policy(SpotColorPolicy::Render);
            let frames = associated::decode(&backend, &data, request, keep);
            if let Some(expected) = expected {
                let stride = pixels * (4 + image.extra_channels.len());
                assert_eq!(expected.len(), frames.len() * stride);
                for ((layout, bytes), expected) in frames.iter().zip(expected.chunks_exact(stride))
                {
                    let actual = associated::unpack(layout, bytes, keep);
                    compare(name, &actual, &expected[..pixels * 4], 0.0005);
                }
            }
        }
    }
}

fn multiple_spots() -> [(&'static str, &'static str); 10] {
    [
        (
            "modular_rgb",
            include_str!("../../../test-data/extras_spots_rgb.jxl.hex"),
        ),
        (
            "vardct_rgb",
            include_str!("../../../test-data/vardct_extras_spots_rgb.jxl.hex"),
        ),
        (
            "modular_gray",
            include_str!("../../../test-data/extras_spots_gray.jxl.hex"),
        ),
        (
            "vardct_gray",
            include_str!("../../../test-data/vardct_extras_spots_gray.jxl.hex"),
        ),
        (
            "modular_thin",
            include_str!("../../../test-data/extras_spots_thin.jxl.hex"),
        ),
        (
            "vardct_thin",
            include_str!("../../../test-data/vardct_extras_spots_thin.jxl.hex"),
        ),
        (
            "modular_resampled",
            include_str!("../../../test-data/extras_spots_resampled.jxl.hex"),
        ),
        (
            "vardct_resampled",
            include_str!("../../../test-data/vardct_extras_spots_resampled.jxl.hex"),
        ),
        (
            "modular_distributed",
            include_str!("../../../test-data/extras_spots_distributed.jxl.hex"),
        ),
        (
            "vardct_distributed",
            include_str!("../../../test-data/vardct_extras_spots_distributed.jxl.hex"),
        ),
    ]
}

// An independent pixel formula uses libjxl's untinted reconstructed color/extra planes.
// This also checks extended ink RGB and solidity without inheriting gray CMS behavior.
fn formula(
    data: &[u8],
    image: &jxl_gpu_bitstream::ImageHeaderInventory,
    linear: bool,
    keep: bool,
) -> Option<Vec<f32>> {
    let mut flags = vec!["--preserve-alpha"];
    if keep {
        flags.push("--keep-orientation");
    }
    if image.xyb_encoded {
        flags.push("--linear");
    }
    let planes = oracle::libjxl_output(data, &flags)?;
    let pixels = image.width as usize * image.height as usize;
    assert_eq!(planes.len(), pixels * (4 + image.extra_channels.len()));
    let mut color = planes[..pixels * 4].to_vec();
    for (index, extra) in image.extra_channels.iter().enumerate() {
        if let ExtraChannelTypeInventory::SpotColour {
            red,
            green,
            blue,
            solidity,
        } = extra.channel_type
        {
            let ink = [red.to_f32(), green.to_f32(), blue.to_f32()];
            for (pixel, output) in color.chunks_exact_mut(4).enumerate() {
                let mix = solidity.to_f32() * planes[pixels * (4 + index) + pixel];
                for (output, ink) in output[..3].iter_mut().zip(ink) {
                    *output = mix * ink + (1.0 - mix) * *output;
                }
            }
        }
    }
    if linear && !image.xyb_encoded {
        associated::linearize(&mut color);
    }
    if !linear && image.xyb_encoded {
        for pixel in color.chunks_exact_mut(4) {
            for c in &mut pixel[..3] {
                let value = c.abs();
                *c = if value <= 0.0031308 {
                    12.92 * value
                } else {
                    1.055 * value.powf(1.0 / 2.4) - 0.055
                }
                .copysign(*c);
            }
        }
    }
    Some(color)
}

#[test]
fn multiple_spots_preserve_declaration_order_extended_values_and_alpha_policy() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for (name, hex) in multiple_spots() {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        assert_eq!(
            image
                .extra_channels
                .iter()
                .filter(|e| matches!(e.channel_type, ExtraChannelTypeInventory::SpotColour { .. }))
                .count(),
            5
        );
        let ExtraChannelTypeInventory::Alpha { associated } = image.extra_channels[2].channel_type
        else {
            panic!("non-leading alpha")
        };
        for (linear, keep) in [(false, false), (true, true)] {
            let expected = formula(&data, image, linear, keep);
            for policy in [
                AlphaOutputPolicy::Preserve,
                AlphaOutputPolicy::Unassociated,
                AlphaOutputPolicy::Associated,
            ] {
                let request = associated::floating_request(policy, linear, keep)
                    .with_spot_color_policy(SpotColorPolicy::Render);
                let whole = associated::decode(&backend, &data, request.clone(), false);
                assert_eq!(
                    whole,
                    associated::decode(&backend, &data, request, true),
                    "{name}: bounded input"
                );
                assert_eq!(whole.len(), 1);
                if let Some(expected) = &expected {
                    let mut expected = expected.clone();
                    associated::associate(&mut expected, policy, associated);
                    let mut actual = associated::unpack(&whole[0].0, &whole[0].1, keep);
                    if associated && policy == AlphaOutputPolicy::Unassociated {
                        // Measure reconstruction before near-zero alpha magnifies the error.
                        for (a, b) in actual.chunks_exact_mut(4).zip(expected.chunks_exact_mut(4)) {
                            let scale = b[3].max(1.0 / 67108864.0);
                            for c in 0..3 {
                                a[c] *= scale;
                                b[c] *= scale;
                            }
                        }
                    }
                    compare(
                        name,
                        &actual,
                        &expected,
                        if image.xyb_encoded { 0.002 } else { 3e-6 },
                    );
                }
            }
        }
    }
}
