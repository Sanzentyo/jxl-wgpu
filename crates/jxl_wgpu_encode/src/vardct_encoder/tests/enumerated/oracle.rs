//! Independent source declarations and f64 conversion. No production color-plan helpers.
use super::*;
use jxl_gpu_protocol::{Chromaticity as Xy, GammaExponent, RgbChromaticities as Rgb};
use jxl_test_support::oracles::{color, hdr};

pub(super) fn spaces() -> [ColorSpace; 7] {
    [
        ColorSpace::Bt709,
        ColorSpace::Bt2020,
        ColorSpace::DisplayP3,
        ColorSpace::CustomRgb(Rgb {
            white: Xy::DCI,
            ..Rgb::DISPLAY_P3
        }),
        ColorSpace::CustomRgb(Rgb {
            white: Xy::E,
            ..Rgb::BT709
        }),
        ColorSpace::CustomRgb(Rgb {
            white: Xy::new(0.34567, 0.35850).unwrap(),
            ..Rgb::BT2020
        }),
        ColorSpace::CustomRgb(Rgb {
            red: Xy::new(0.6600003, 0.3200004).unwrap(),
            green: Xy::new(0.2399998, 0.7000002).unwrap(),
            blue: Xy::new(0.1200001, 0.0399997).unwrap(),
            white: Xy::new(0.3212504, 0.3375001).unwrap(),
        }),
    ]
}

pub(super) fn transfers() -> [TransferFunction; 7] {
    [
        TransferFunction::Linear,
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Pq,
        TransferFunction::Hlg,
        TransferFunction::Dci,
        TransferFunction::Gamma(GammaExponent::new(1.0 / 2.2).unwrap()),
    ]
}

pub(super) const INTENTS: [IccRenderingIntent; 4] = [
    IccRenderingIntent::Perceptual,
    IccRenderingIntent::Relative,
    IccRenderingIntent::Saturation,
    IccRenderingIntent::Absolute,
];

pub(super) fn spec(space: ColorSpace, transfer: TransferFunction) -> ColorSpecification {
    ColorSpecification::Defined(ColorSpec {
        space,
        transfer,
        range: ColorRange::Full,
        encoding: YcbcrEncoding::Undefined,
        chroma_location: jxl_gpu_formats::ChromaLocation2d::CENTER,
    })
}

pub(super) fn config(
    channels: ColorChannels,
    index: usize,
    color: VarDctColorTransform,
) -> VarDctConfig {
    VarDctConfig {
        sample_format: ColorSampleFormat::integer(channels, 12).unwrap(),
        source_color: spec(spaces()[index / 7 % 7], transfers()[index % 7]),
        image_options: crate::ImageOptions {
            rendering_intent: INTENTS[index % 4],
            intensity_target: FiniteF16::from_bits([0x5bf8, 0x63d0, 0x6bd0][index % 3]).unwrap(),
            ..Default::default()
        },
        ..precision::configuration(8, color)
    }
}

/// Gray omits primaries; custom coordinates and gamma have the JPEG XL wire precision.
pub(super) fn wire_spec(config: &VarDctConfig) -> ColorSpec {
    let ColorSpecification::Defined(mut spec) = config.source_color else {
        panic!("explicit test color")
    };
    if let ColorSpace::CustomRgb(mut rgb) = spec.space {
        let xy = |p: Xy| Xy::new((p.x() * 1e6).round() / 1e6, (p.y() * 1e6).round() / 1e6).unwrap();
        rgb.red = xy(rgb.red);
        rgb.green = xy(rgb.green);
        rgb.blue = xy(rgb.blue);
        // Enumerated E/DCI/D65 are exact constants, not custom integer coordinates.
        if ![Xy::D65, Xy::E, Xy::DCI].contains(&rgb.white) {
            rgb.white = xy(rgb.white);
        }
        spec.space = ColorSpace::CustomRgb(rgb);
    }
    if config.sample_format.channels() == ColorChannels::Gray {
        let white = spec
            .space
            .rgb_space()
            .unwrap()
            .chromaticities()
            .unwrap()
            .white;
        spec.space = if white == Xy::D65 {
            ColorSpace::Bt709
        } else {
            ColorSpace::CustomRgb(Rgb {
                white,
                ..Rgb::BT709
            })
        };
    }
    if let TransferFunction::Gamma(gamma) = spec.transfer {
        spec.transfer = TransferFunction::Gamma(
            GammaExponent::new(((f64::from(gamma.value()) * 1e7).round() / 1e7) as f32).unwrap(),
        );
    }
    spec
}

pub(super) fn components(values: &[[f64; 3]], config: &VarDctConfig) -> Vec<[f64; 3]> {
    let spec = wire_spec(config);
    let matrix = color::matrix(spec.space, ColorSpace::Bt709);
    let nits = f64::from(config.image_options.intensity_target.to_f32());
    values
        .iter()
        .map(|&rgb| {
            if config.color_transform == VarDctColorTransform::Original {
                return rgb;
            }
            let linear = hdr::to_linear(rgb, spec.transfer, spec.space, nits);
            let scaled: [f64; 3] =
                matrix.map(|row| (0..3).map(|c| row[c] * linear[c]).sum::<f64>() * nits / 255.0);
            let bias = 0.0037930732552754493_f64;
            let opsin = [
                [0.3, 0.622, 0.078],
                [0.23, 0.692, 0.078],
                [0.2434226892, 0.2047674442, 0.5518098665],
            ]
            .map(|row| {
                (bias + (0..3).map(|c| row[c] * scaled[c]).sum::<f64>())
                    .max(0.0)
                    .cbrt()
                    - bias.cbrt()
            });
            [
                (opsin[0] - opsin[1]) * 0.5,
                (opsin[0] + opsin[1]) * 0.5,
                opsin[2],
            ]
        })
        .collect()
}

/// Public libjxl encoding receives source declarations independently of the tested writer.
pub(super) fn declaration(config: &VarDctConfig) -> String {
    let spec = wire_spec(config);
    let xy = spec.space.rgb_space().unwrap().chromaticities().unwrap();
    let white = match xy.white {
        Xy::D65 => 1,
        Xy::E => 10,
        Xy::DCI => 11,
        _ => 2,
    };
    let primaries = match spec.space {
        ColorSpace::Bt709 => 1,
        ColorSpace::Bt2020 => 9,
        ColorSpace::DisplayP3 => 11,
        _ => 2,
    };
    let (transfer, gamma) = match spec.transfer {
        TransferFunction::Linear => (8, 1.0),
        TransferFunction::Srgb => (13, 0.0),
        TransferFunction::Bt709 => (1, 0.0),
        TransferFunction::Pq => (16, 0.0),
        TransferFunction::Hlg => (18, 0.0),
        TransferFunction::Dci => (17, 0.0),
        TransferFunction::Gamma(g) => (65535, f64::from(g.value())),
        _ => unreachable!(),
    };
    format!(
        "{} {white} {primaries} {transfer} {} {} {} {} {} {} {} {} {} {gamma}",
        u32::from(config.sample_format.channels() == ColorChannels::Gray),
        config.image_options.rendering_intent as u32,
        xy.white.x(),
        xy.white.y(),
        xy.red.x(),
        xy.red.y(),
        xy.green.x(),
        xy.green.y(),
        xy.blue.x(),
        xy.blue.y()
    )
}

pub(super) fn pixels(bytes: &[u8], config: &VarDctConfig) -> Vec<f32> {
    let nits = f64::from(config.image_options.intensity_target.to_f32());
    let exponent = (1.2 * 1.111_f64.powf((nits / 1000.0).log2())).recip() - 1.0;
    let hlg_threshold_mismatch = wire_spec(config).transfer == TransferFunction::Hlg
        && (0.01..0.1).contains(&exponent.abs());
    if config.color_transform == VarDctColorTransform::Original || !hlg_threshold_mismatch {
        let mut frames = rust_frames(bytes, config);
        assert_eq!(frames.len(), 1);
        return frames.remove(0);
    }
    // Rust jxl 0.6.0 skips HLG OOTF for |exponent| < 0.1 (the wire uses 0.01).
    // jxl-oxide additionally tone maps some requested RGB targets automatically.
    // Independently decoded raw XYB plus f64 colorimetry avoids both color policies.
    let mut image = jxl_oxide::JxlImage::read_with_defaults(bytes).unwrap();
    image.request_color_encoding(jxl_oxide::EnumColourEncoding {
        colour_space: jxl_oxide::color::ColourSpace::Xyb,
        ..jxl_oxide::EnumColourEncoding::srgb_linear(jxl_oxide::RenderingIntent::Relative)
    });
    let opsin = &image.image_header().metadata.opsin_inverse_matrix;
    let bias = opsin.opsin_bias.map(f64::from);
    let matrix = opsin.inv_mat.map(|row| row.map(f64::from));
    let nits = f64::from(config.image_options.intensity_target.to_f32());
    let spec = wire_spec(config);
    let render = image.render_frame(0).unwrap();
    let pixels = render.image_all_channels();
    assert_eq!(pixels.channels(), 3);
    pixels
        .buf()
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|p| {
            let [x, y, b] = p.map(f64::from);
            let mixed = [y + x, y - x, b];
            let lms: [f64; 3] = std::array::from_fn(|c| {
                ((mixed[c] - bias[c].cbrt()).powi(3) + bias[c]) * 255.0 / nits
            });
            let rgb = matrix.map(|row| (0..3).map(|c| row[c] * lms[c]).sum());
            let linear = color::xyb_original_linear_for_profile(
                rgb,
                spec.space,
                config.sample_format.channels() == ColorChannels::Gray,
            );
            let rgb = hdr::from_linear(linear, spec.transfer, spec.space, nits);
            [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, 1.0]
        })
        .collect()
}

pub(super) fn rust_frames(bytes: &[u8], config: &VarDctConfig) -> Vec<Vec<f32>> {
    use jxl::api::{JxlColorEncoding, JxlColorProfile};
    let (frames, profile) =
        jxl_test_support::oracles::extra_channels::rust_frame_planes_with_profile(bytes);
    let spec = wire_spec(config);
    let gray_linear = config.color_transform == VarDctColorTransform::Xyb
        && config.sample_format.channels() == ColorChannels::Gray
        && spec
            .space
            .rgb_space()
            .unwrap()
            .chromaticities()
            .unwrap()
            .white
            != Xy::D65;
    if gray_linear {
        // Rust jxl explicitly selects linear-sRGB F32 when it cannot render a non-D65
        // Gray profile. Preserve that reported domain, then independently encode Gray.
        let JxlColorProfile::Simple(encoding) = profile else {
            panic!("expected enumerated output")
        };
        assert_eq!(encoding, JxlColorEncoding::linear_srgb(true));
    }
    frames
        .into_iter()
        .map(|(pixels, extras)| {
            assert!(extras.is_empty());
            if !gray_linear {
                return pixels;
            }
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    let rgb = hdr::from_linear(
                        [p[0], p[1], p[2]].map(f64::from),
                        spec.transfer,
                        spec.space,
                        f64::from(config.image_options.intensity_target.to_f32()),
                    );
                    [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, p[3]]
                })
                .collect()
        })
        .collect()
}
