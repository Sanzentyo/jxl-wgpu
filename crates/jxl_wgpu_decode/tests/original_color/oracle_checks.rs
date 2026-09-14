//! Test-only references: jxl-oxide and f64 colorimetry derived from primary chromaticities.
use super::oracle::{from_linear, xyb_original_linear};
use super::{corpus, tolerance};
use jxl_gpu_formats::TransferFunction;

#[test]
fn native_still_references_agree_with_independent_decoder() {
    check_still_references(corpus::cases());
}

#[test]
fn analytic_native_still_references_agree_with_independent_decoder() {
    check_still_references(corpus::analytic_cases());
}

fn check_still_references(cases: Vec<corpus::Case>) {
    use jxl_oxide::color::{ColourSpace, Customxy, Primaries, TransferFunction as Tf, WhitePoint};
    let xy = |p: jxl_gpu_bitstream::ChromaticityInventory| Customxy { x: p.x, y: p.y };
    // jxl-frame 0.13.3 reads an extra-channel source selector using the color blend mode.
    // These sequences use full-canvas color Mul/Blend with alpha Replace, where that selector
    // is absent. libjxl and our parser use the extra's own mode; jxl-oxide loses bit alignment.
    // All 148 originals still have native references and GPU coverage; this second oracle
    // therefore covers the 74 stills. See the generator README for the precise upstream sites.
    for case in cases.into_iter().filter(|case| !case.sequence) {
        let mut image = jxl_oxide::JxlImage::read_with_defaults(case.bytes().as_slice()).unwrap();
        image.set_render_spot_color(false);
        image.request_color_encoding(jxl_oxide::EnumColourEncoding {
            colour_space: if case.profile.grayscale {
                ColourSpace::Grey
            } else {
                ColourSpace::Rgb
            },
            white_point: match case.profile.white {
                jxl_gpu_bitstream::WhitePointInventory::D65 => WhitePoint::D65,
                jxl_gpu_bitstream::WhitePointInventory::E => WhitePoint::E,
                jxl_gpu_bitstream::WhitePointInventory::Dci => WhitePoint::Dci,
                jxl_gpu_bitstream::WhitePointInventory::Custom(p) => WhitePoint::Custom(xy(p)),
            },
            primaries: match case.profile.primaries {
                jxl_gpu_bitstream::PrimariesInventory::Srgb => Primaries::Srgb,
                jxl_gpu_bitstream::PrimariesInventory::Bt2100 => Primaries::Bt2100,
                jxl_gpu_bitstream::PrimariesInventory::P3 => Primaries::P3,
                jxl_gpu_bitstream::PrimariesInventory::Custom { red, green, blue } => {
                    Primaries::Custom {
                        red: xy(red),
                        green: xy(green),
                        blue: xy(blue),
                    }
                }
            },
            tf: match case.transfer.transfer {
                jxl_gpu_bitstream::TransferFunctionInventory::Linear => Tf::Linear,
                jxl_gpu_bitstream::TransferFunctionInventory::Srgb => Tf::Srgb,
                jxl_gpu_bitstream::TransferFunctionInventory::Bt709 => Tf::Bt709,
                jxl_gpu_bitstream::TransferFunctionInventory::Dci => Tf::Dci,
                jxl_gpu_bitstream::TransferFunctionInventory::Gamma {
                    scaled_gamma,
                    inverted,
                } => Tf::Gamma {
                    g: scaled_gamma,
                    inverted,
                },
                _ => unreachable!(),
            },
            rendering_intent: jxl_oxide::RenderingIntent::Relative,
        });
        // jxl-color clips or gamut maps before a primary/Gray conversion. Ask it for the
        // unbounded XYB intermediate, then perform the reference's requested conversion in f64.
        if case.mode.xyb() {
            image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));
        }
        let expected = case.reference();
        assert_eq!(
            image.num_loaded_keyframes(),
            if case.sequence { 4 } else { 1 }
        );
        for frame in 0..image.num_loaded_keyframes() {
            let render = image.render_frame(frame).unwrap();
            let pixels = render.image_all_channels();
            assert!(pixels.channels() == 4 || (case.profile.grayscale && pixels.channels() == 2));
            let mut words: Vec<_> = if pixels.channels() == 2 {
                pixels
                    .buf()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|p| [p[0], p[0], p[0], p[1]])
                    .map(f32::to_bits)
                    .collect()
            } else {
                pixels.buf().iter().map(|v| v.to_bits()).collect()
            };
            if case.mode.xyb() {
                let jxl_gpu_formats::ColorSpecification::Defined(target) = case.format().color_spec
                else {
                    unreachable!()
                };
                for pixel in words.as_chunks_mut::<4>().0 {
                    let rgb = [pixel[0], pixel[1], pixel[2]].map(|v| f64::from(f32::from_bits(v)));
                    let linear = xyb_original_linear(rgb, &case);
                    for (channel, value) in linear.into_iter().enumerate() {
                        // The native JPEG XL original OETF has a black floor, unlike a general CMS conversion.
                        let value = if matches!(
                            target.transfer,
                            TransferFunction::Gamma(_) | TransferFunction::Dci
                        ) && value <= 1e-5
                        {
                            0.0
                        } else {
                            value
                        };
                        pixel[channel] = (from_linear(value, target.transfer) as f32).to_bits();
                    }
                }
            }
            super::compare(
                &words,
                &expected[frame * 37 * 19 * 4..(frame + 1) * 37 * 19 * 4],
                tolerance(&case),
                &format!("oxide {}", case.name),
            );
        }
    }
}
