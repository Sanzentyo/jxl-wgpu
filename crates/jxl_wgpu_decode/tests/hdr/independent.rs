use super::{backend, compare, corpus, planes};
use jxl_gpu_formats::{ColorSpace, TransferFunction};
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuDecoder, GpuOutputRequest};

#[test]
fn stills_match_independent_oxide_components_without_automatic_tone_mapping() {
    use jxl_oxide::color::{ColourSpace, Primaries, TransferFunction as Tf, WhitePoint};
    let backend = backend();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let mut count = 0;
    for case in corpus::cases().into_iter().filter(|case| !case.sequence) {
        let data = case.bytes();
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let mut image = jxl_oxide::JxlImage::read_with_defaults(data.as_slice()).unwrap();
        image.set_render_spot_color(false);
        // Request raw XYB, avoiding jxl-color's implicit Rec.2408 tone mapping
        // for >255-nit sources when a linear RGB target is selected.
        image.request_color_encoding(jxl_oxide::EnumColourEncoding {
            colour_space: if case.xyb {
                ColourSpace::Xyb
            } else if case.gray {
                ColourSpace::Grey
            } else {
                ColourSpace::Rgb
            },
            white_point: WhitePoint::D65,
            primaries: match case.space {
                ColorSpace::Bt709 => Primaries::Srgb,
                ColorSpace::Bt2020 => Primaries::Bt2100,
                ColorSpace::DisplayP3 => Primaries::P3,
                _ => unreachable!(),
            },
            tf: match case.transfer {
                TransferFunction::Pq => Tf::Pq,
                TransferFunction::Hlg => Tf::Hlg,
                _ => unreachable!(),
            },
            rendering_intent: jxl_oxide::RenderingIntent::Relative,
        });
        let render = image.render_frame(0).unwrap();
        let pixels = render.image_all_channels();
        let mut expected = Vec::new();
        if case.xyb {
            assert_eq!(pixels.channels(), 4);
            let opsin = inventory.image_header.opsin_inverse_matrix.unwrap();
            let bias = opsin.opsin_bias.map(|v| f64::from(v.to_f32()));
            let matrix = opsin
                .inverse_matrix
                .map(|row| row.map(|v| f64::from(v.to_f32())));
            for pixel in pixels.buf().as_chunks::<4>().0 {
                let [x, y, b] = [pixel[0], pixel[1], pixel[2]].map(f64::from);
                let mixed = [y + x, y - x, b];
                let lms: [f64; 3] = std::array::from_fn(|c| {
                    ((mixed[c] - bias[c].cbrt()).powi(3) + bias[c]) * 255.0 / case.nits
                });
                let rgb = matrix.map(|row| (0..3).map(|c| row[c] * lms[c]).sum());
                let linear = jxl_test_support::oracles::color::xyb_original_linear_for_profile(
                    rgb, case.space, case.gray,
                );
                expected.extend(linear.map(|v| v as f32));
                expected.push(pixel[3]);
            }
        } else if case.gray {
            assert_eq!(pixels.channels(), 2);
            for pixel in pixels.buf().as_chunks::<2>().0 {
                expected.extend([pixel[0], pixel[0], pixel[0], pixel[1]]);
            }
        } else {
            assert_eq!(pixels.channels(), 4);
            expected.extend_from_slice(pixels.buf());
        }
        let transfer = if case.xyb {
            TransferFunction::Linear
        } else {
            case.transfer
        };
        let request = GpuOutputRequest::color(case.format(transfer, case.space))
            .unwrap()
            .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
        let mut session = decoder.open(&data, request).unwrap();
        let frame = session.next_frame().unwrap().unwrap();
        compare(
            &planes::read(&backend, &frame.output().outputs[0]),
            &expected,
            if case.xyb { 1e-4 } else { case.tolerance() },
            &format!("{} oxide", case.name),
        );
        assert!(session.next_frame().unwrap().is_none());
        drop((frame, session));
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        count += 1;
    }
    assert_eq!(count, 48);
}
