use super::*;
use jxl_gpu_formats::{Channel, PixelFormat, SampleKind};
use jxl_test_support::fixtures::{
    hdr,
    tone_mapping::{Metadata, replace},
};

#[test]
fn tone_mapping_is_presentation_only_and_relative_threshold_uses_display_peak() {
    let case = hdr::cases().remove(0);
    let data = replace(
        &case.bytes(),
        Metadata {
            intensity_target: 1000.0,
            min_nits: 0.0625,
            relative_to_max_display: true,
            linear_below: 0.125,
        },
    );
    let image = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .image_header;
    let target = LuminanceRange::new(0.0, 80.0).unwrap();
    let color = GpuOutputRequest::color(crate::vardct_rgb8_format())
        .unwrap()
        .with_tone_mapping(target);
    let mapping = for_image(&image, &color).unwrap().unwrap();
    assert_eq!(mapping.linear_below_nits(), 10.0);
    assert_eq!(mapping.source().black_nits(), 0.0625);
    assert_eq!(mapping.source().white().nits(), 1000.0);
    let mut fractional = image.clone();
    fractional.tone_mapping.linear_below =
        jxl_gpu_bitstream::FiniteF32::from_bits(0.375f32.to_bits()).unwrap();
    let request = color
        .clone()
        .with_tone_mapping(LuminanceRange::new(0.0, 80.12345).unwrap());
    let mapping = for_image(&fractional, &request).unwrap().unwrap();
    let exact = 0.375 * f64::from(80.12345_f32);
    assert_ne!(
        exact,
        f64::from(exact as f32),
        "exercise a nonrepresentable F32 threshold"
    );
    assert_eq!(mapping.linear_below_nits(), exact);
    let surface = color.for_frame_surface(crate::frame_surface::FrameSurfaceEncoding::Rgb(
        jxl_gpu_protocol::RgbColorEncoding::LINEAR_BT709,
    ));
    assert_eq!(for_image(&image, &surface).unwrap(), None);
    let numeric = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
        crate::NumericSampleMapping::NativeFloat,
    )
    .unwrap()
    .with_tone_mapping(target);
    assert_eq!(for_image(&image, &numeric).unwrap(), None);
}
