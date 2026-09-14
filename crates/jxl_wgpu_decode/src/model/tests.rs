use super::*;
use jxl_gpu_protocol::{Chromaticity, GamutMapping, RgbChromaticities, RgbColorSpace};

#[test]
fn gamut_policy_requires_a_defined_presentation_gamut() {
    let mapping = GamutMapping::default();
    let request = GpuOutputRequest::color(crate::vardct_rgb8_format()).unwrap();
    assert_eq!(request.gamut_mapping(), None);
    let mut request = request.with_gamut_mapping(mapping).unwrap();
    assert_eq!(request.gamut_mapping(), Some(mapping));
    request.frame_surface = Some(crate::frame_surface::FrameSurfaceEncoding::Rgb(
        jxl_gpu_protocol::RgbColorEncoding::SRGB_BT709,
    ));
    assert_eq!(request.gamut_mapping(), None);

    let numeric = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
        NumericSampleMapping::NativeFloat,
    )
    .unwrap();
    assert!(matches!(
        numeric.with_gamut_mapping(mapping),
        Err(Error::UnsupportedOutputFormat(_))
    ));
    let native =
        GpuOutputRequest::color(native_modular_pixel_format(ModularChannels::Rgb, 12).unwrap())
            .unwrap();
    assert!(matches!(
        native.with_gamut_mapping(mapping),
        Err(Error::UnsupportedOutputFormat(_))
    ));
    let profile = jxl_gpu_protocol::icc::IccProfile::parse(
        include_bytes!("../../../jxl_wgpu/test-data/icc/mpe/identity.icc")
            .as_slice()
            .into(),
        Default::default(),
    )
    .unwrap();
    let icc = GpuOutputRequest::color(PixelFormat::rgb_f32(
        jxl_gpu_formats::RgbChannelOrder::Rgb,
        false,
        ColorSpecification::Icc(profile),
    ))
    .unwrap();
    assert!(matches!(
        icc.with_gamut_mapping(mapping),
        Err(Error::UnsupportedOutputFormat(_))
    ));
}

#[test]
fn gamut_white_must_have_nonnegative_primary_luminances() {
    let outside = RgbColorSpace::Custom(RgbChromaticities {
        white: Chromaticity::new(0.6, 0.2).unwrap(),
        ..RgbChromaticities::BT709
    });
    assert!(jxl_wgpu::GamutMappingParams::new(outside, GamutMapping::default()).is_err());
    assert!(
        jxl_wgpu::GamutMappingParams::new(RgbColorSpace::Undefined, GamutMapping::default())
            .is_err()
    );
    let inside = RgbColorSpace::Custom(RgbChromaticities {
        white: Chromaticity::E,
        ..RgbChromaticities::BT709
    });
    assert!(jxl_wgpu::GamutMappingParams::new(inside, GamutMapping::default()).is_ok());
}
