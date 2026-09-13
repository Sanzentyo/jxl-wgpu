use super::profile;
use jxl_gpu_formats::{
    ColorFormatClass, ColorModel, ColorSample, ColorSpecification, ColorStorage, ImageLayout,
    PixelFormat, PixelFormatError, RgbChannelOrder, classify_pixel_format,
};
use jxl_gpu_protocol::{Extent2d, icc::IccSignature};

#[test]
fn layouts_own_exact_profiles_and_share_payloads_across_clones() {
    for (name, gray) in [("srgb", false), ("sampled", false), ("gray", true)] {
        let original = profile(name);
        let bytes = std::sync::Arc::clone(original.bytes());
        let profile = ColorSpecification::Icc(original.clone());
        let format = if gray {
            PixelFormat::gray_f32(true, true, profile)
        } else {
            PixelFormat::rgb_f32(RgbChannelOrder::Rgba, true, profile)
        };
        let layout = ImageLayout::packed(Extent2d::new(37, 17), format).unwrap();
        let cloned = layout.clone();
        assert_eq!(layout, cloned);
        let ColorSpecification::Icc(copy) = &cloned.format.color_spec else {
            panic!("lost ICC ownership")
        };
        assert!(std::sync::Arc::ptr_eq(&bytes, copy.bytes()));
        assert!(std::ptr::eq(original.tags(), copy.tags()));
        drop(original);
        drop(layout);
        assert_eq!(&**copy.bytes(), &*bytes);
        let expected = if gray {
            ColorFormatClass::Gray {
                sample: ColorSample::F32,
                storage: ColorStorage::Planar,
                alpha: true,
            }
        } else {
            ColorFormatClass::Rgb {
                sample: ColorSample::F32,
                storage: ColorStorage::Planar,
                order: RgbChannelOrder::Rgba,
            }
        };
        assert_eq!(
            classify_pixel_format(&cloned.format).unwrap().color(),
            Some(expected)
        );
    }
}

#[test]
fn profiles_cannot_relabel_numeric_ycbcr_or_incompatible_device_channels() {
    let gray = ColorSpecification::Icc(profile("gray"));
    let rgb = ColorSpecification::Icc(profile("srgb"));
    for (format, signature) in [
        (
            PixelFormat::rgb8(RgbChannelOrder::Rgb, false, gray.clone()),
            IccSignature(*b"GRAY"),
        ),
        (
            PixelFormat::gray8(false, false, rgb.clone()),
            IccSignature(*b"RGB "),
        ),
        (PixelFormat::nv12(rgb.clone()), IccSignature(*b"RGB ")),
        (
            {
                let mut f = PixelFormat::non_color(
                    jxl_gpu_formats::SampleKind::Float,
                    32,
                    &[jxl_gpu_formats::Channel::X],
                );
                f.color_spec = gray;
                f
            },
            IccSignature(*b"GRAY"),
        ),
    ] {
        let expected = PixelFormatError::IccDeviceSpace {
            model: format.model,
            signature,
        };
        assert_eq!(format.validate(), Err(expected));
        assert!(classify_pixel_format(&format).is_err());
        assert!(ImageLayout::packed(Extent2d::new(2, 3), format).is_err());
    }
    let format = PixelFormat::gray8(false, false, ColorSpecification::Icc(profile("gray")));
    assert_eq!(format.model, ColorModel::Gray);
    assert!(classify_pixel_format(&format).unwrap().numeric().is_none());
    // Fixed YCbCr constructors retain metadata without a panic; the complete descriptor
    // must still fail color validation, classification and layout construction.
    for format in [
        PixelFormat::nv21(rgb.clone()),
        PixelFormat::nv24(rgb.clone()),
        PixelFormat::nv42(rgb.clone()),
        PixelFormat::nv16(rgb.clone()),
        PixelFormat::nv61(rgb.clone()),
        PixelFormat::p010(rgb.clone()),
        PixelFormat::p012(rgb.clone()),
        PixelFormat::p016(rgb.clone()),
        PixelFormat::p210(rgb.clone()),
        PixelFormat::p212(rgb.clone()),
        PixelFormat::p216(rgb.clone()),
        PixelFormat::p410(rgb.clone()),
        PixelFormat::p412(rgb.clone()),
        PixelFormat::p416(rgb.clone()),
        PixelFormat::i444(8, 8, rgb.clone()).unwrap(),
        PixelFormat::i422(10, 16, rgb.clone()).unwrap(),
        PixelFormat::i420(12, 16, rgb).unwrap(),
    ] {
        assert!(matches!(
            format.validate(),
            Err(PixelFormatError::IccDeviceSpace { .. })
        ));
        assert!(classify_pixel_format(&format).is_err());
        assert!(ImageLayout::packed(Extent2d::new(2, 3), format).is_err());
    }
}

#[test]
fn enumerated_packers_reject_an_unexecuted_icc_target_before_submission() {
    let format = PixelFormat::rgb_f32(
        RgbChannelOrder::Rgb,
        true,
        ColorSpecification::Icc(profile("sampled")),
    );
    let layout = ImageLayout::packed(Extent2d::new(2, 3), format).unwrap();
    assert!(matches!(jxl_wgpu::ImageOutputParams::new(
        &layout,
        jxl_wgpu::ImageOutputSource {
            extent: layout.extent,
            orientation: jxl_gpu_protocol::OutputOrientation::Identity,
            strides: [2; 3],
            encoding: jxl_gpu_protocol::RgbColorEncoding::LINEAR_BT709,
        },
        1,
        jxl_gpu_protocol::WhitePointAdaptation::Bradford,
    ), Err(jxl_wgpu::Error::Unsupported(message)) if message.contains("ICC")));
}
