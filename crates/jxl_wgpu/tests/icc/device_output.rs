use super::*;
use jxl_gpu_formats::{
    Channel, ColorFormatClass, ColorModel, ColorSample, ColorSpecification, ColorStorage,
    ImageLayout, PackingFieldKind, PixelFormat, PixelFormatClass, Swizzle, classify_pixel_format,
};
use jxl_gpu_protocol::OutputOrientation;
use jxl_wgpu::{
    AlphaConversion, DEVICE_OUTPUT_SHADER, DeviceOutputParams, DeviceOutputSource,
    ResidentIccSampleEncoding,
};

mod packing;

const PROFILES: [(&str, u8); 6] = [
    ("lut/lut16_xyz_1", 1),
    ("lut/lut16_channels2", 2),
    ("lut/ab_xyz_3", 3),
    ("lut/ab_lab_4", 4),
    ("lut/ab_channels5", 5),
    ("lut/ab_channels15", 15),
];

#[test]
fn device_formats_own_each_profile_component_and_keep_alpha_independent() {
    for (name, channels) in PROFILES {
        let profile = profile(name);
        assert_eq!(
            profile.header().device_space.device_channels(),
            Some(channels)
        );
        for sample in [ColorSample::U8, ColorSample::F32] {
            for storage in [ColorStorage::Interleaved, ColorStorage::Planar] {
                for alpha in [false, true] {
                    let mut format =
                        PixelFormat::icc_device(profile.clone(), sample, storage, alpha).unwrap();
                    let expected = PixelFormatClass::Color(ColorFormatClass::IccDevice {
                        sample,
                        storage: if channels == 1 && !alpha {
                            ColorStorage::Interleaved
                        } else {
                            storage
                        },
                        channels,
                        alpha,
                    });
                    assert_eq!(classify_pixel_format(&format).unwrap(), expected);
                    assert_eq!(format.model, ColorModel::IccDevice);
                    assert_eq!(format.swizzle, Swizzle::Device);
                    assert_eq!(format.color_spec, ColorSpecification::Icc(profile.clone()));
                    if storage == ColorStorage::Planar {
                        format.planes.reverse();
                    } else {
                        format.planes[0].words.reverse();
                    }
                    assert_eq!(classify_pixel_format(&format).unwrap(), expected);
                    ImageLayout::packed(Extent2d::new(17, 9), format).unwrap();
                }
            }
        }
    }
    let original = PixelFormat::icc_device(
        profile("lut/ab_lab_4"),
        ColorSample::F32,
        ColorStorage::Interleaved,
        true,
    )
    .unwrap();
    for channel in [
        Channel::Device(0),
        Channel::Device(4),
        Channel::Alpha,
        Channel::W,
    ] {
        let mut bad = original.clone();
        bad.planes[0].words[1].fields[0].kind = PackingFieldKind::Channel(channel);
        assert!(matches!(
            classify_pixel_format(&bad),
            Err(jxl_gpu_formats::PixelFormatClassificationError::UnsupportedColorPacking)
        ));
    }
    let mut missing = original.clone();
    missing.planes[0].words.remove(2);
    assert!(matches!(
        classify_pixel_format(&missing),
        Err(jxl_gpu_formats::PixelFormatClassificationError::UnsupportedColorPacking)
    ));
    let mut unprofiled = original;
    unprofiled.color_spec = ColorSpecification::Undefined;
    assert!(matches!(
        unprofiled.validate(),
        Err(jxl_gpu_formats::PixelFormatError::IccProfileRequired)
    ));
}

#[test]
fn device_packing_shader_and_uniform_are_portable() {
    let module = naga::front::wgsl::parse_str(DEVICE_OUTPUT_SHADER).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let (_, params) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("DeviceOutputParams"))
        .unwrap();
    let naga::TypeInner::Struct { members, span } = &params.inner else {
        panic!("not a struct")
    };
    assert_eq!(*span, std::mem::size_of::<DeviceOutputParams>() as u32);
    assert_eq!(
        members.iter().map(|m| m.offset).collect::<Vec<_>>(),
        [0, 16, 32, 48, 64, 128, 192, 256]
    );
}
