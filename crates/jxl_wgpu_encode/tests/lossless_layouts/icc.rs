use super::*;
use jxl_gpu_formats::{Channel, ColorModel, ColorSpecification, PackingFieldKind, Swizzle};
use jxl_gpu_protocol::icc::IccProfile;
use jxl_test_support::oracles::icc_profile::IccProfileOracle;
use jxl_wgpu_encode::{AlphaAssociation, ImageOptions};

mod animation;
mod lifetime;
mod output;
mod profiles;

fn profile(gray: bool) -> IccProfile {
    let directory = jxl_test_support::fixtures::embedded_icc::directory();
    IccProfile::parse(
        std::fs::read(directory.join(if gray { "gray.icc" } else { "rgb.icc" }))
            .unwrap()
            .into(),
        Default::default(),
    )
    .unwrap()
}

fn encoder(
    rig: &Rig,
    profile: &IccProfile,
    tree: LosslessModularTreeMode,
) -> LosslessModularEncoder {
    LosslessModularEncoder::with_tree_mode(rig.context.clone(), tree)
        .with_image_options(ImageOptions {
            rendering_intent: profile.header().rendering_intent,
            ..Default::default()
        })
        .unwrap()
}

fn attach(source: &mut BufferImageSource, profile: &IccProfile, device: bool) {
    let format = &mut source.layout.format;
    if format.model == ColorModel::NonColor {
        format.model = ColorModel::Gray;
        format.swizzle = Swizzle::X001;
    }
    if device {
        let Swizzle::Xyzw(swizzle) = format.swizzle else {
            panic!("source component swizzle")
        };
        let color_count = profile.header().device_space.device_channels().unwrap();
        for field in format
            .planes
            .iter_mut()
            .flat_map(|plane| &mut plane.words)
            .flat_map(|word| &mut word.fields)
        {
            let PackingFieldKind::Channel(channel) = &mut field.kind else {
                continue;
            };
            let physical = match *channel {
                Channel::X => jxl_gpu_formats::SwizzleComponent::X,
                Channel::Y => jxl_gpu_formats::SwizzleComponent::Y,
                Channel::Z => jxl_gpu_formats::SwizzleComponent::Z,
                Channel::W => jxl_gpu_formats::SwizzleComponent::W,
                _ => panic!("physical source component"),
            };
            *channel = if swizzle[3] == physical {
                Channel::Alpha
            } else {
                Channel::Device(
                    (0..color_count)
                        .find(|&index| swizzle[usize::from(index)] == physical)
                        .unwrap(),
                )
            };
        }
        format.model = ColorModel::IccDevice;
        format.swizzle = Swizzle::Device;
    }
    format.color_spec = ColorSpecification::Icc(profile.clone());
}

fn original(encoded: &[u8]) -> Vec<f32> {
    extra_channels::libjxl_output(
        encoded,
        &["--original-icc", "--preserve-alpha", "--keep-orientation"],
    )
    .expect("required native original ICC component oracle")
}

#[test]
fn embedded_icc_keeps_original_profile_and_words_across_layouts() {
    let rig = Rig::new();
    let native_profile = IccProfileOracle::compile();
    for format in [
        LosslessModularFormat::Gray,
        LosslessModularFormat::GrayAlpha,
        LosslessModularFormat::Rgb,
        LosslessModularFormat::Rgba,
    ] {
        let profile = profile(format.color_channel_count() == 1);
        for tree in TREES {
            let encoder =
                encoder(&rig, &profile, tree).with_alpha_association(if format.has_alpha() {
                    AlphaAssociation::Associated
                } else {
                    AlphaAssociation::Unassociated
                });
            for (kind, bits) in [
                (SampleKind::Unsigned, 8),
                (SampleKind::Unsigned, 31),
                (SampleKind::Float, 16),
                (SampleKind::Float, 32),
            ] {
                let case = Case {
                    format,
                    bits,
                    kind,
                    storage: Storage::Planar,
                    reversed: true,
                    byte_order: ByteOrder::Big,
                    shifted: true,
                };
                let extent = Extent2d::new(257, 3);
                let samples = case.samples(extent);
                let mut source = upload(&rig.context, &case, extent, &samples, 4099);
                attach(&mut source, &profile, false);
                let plan = encoder.memory_plan(&source).unwrap();
                assert_eq!(plan.icc_profile_bytes, profile.bytes().len() as u64);
                assert!(plan.icc_storage_bytes > 2 * plan.icc_profile_bytes);
                let encoded = encoder.encode_container(source).unwrap();
                let mut canonical = upload(&rig.context, &case.canonical(), extent, &samples, 0);
                attach(&mut canonical, &profile, true);
                assert_eq!(encoded, encoder.encode_container(canonical).unwrap());
                let inventory = jxl_gpu_bitstream::parse(&encoded, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                assert_eq!(
                    inventory
                        .image_header
                        .embedded_icc
                        .unwrap()
                        .profile
                        .as_ref(),
                    profile.bytes().as_ref()
                );
                assert_eq!(
                    native_profile.read(&encoded).profile,
                    profile.bytes().as_ref()
                );
                check_frame_samples(&encoded, &[&samples], &case, &original(&encoded));
                color::check_numeric(&rig, &encoded, &[samples], &case);
                assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}
