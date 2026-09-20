//! Pin the remaining descriptors to the independently exercised spline and CMYK families.
//! Their existing GPU tests retain the stricter original per-component pixel bounds.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

fn hash(bytes: &[u8], expected: &str) {
    assert_eq!(
        Sha256::digest(bytes).as_slice(),
        jxl_test_support::offline::hex::unhex(expected)
    );
}
fn descriptor(name: &str, expected: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test-data/official_conformance/variants")
        .join(format!("{name}.json"));
    let bytes = std::fs::read(path).unwrap();
    hash(&bytes, expected);
    serde_json::from_slice(&bytes).unwrap()
}
fn metadata_and_objects(
    descriptor: &Value,
    input: &Path,
    input_hash: &str,
    pixels: &Path,
    profile: &Path,
) {
    let data = std::fs::read(input).unwrap();
    hash(&data, input_hash);
    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let image = &inventory.image_header;
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(pixels).unwrap())
        .read_to_end(&mut bytes)
        .unwrap();
    hash(
        &bytes,
        descriptor["sha256sums"]["reference_image.npy"]
            .as_str()
            .unwrap(),
    );
    let profile = std::fs::read(profile).unwrap();
    hash(
        &profile,
        descriptor["sha256sums"]["reference.icc"].as_str().unwrap(),
    );
    if descriptor.get("original_icc").is_some() {
        hash(
            &profile,
            descriptor["sha256sums"][descriptor["original_icc"].as_str().unwrap()]
                .as_str()
                .unwrap(),
        );
        assert_eq!(
            image.embedded_icc.as_ref().unwrap().profile.as_ref(),
            profile
        );
        assert_eq!(
            image
                .original_icc_profile(Default::default())
                .unwrap()
                .as_ref(),
            profile
        );
    }
    assert_eq!(
        image.tone_mapping.intensity_target.to_f32(),
        descriptor["intensity_target"].as_f64().unwrap() as f32
    );
    assert_eq!(
        image.tone_mapping.min_nits.to_f32(),
        descriptor["min_nits"].as_f64().unwrap() as f32
    );
    assert_eq!(
        u64::from(image.tone_mapping.relative_to_max_display),
        descriptor["relative_to_max_display"].as_u64().unwrap()
    );
    assert_eq!(
        image.tone_mapping.linear_below.to_f32(),
        descriptor["linear_below"].as_f64().unwrap() as f32
    );
    let depths: Vec<_> = std::iter::once(image.bit_depth)
        .chain(image.extra_channels.iter().map(|channel| channel.bit_depth))
        .collect();
    for (index, depth) in depths.iter().enumerate() {
        assert!(
            matches!(depth, jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample }
            if u64::from(*bits_per_sample) == descriptor["bits_per_sample"][index].as_u64().unwrap())
        );
        assert_eq!(descriptor["exp_bits_per_sample"][index], 0);
    }
    assert_eq!(
        depths.len(),
        descriptor["bits_per_sample"].as_array().unwrap().len()
    );
    let extras: Vec<_> = image
        .extra_channels
        .iter()
        .map(|channel| match channel.channel_type {
            jxl_gpu_bitstream::ExtraChannelTypeInventory::Black => "Black",
            jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { .. } => "Alpha",
            _ => panic!("unclassified existing family extra"),
        })
        .collect();
    assert_eq!(
        serde_json::to_value(extras).unwrap(),
        descriptor["extra_channel_type"]
    );
    let visible: Vec<_> = inventory
        .frames
        .iter()
        .filter(|frame| {
            frame.frame_type == jxl_gpu_bitstream::FrameType::Regular
                && (frame.duration_ticks != 0 || frame.is_last)
        })
        .collect();
    let frames = descriptor["frames"].as_array().unwrap();
    assert_eq!(visible.len(), frames.len());
    for (frame, expected) in visible.iter().zip(frames) {
        assert_eq!(
            frame.name_bytes,
            expected["name"].as_str().unwrap().as_bytes()
        );
        if let Some(seconds) = expected.get("duration") {
            let animation = image.animation.unwrap();
            let actual = f64::from(frame.duration_ticks)
                * f64::from(animation.ticks_per_second_denominator)
                / f64::from(animation.ticks_per_second_numerator);
            assert_eq!(actual, seconds.as_f64().unwrap());
        }
    }
}

#[test]
fn spline_alternate_shares_the_existing_stricter_gpu_reference() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut original = descriptor(
        "animation_spline",
        "8f54bb412f1081378aa86b842fde4b38f12c33bae180870701bfb982f8cc82dd",
    );
    let mut alternate = descriptor(
        "animation_spline_5",
        "1d6d6e11276d602c020dddde6174009970df3b94df8cfacbacea776ebb2a1818",
    );
    metadata_and_objects(
        &original,
        &root.join("../../fixtures/animation_spline.jxl"),
        "87793cac33d05eaa380011e3b0754ff6f228967431a126fd0a0ace2106940c79",
        &root.join("test-data/splines/animation_spline.npy.gz"),
        &root.join("test-data/splines/animation_spline.icc"),
    );
    let frames = original["frames"].as_array_mut().unwrap();
    assert_eq!(frames.len(), 60);
    let alternates = alternate["frames"].as_array_mut().unwrap();
    assert_eq!(frames.len(), alternates.len());
    for (frame, alternate) in frames.iter_mut().zip(alternates) {
        assert_eq!(frame["rms_error"], 0.0001);
        assert_eq!(frame["peak_error"], 0.004);
        assert_eq!(alternate["rms_error"], 0.02);
        assert_eq!(alternate["peak_error"], 0.06);
        for key in ["rms_error", "peak_error"] {
            frame.as_object_mut().unwrap().remove(key);
            alternate.as_object_mut().unwrap().remove(key);
        }
    }
    assert_eq!(
        original, alternate,
        "same frames, metadata, original reference hashes"
    );
}

#[test]
fn cmyk_descriptor_matches_the_existing_five_channel_gpu_reference() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/cmyk");
    let descriptor = descriptor(
        "cmyk_layers",
        "f8faed062a3e6d1613f755a2a802eaee5d6f2dc50510c8021e60df600df29c2f",
    );
    metadata_and_objects(
        &descriptor,
        &root.join("layers.jxl"),
        "d732c8836bf1abeadf310d2e07387a32813ed4690d32650c1c25e541b80eed4a",
        &root.join("layers.npy.gz"),
        &root.join("layers.icc"),
    );
    let frames = descriptor["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0]["rms_error"], 0.000976562);
    assert_eq!(frames[0]["peak_error"], 0.000976562);
}
