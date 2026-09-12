//! Preserve native entropy while composing patches before upsampling and noise.

use jxl_test_support::fixtures::noise;
use jxl_test_support::fixtures::patch_features as fixtures;
use jxl_test_support::fixtures::patches;
use jxl_test_support::offline;
use jxl_test_support::oracles::extra_channels as native;

fn main() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = data.join("patches/features");
    std::fs::create_dir_all(&output).unwrap();
    for family in fixtures::FAMILIES {
        let original = offline::unhex(
            &std::fs::read_to_string(data.join(format!("{}.jxl.hex", family.source))).unwrap(),
        );
        let info = inventory(&original);
        eprintln!(
            "{}: {:?}",
            family.name,
            info.frames
                .iter()
                .map(|f| (
                    f.encoding,
                    f.lf_level,
                    f.upsampling,
                    f.extra_channel_upsampling.clone(),
                    f.flags
                ))
                .collect::<Vec<_>>()
        );
        for suffix in family.suffixes() {
            let source = if suffix.contains("noise") || (suffix == "_chain" && family.inject_noise)
            {
                patches::with_noise(&original, family.lf)
            } else if suffix == "_zero" {
                noise::zero_noise(&original, &info, None)
            } else {
                original.clone()
            };
            let count = if suffix.ends_with("empty") { 0 } else { 16 };
            let dictionary = |frame: &jxl_gpu_bitstream::FrameInventory| {
                assert!(
                    frame.upsampling == 1
                        || frame
                            .extra_channel_upsampling
                            .iter()
                            .all(|&factor| factor == frame.upsampling)
                );
                let mut coded = frame.clone();
                (coded.width, coded.height) = frame.color_sample_extent().unwrap();
                if suffix.starts_with("_padded") {
                    assert_eq!(frame.encoding, jxl_gpu_bitstream::FrameEncoding::VarDct);
                    let mut values = vec![1, 3, 0, 0, 1, 1, 0, coded.width - 1, coded.height - 1];
                    values.extend(std::iter::repeat_n(
                        1,
                        1 + info.image_header.extra_channels.len(),
                    ));
                    return values;
                }
                patches::values(&coded, info.image_header.extra_channels.len(), count)
            };
            let encoded = if family.lf {
                let dictionaries: Vec<_> = info.frames[..info.frames.len() - 1]
                    .iter()
                    .map(dictionary)
                    .collect();
                patches::assemble_lf_producers(
                    &source,
                    &dictionaries
                        .iter()
                        .map(|values| Some(values.as_slice()))
                        .collect::<Vec<_>>(),
                )
            } else {
                let values = dictionary(info.frames.last().unwrap());
                if suffix == "_chain" {
                    patches::assemble_frames(&[
                        patches::Frame {
                            codestream: &source,
                            reference: Some((3, true)),
                            patches: None,
                        },
                        patches::Frame {
                            codestream: &source,
                            reference: Some((3, true)),
                            patches: Some(&values),
                        },
                        patches::Frame {
                            codestream: &source,
                            reference: Some((3, true)),
                            patches: Some(&values),
                        },
                        patches::Frame {
                            codestream: &source,
                            reference: None,
                            patches: Some(&values),
                        },
                    ])
                } else {
                    patches::assemble(&source, &values)
                }
            };
            let parsed = inventory(&encoded);
            assert!(parsed.frames.iter().any(|frame| frame.flags & 2 != 0));
            let samples = native::libjxl_output(
                &encoded,
                &["--linear", "--preserve-alpha", "--keep-orientation"],
            )
            .expect("native libjxl must accept every patch feature fixture");
            assert_eq!(
                samples.len(),
                info.image_header.width as usize
                    * info.image_header.height as usize
                    * (4 + info.image_header.extra_channels.len())
            );
            assert!(samples.iter().all(|sample| sample.is_finite()));
            let name = format!("{}{suffix}", family.name);
            std::fs::write(
                output.join(format!("{name}.jxl.hex")),
                offline::hex(&encoded),
            )
            .unwrap();
            std::fs::write(
                output.join(format!("{name}.f32.hex")),
                offline::float_hex(
                    &samples
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>(),
                ),
            )
            .unwrap();
            eprintln!(
                "{name}: {} physical frames, {} bytes",
                parsed.frames.len(),
                encoded.len()
            );
        }
    }
}

fn inventory(bytes: &[u8]) -> jxl_gpu_bitstream::CodestreamInventory {
    jxl_gpu_bitstream::parse(bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
}
