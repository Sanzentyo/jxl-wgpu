//! Reuse native LF entropy with component-reference patch dictionaries in the LF producers.
use std::path::Path;

#[allow(dead_code)]
#[path = "../tests/common/extra_channel_oracle.rs"]
mod native;
#[allow(dead_code)]
#[path = "support/offline.rs"]
mod offline;
#[allow(dead_code)]
#[path = "../tests/support/patches.rs"]
mod patches;

fn main() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = data.join("patches/lf_producers");
    std::fs::create_dir_all(&output).unwrap();
    for name in [
        "vardct_gab0",
        "modular_gab1",
        "nested_vardct_gab1",
        "nested_modular_gab1",
    ] {
        let source = offline::unhex(
            &std::fs::read_to_string(data.join(format!("lf_extra_channels/{name}.jxl.hex")))
                .unwrap(),
        );
        let inventory = jxl_gpu_bitstream::parse(&source, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let native_source = native::libjxl_output(
            &source,
            &["--linear", "--preserve-alpha", "--keep-orientation"],
        )
        .expect("native libjxl source");
        let mut native_patched = None;
        let variants = [("", 16), ("_empty", 0), ("_unused", 16)]
            .into_iter()
            .chain(name.contains("vardct").then_some(("_padded", 1)));
        for (suffix, count) in variants {
            let dictionaries: Vec<_> = inventory.frames[..inventory.frames.len() - 1]
                .iter()
                .map(|frame| {
                    let mut coded = frame.clone();
                    (coded.width, coded.height) = frame.color_sample_extent().unwrap();
                    if suffix == "_padded" {
                        assert_eq!(frame.encoding, jxl_gpu_bitstream::FrameEncoding::VarDct);
                        // A two-by-two replacement straddles the last visible row and column.
                        // Its destination remains inside VarDCT's padded block rectangle.
                        return vec![
                            1,
                            3,
                            0,
                            0,
                            1,
                            1,
                            0,
                            coded.width - 1,
                            coded.height - 1,
                            1,
                            1,
                            1,
                        ];
                    }
                    patches::values(&coded, inventory.image_header.extra_channels.len(), count)
                })
                .collect();
            let mut encoded = patches::assemble_lf_producers(
                &source,
                &dictionaries
                    .iter()
                    .map(|values| Some(values.as_slice()))
                    .collect::<Vec<_>>(),
            );
            if suffix == "_unused" {
                let info = jxl_gpu_bitstream::parse(&encoded, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                let root = inventory.frames.len();
                let start = info.frames[root].header_bits.offset as usize / 8;
                let end = info.frames[root + 1].header_bits.offset as usize / 8;
                let duplicate = encoded[start..end].to_vec();
                encoded.splice(start..start, duplicate);
            }
            let info = jxl_gpu_bitstream::parse(&encoded, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            assert_eq!(
                info.frames.len(),
                inventory.frames.len() * 2 + usize::from(suffix == "_unused")
            );
            if suffix == "_unused" {
                let plan = jxl_wgpu_decode::FrameExecutionPlan::negotiate(&info).unwrap();
                assert!(plan.nodes[inventory.frames.len()].lf_last_use.is_none());
            }
            let final_image = native::libjxl_output(
                &encoded,
                &["--linear", "--preserve-alpha", "--keep-orientation"],
            )
            .expect("native libjxl is required to regenerate LF producer patch references");
            assert_eq!(
                final_image.len(),
                inventory.image_header.width as usize
                    * inventory.image_header.height as usize
                    * (4 + inventory.image_header.extra_channels.len())
            );
            assert!(final_image.iter().all(|value| value.is_finite()));
            match suffix {
                "" => native_patched = Some(final_image.clone()),
                "_empty" => assert_eq!(final_image, native_source),
                "_unused" => assert_eq!(Some(&final_image), native_patched.as_ref()),
                _ => {}
            }
            let name = format!("{name}{suffix}");
            std::fs::write(
                output.join(format!("{name}.jxl.hex")),
                offline::hex(&encoded),
            )
            .unwrap();
            let words: Vec<_> = final_image.into_iter().flat_map(f32::to_le_bytes).collect();
            std::fs::write(
                output.join(format!("{name}.f32.hex")),
                offline::float_hex(&words),
            )
            .unwrap();
            eprintln!(
                "{name}: {} physical frames, {} bytes",
                info.frames.len(),
                encoded.len()
            );
        }
    }
}
