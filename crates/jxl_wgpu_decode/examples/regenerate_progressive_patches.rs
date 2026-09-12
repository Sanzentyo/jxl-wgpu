//! Freeze native libjxl flushes of independently assembled patch-bearing pass prefixes.
use std::path::Path;

use jxl_test_support::fixtures::patches;
use jxl_test_support::offline;
use jxl_test_support::oracles::progressive as oracle;

fn main() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = data.join("patches/progressive");
    std::fs::create_dir_all(&output).unwrap();
    for (name, source) in [
        ("vardct", "vardct_extras_rgba_progressive"),
        ("gray", "testsrc_vardct_gray_progressive"),
        ("modular", "modular_passes/squeeze"),
        ("float", "floating/vardct_extras_float_squeeze"),
    ] {
        let source = offline::unhex(
            &std::fs::read_to_string(data.join(format!("{source}.jxl.hex"))).unwrap(),
        );
        let source_inventory = jxl_gpu_bitstream::parse(&source, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &source_inventory.image_header;
        let source_frame = &source_inventory.frames[0];
        assert!(source_frame.num_passes > 1);
        for (suffix, count) in [("", 16), ("_empty", 0)] {
            let name = format!("{name}{suffix}");
            let values = patches::values(source_frame, image.extra_channels.len(), count);
            let encoded = patches::assemble(&source, &values);
            let inventory = jxl_gpu_bitstream::parse(&encoded, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            assert_eq!(inventory.frames.len(), 2);
            for frame in &inventory.frames {
                assert_eq!(frame.encoding, source_frame.encoding);
                assert_eq!(frame.num_passes, source_frame.num_passes);
                assert_eq!(frame.progressive_passes, source_frame.progressive_passes);
                assert_eq!(frame.restoration_filter, source_frame.restoration_filter);
                assert_eq!(
                    frame.extra_channel_upsampling,
                    source_frame.extra_channel_upsampling
                );
            }
            let frame = inventory.frames.last().unwrap();
            let mut snapshots = Vec::new();
            for completed in 0..=frame.num_passes {
                let end =
                    frame
                        .sections
                        .iter()
                        .filter_map(|section| match section.kind {
                            jxl_gpu_bitstream::FrameSectionKind::PassGroup {
                                pass_index, ..
                            } if pass_index >= completed => Some(section.bytes.offset as usize),
                            _ => None,
                        })
                        .min()
                        .unwrap_or(encoded.len());
                let updates = oracle::native_updates_all_channels(
                    &encoded[..end],
                    image.xyb_encoded,
                    false,
                    completed < frame.num_passes,
                )
                .expect("native libjxl is required to regenerate conformance snapshots");
                let last = updates.last().expect("native prefix output");
                assert_eq!(last.complete, completed == frame.num_passes);
                assert_eq!(
                    last.pixels.len(),
                    image.width as usize
                        * image.height as usize
                        * (4 + image.extra_channels.len())
                        * 4
                );
                snapshots.extend_from_slice(&last.pixels);
            }
            std::fs::write(
                output.join(format!("{name}.jxl.hex")),
                offline::hex(&encoded),
            )
            .unwrap();
            std::fs::write(
                output.join(format!("{name}.f32.hex")),
                offline::float_hex(&snapshots),
            )
            .unwrap();
            eprintln!(
                "{name}: {} passes, {}x{}, {} extras, {} bytes",
                frame.num_passes,
                image.width,
                image.height,
                image.extra_channels.len(),
                encoded.len()
            );
        }
    }
}
