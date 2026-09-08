//! Signaled noise controls that retain every entropy byte and physical frame.
use jxl_gpu_bitstream::{CodestreamInventory, FrameSectionKind};

pub fn encoded(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("test-data/noise/{name}.jxl.hex"));
    let text: String = std::fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .collect();
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

pub fn zero_noise(
    bytes: &[u8],
    inventory: &CodestreamInventory,
    physical: Option<usize>,
) -> Vec<u8> {
    // Inventory ranges address logical codestream bytes, including container-wrapped inputs.
    let mut bytes = jxl_gpu_bitstream::parse(bytes, Default::default())
        .unwrap()
        .codestream()
        .to_vec();
    let mut changed = 0;
    for (index, frame) in inventory.frames.iter().enumerate() {
        if frame.flags & 1 == 0 || physical.is_some_and(|selected| index != selected) {
            continue;
        }
        assert_eq!(frame.flags & (2 | 16), 0); // No preceding patches or splines in these fixtures.
        let section = frame
            .sections
            .iter()
            .find(|section| {
                matches!(
                    section.kind,
                    FrameSectionKind::Single | FrameSectionKind::LowFrequencyGlobal
                )
            })
            .unwrap();
        assert_eq!(section.bits.offset % 8, 0);
        let start = section.bytes.offset as usize;
        bytes[start..start + 10].fill(0);
        changed += 1;
    }
    assert!(changed > 0);
    bytes
}
