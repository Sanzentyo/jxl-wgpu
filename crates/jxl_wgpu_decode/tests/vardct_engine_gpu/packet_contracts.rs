use std::sync::atomic::Ordering;

use jxl_gpu_bitstream::{BitWriter, CodestreamInventory};
use jxl_test_support::fixtures::frame_features::copy_bits;
use jxl_wgpu_decode::vardct::packet::{BoundedVarDctGroupEntry, BoundedVarDctPacketPlan};
use jxl_wgpu_encode::{
    BitFragment, FrameGroupLayout, FramePacketSet, GroupPacket, GroupPacketKind, assemble_frame,
};

pub(super) fn truncate_lf_coefficients(
    encoded: &[u8],
    inventory: &CodestreamInventory,
    plan: &BoundedVarDctPacketPlan,
) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(encoded, Default::default()).unwrap();
    let data = parsed.codestream();
    assert_eq!(inventory.frames.len(), 1);
    let frame = &inventory.frames[0];
    assert!(!frame.toc_permuted);
    assert_eq!(frame.sections.len(), 1);
    assert_eq!(frame.header_bits.offset % 8, 0);
    let group = &plan.groups[0];
    let BoundedVarDctGroupEntry::LfCoefficients {
        token_bit_offset, ..
    } = group.entry
    else {
        panic!("expected a combined packet beginning with LF coefficient entropy");
    };
    // Arbitrary XOR changes need not invalidate a prefix stream: both independent decoders
    // accepted the previous corruption. Truncate at a known entropy boundary instead.
    // Keep the complete descriptors and at most fifteen coefficient bits. The encoded
    // 4x4 LF grid needs 48 samples from a 33-symbol prefix tree, so this cannot finish.
    assert_eq!(group.padded_block_extent, [4, 4]);
    let cut = u64::from(token_bit_offset).div_ceil(8) + 1;
    let section = &frame.sections[0].bytes;
    assert!(section.offset < cut && cut < section.end().unwrap());
    let mut header = BitWriter::new();
    copy_bits(
        &mut header,
        data,
        frame.header_bits.offset,
        frame.header_bits.end().unwrap(),
    );
    let packet = GroupPacket::new(
        GroupPacketKind::Single,
        data[section.offset as usize..cut as usize].to_vec(),
    );
    let replacement = assemble_frame(
        FramePacketSet::new(
            BitFragment::new(header.into_bytes(), frame.header_bits.length as usize).unwrap(),
            FrameGroupLayout::new(1, 1, 1).unwrap(),
            [packet],
        )
        .unwrap(),
    )
    .unwrap()
    .into_bytes();
    let mut truncated = data[..frame.header_bits.offset as usize / 8].to_vec();
    truncated.extend_from_slice(&replacement);
    // Rebuild the TOC so the container and declared section ranges remain valid. The
    // failure must be GPU entropy exhaustion, rather than a host input-length error.
    let inspected = jxl_gpu_bitstream::parse(&truncated, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let short = BoundedVarDctPacketPlan::parse(&truncated, &inspected).unwrap();
    assert!(short.requires_lf_staging());
    assert_eq!(
        short.groups[0].padded_block_extent,
        group.padded_block_extent
    );
    truncated
}

pub(super) fn assert_native_rejection(encoded: &[u8]) {
    if std::process::Command::new("djxl")
        .arg("--version")
        .output()
        .is_err()
    {
        assert!(
            std::env::var_os("JXL_REQUIRE_NATIVE_ORACLES").is_none(),
            "djxl is required"
        );
        eprintln!("native malformed-packet check unavailable: djxl is not installed");
        return;
    }
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        super::DJXL_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    );
    let input = std::env::temp_dir().join(format!("jxl-wgpu-truncated-lf-{nonce}.jxl"));
    let output = input.with_extension("pfm");
    std::fs::write(&input, encoded).unwrap();
    let result = std::process::Command::new("djxl")
        .args([&input, &output])
        .output()
        .unwrap();
    std::fs::remove_file(input).unwrap();
    if output.exists() {
        std::fs::remove_file(output).unwrap();
    }
    assert!(
        !result.status.success(),
        "djxl accepted truncated LF coefficients"
    );
    eprintln!(
        "djxl rejects truncated LF coefficients: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}
