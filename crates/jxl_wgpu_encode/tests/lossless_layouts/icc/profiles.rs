use super::*;

// Add an uninterpreted private tag while preserving shared tag offsets and original payloads.
fn private_profile(profile: &IccProfile, index: usize) -> IccProfile {
    let mut bytes = profile.bytes().to_vec();
    let count = u32::from_be_bytes(bytes[128..132].try_into().unwrap()) as usize;
    let table_end = 132 + count * 12;
    bytes.splice(table_end..table_end, [0; 12]);
    for entry in bytes[132..table_end].as_chunks_mut::<12>().0 {
        let offset = u32::from_be_bytes(entry[4..8].try_into().unwrap());
        entry[4..8].copy_from_slice(&(offset + 12).to_be_bytes());
    }
    bytes[128..132].copy_from_slice(&((count + 1) as u32).to_be_bytes());
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    let offset = bytes.len() as u32;
    let size = 8 + [0, 1, 128, 255, 4096, 65536][index];
    bytes[table_end..table_end + 4].copy_from_slice(b"zzzz");
    bytes[table_end + 4..table_end + 8].copy_from_slice(&offset.to_be_bytes());
    bytes[table_end + 8..table_end + 12].copy_from_slice(&(size as u32).to_be_bytes());
    bytes.extend_from_slice(b"test\0\0\0\0");
    bytes.extend((0..size - 8).map(|index| (index % 256) as u8));
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    bytes[4..8].copy_from_slice(b"abcd");
    bytes[8..12].copy_from_slice(if index.is_multiple_of(2) {
        &[2, 0x10, 0, 0]
    } else {
        &[4, 0x30, 0, 0]
    });
    bytes[40..44]
        .copy_from_slice(&[*b"APPL", *b"MSFT", *b"SGI ", *b"SUNW", *b"TEST", [0; 4]][index]);
    bytes[64..68].copy_from_slice(&(index as u32 % 4).to_be_bytes());
    bytes[80..84].copy_from_slice(if index.is_multiple_of(2) {
        b"abcd"
    } else {
        b"WXYZ"
    });
    bytes[84..128].fill(0); // no profile ID; v2 also reserves these bytes
    let size = bytes.len() as u32;
    bytes[..4].copy_from_slice(&size.to_be_bytes());
    IccProfile::parse(bytes.into(), Default::default()).unwrap()
}

#[test]
fn original_icc_versions_platforms_private_tags_and_all_intents_survive() {
    let rig = Rig::new();
    let native_profile = IccProfileOracle::compile();
    for format in [
        LosslessModularFormat::GrayAlpha,
        LosslessModularFormat::Rgba,
    ] {
        let base = profile(format.color_channel_count() == 1);
        for index in 0..6 {
            let profile = private_profile(&base, index);
            let encoder = encoder(&rig, &profile, TREES[index % 2]);
            let case = Case {
                format,
                bits: 12,
                kind: SampleKind::Unsigned,
                storage: Storage::Split,
                reversed: true,
                byte_order: ByteOrder::Big,
                shifted: true,
            };
            let extent = Extent2d::new(1, 257);
            let samples = case.samples(extent);
            let mut source = upload(&rig.context, &case, extent, &samples, 4099);
            attach(&mut source, &profile, index % 2 == 0);
            let encoded = encoder.encode_container(source).unwrap();
            assert_eq!(
                native_profile.read(&encoded).profile,
                profile.bytes().as_ref()
            );
            check_frame_samples(&encoded, &[&samples], &case, &original(&encoded));
            color::check_numeric(&rig, &encoded, &[samples], &case);
        }
    }
}
