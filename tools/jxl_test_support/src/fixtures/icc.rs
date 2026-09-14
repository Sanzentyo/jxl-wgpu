//! Structurally complete ICC methods for testing transform selection independently of pixels.
use jxl_gpu_protocol::icc::{IccDirection, IccProfile, IccRenderingIntent, IccSignature};

/// A selected, structurally complete matrix containing an invalid IEEE infinity.
/// Unused directional methods must remain uninterpreted; selecting this one must fail.
pub fn with_nonfinite_matrix_mpe(
    profile: &IccProfile,
    direction: IccDirection,
    intent: IccRenderingIntent,
) -> IccProfile {
    let profile = with_matrix_mpe(profile, direction, intent);
    let mut signature = match direction {
        IccDirection::DeviceToPcs => *b"D2B0",
        IccDirection::PcsToDevice => *b"B2D0",
    };
    signature[3] += intent as u8;
    let start = profile.tag(IccSignature(signature)).unwrap().offset as usize + 24 + 12;
    let mut bytes = profile.bytes().to_vec();
    bytes[start..start + 4].copy_from_slice(&f32::INFINITY.to_be_bytes());
    IccProfile::parse(bytes.into(), Default::default()).unwrap()
}

/// Adds an intent-specific MPE with one matrix element. Matrix/TRC execution must
/// select this higher-priority method instead of silently using the original TRCs.
pub fn with_matrix_mpe(
    profile: &IccProfile,
    direction: IccDirection,
    intent: IccRenderingIntent,
) -> IccProfile {
    let channels: u16 = match &profile.header().device_space.0 {
        b"RGB " => 3,
        b"GRAY" => 1,
        other => panic!("unsupported fixture color space {other:?}"),
    };
    let (inputs, outputs, signature) = match direction {
        IccDirection::DeviceToPcs => (channels, 3, [b'D', b'2', b'B', b'0' + intent as u8]),
        IccDirection::PcsToDevice => (3, channels, [b'B', b'2', b'D', b'0' + intent as u8]),
    };
    let signature = IccSignature(signature);
    assert!(profile.tag(signature).is_none());
    let mut matrix = b"matf\0\0\0\0".to_vec();
    matrix.extend_from_slice(&inputs.to_be_bytes());
    matrix.extend_from_slice(&outputs.to_be_bytes());
    for row in 0..outputs {
        for column in 0..inputs {
            let value: f32 = if inputs == 1 {
                [0.9642, 1.0, 0.8249][usize::from(row)]
            } else if outputs == 1 {
                if column == 1 { 1.0 } else { 0.0 }
            } else if row == column {
                1.0
            } else {
                0.0
            };
            matrix.extend_from_slice(&value.to_be_bytes());
        }
    }
    for _ in 0..outputs {
        matrix.extend_from_slice(&0_f32.to_be_bytes());
    }
    let mut mpe = b"mpet\0\0\0\0".to_vec();
    mpe.extend_from_slice(&inputs.to_be_bytes());
    mpe.extend_from_slice(&outputs.to_be_bytes());
    mpe.extend_from_slice(&1_u32.to_be_bytes());
    mpe.extend_from_slice(&24_u32.to_be_bytes());
    mpe.extend_from_slice(&(matrix.len() as u32).to_be_bytes());
    mpe.extend_from_slice(&matrix);

    let mut tags = profile
        .tags()
        .iter()
        .map(|tag| {
            (
                tag.signature,
                profile.tag_data(tag.signature).unwrap().to_vec(),
            )
        })
        .collect::<Vec<_>>();
    tags.push((signature, mpe));
    let mut bytes = profile.bytes()[..128].to_vec();
    bytes[84..100].fill(0); // The edited profile has no assigned profile ID.
    bytes.extend_from_slice(&(tags.len() as u32).to_be_bytes());
    bytes.resize(132 + 12 * tags.len(), 0);
    for (index, (signature, data)) in tags.iter().enumerate() {
        let offset = bytes.len() as u32;
        let entry = 132 + 12 * index;
        bytes[entry..entry + 4].copy_from_slice(&signature.0);
        bytes[entry + 4..entry + 8].copy_from_slice(&offset.to_be_bytes());
        bytes[entry + 8..entry + 12].copy_from_slice(&(data.len() as u32).to_be_bytes());
        bytes.extend_from_slice(data);
        bytes.resize(bytes.len().next_multiple_of(4), 0);
    }
    let size = bytes.len() as u32;
    bytes[..4].copy_from_slice(&size.to_be_bytes());
    IccProfile::parse(bytes.into(), Default::default()).unwrap()
}
