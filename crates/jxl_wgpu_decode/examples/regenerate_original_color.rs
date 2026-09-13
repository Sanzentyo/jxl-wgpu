//! Complete native component sources as YCbCr fixtures and verify every declared case.
use jxl_test_support::{fixtures::original_color as corpus, offline, oracles::extra_channels};

fn main() {
    for case in corpus::cases() {
        let encoded = if case.mode.ycbcr() {
            let encoded = case.encode_ycbcr();
            let native = extra_channels::libjxl_output(
                &encoded,
                &["--preserve-alpha", "--keep-orientation"],
            )
            .expect("native libjxl is required to generate original-color references");
            let pixels = 37 * 19;
            assert!(native.len().is_multiple_of(pixels * 5));
            let mut reference = Vec::new();
            for frame in native.chunks_exact(pixels * 5) {
                for pixel in 0..pixels {
                    assert_eq!(frame[pixel * 4 + 3], frame[pixels * 4 + pixel]);
                }
                reference.extend_from_slice(&frame[..pixels * 4]);
            }
            std::fs::write(
                corpus::directory().join(format!("{}.jxl.hex", case.name)),
                offline::hex(&encoded),
            )
            .unwrap();
            let bytes: Vec<_> = reference
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect();
            std::fs::write(
                corpus::directory().join(format!("{}.original.f32.hex", case.name)),
                offline::float_hex(&bytes),
            )
            .unwrap();
            encoded
        } else {
            case.bytes()
        };
        let inventory = jxl_gpu_bitstream::parse(&encoded, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        case.validate(&inventory);
        let reference = case.reference();
        assert_eq!(
            reference.len(),
            37 * 19 * 4 * if case.sequence { 4 } else { 1 }
        );
        assert!(reference.iter().all(|value| value.is_finite()));
        eprintln!(
            "{}: {} physical frames verified",
            case.name,
            inventory.frames.len()
        );
    }
}
