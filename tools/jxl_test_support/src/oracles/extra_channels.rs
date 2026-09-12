//! Optional native libjxl oracle shared by Modular and VarDCT substream tests.
use std::process::Command;

pub fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect()
}

pub fn libjxl_planes(
    data: &[u8],
    pixels: usize,
    extras: usize,
) -> Option<(Vec<f32>, Vec<Vec<f32>>)> {
    let values = libjxl_output(data, &[])?;
    assert_eq!(values.len(), pixels * (4 + extras));
    Some((
        values[..pixels * 4].to_vec(),
        values[pixels * 4..]
            .chunks_exact(pixels)
            .map(<[f32]>::to_vec)
            .collect(),
    ))
}

pub fn libjxl_output(data: &[u8], options: &[&str]) -> Option<Vec<f32>> {
    use std::sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering},
    };
    static BINARY: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    static INPUT: AtomicUsize = AtomicUsize::new(0);
    let binary = BINARY
        .get_or_init(|| {
            let flags = Command::new("pkg-config")
                .args(["--cflags", "--libs", "libjxl", "libjxl_cms"])
                .output()
                .ok()?;
            if !flags.status.success() {
                return None;
            }
            let path =
                std::env::temp_dir().join(format!("jxl-wgpu-extra-oracle-{}", std::process::id()));
            let output = Command::new("cc")
                .arg(crate::decoder_directory().join("test-data/decode_extra_channels.c"))
                .args(
                    std::str::from_utf8(&flags.stdout)
                        .unwrap()
                        .split_whitespace(),
                )
                .arg("-o")
                .arg(&path)
                .output()
                .expect("compile available libjxl oracle");
            assert!(
                output.status.success(),
                "libjxl oracle compiler: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            Some(path)
        })
        .as_ref()?;
    let path = std::env::temp_dir().join(format!(
        "jxl-wgpu-extra-input-{}-{}.jxl",
        std::process::id(),
        INPUT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, data).unwrap();
    let decoded = Command::new(binary)
        .arg(&path)
        .args(options)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        decoded.status.success(),
        "libjxl extra oracle: {}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    Some(floats(&decoded.stdout))
}

use jxl::api::{
    Endianness, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlPixelFormat, ProcessingResult, states,
};

pub type FloatPlanes = (Vec<f32>, Vec<Vec<f32>>);

pub fn rust_planes(encoded: &[u8]) -> FloatPlanes {
    let mut frames = rust_frame_planes(encoded);
    assert_eq!(frames.len(), 1);
    frames.remove(0)
}

pub fn rust_frame_planes(encoded: &[u8]) -> Vec<FloatPlanes> {
    let mut input = encoded;
    let mut options = JxlDecoderOptions::default();
    options.render_spot_colors = false;
    options.high_precision = true;
    let decoder = JxlDecoder::<states::Initialized>::new(options);
    let ProcessingResult::Complete {
        result: mut decoder,
    } = decoder.process(&mut input, None).unwrap()
    else {
        panic!("complete header")
    };
    let size = decoder.basic_info().size;
    let count = decoder.basic_info().extra_channels.len();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgba,
        color_data_format: Some(JxlDataFormat::F32 {
            endianness: Endianness::LittleEndian,
        }),
        extra_channel_format: vec![
            Some(JxlDataFormat::F32 {
                endianness: Endianness::LittleEndian
            });
            count
        ],
    });
    let mut frames = Vec::new();
    loop {
        let ProcessingResult::Complete { result: frame } =
            decoder.process(&mut input, None).unwrap()
        else {
            panic!("complete frame")
        };
        let mut color = vec![0u8; size.0 * size.1 * 16];
        let mut extras = vec![vec![0u8; size.0 * size.1 * 4]; count];
        let mut outputs = vec![JxlOutputBuffer::new(&mut color, size.1, size.0 * 16)];
        outputs.extend(
            extras
                .iter_mut()
                .map(|plane| JxlOutputBuffer::new(plane, size.1, size.0 * 4)),
        );
        let ProcessingResult::Complete { result } =
            frame.process(&mut input, &mut outputs, None).unwrap()
        else {
            panic!("complete pixels")
        };
        decoder = result;
        drop(outputs);
        frames.push((floats(&color), extras.iter().map(|p| floats(p)).collect()));
        if !decoder.has_more_frames() {
            break;
        }
    }
    frames
}
