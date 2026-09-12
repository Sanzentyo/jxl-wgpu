//! Optional native libjxl oracle for physical pass boundaries.
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct NativeUpdate {
    pub frame: usize,
    pub duration: u32,
    pub timecode: u32,
    pub is_last: bool,
    pub step: usize,
    pub ratio: u32,
    pub complete: bool,
    pub pixels: Vec<u8>,
}

pub fn native_updates(encoded: &[u8], linear: bool) -> Option<Vec<NativeUpdate>> {
    native_updates_oriented(encoded, linear, false)
}

pub fn native_updates_oriented(
    encoded: &[u8],
    linear: bool,
    keep: bool,
) -> Option<Vec<NativeUpdate>> {
    native_updates_options(encoded, linear, keep, false)
}

pub fn native_updates_options(
    encoded: &[u8],
    linear: bool,
    keep: bool,
    flush_prefix: bool,
) -> Option<Vec<NativeUpdate>> {
    native_updates_with_spots(encoded, linear, keep, flush_prefix, true)
}

pub fn native_updates_with_spots(
    encoded: &[u8],
    linear: bool,
    keep: bool,
    flush_prefix: bool,
    render_spots: bool,
) -> Option<Vec<NativeUpdate>> {
    decode_updates(encoded, linear, keep, flush_prefix, render_spots, false)
}

/// Each snapshot contains packed RGBA followed by every extra channel as a separate f32 plane.
pub fn native_updates_all_channels(
    encoded: &[u8],
    linear: bool,
    keep: bool,
    flush_prefix: bool,
) -> Option<Vec<NativeUpdate>> {
    decode_updates(encoded, linear, keep, flush_prefix, false, true)
}

fn decode_updates(
    encoded: &[u8],
    linear: bool,
    keep: bool,
    flush_prefix: bool,
    render_spots: bool,
    extra_planes: bool,
) -> Option<Vec<NativeUpdate>> {
    use std::process::Command;
    static BINARY: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let binary = BINARY
        .get_or_init(|| {
            let flags = Command::new("pkg-config")
                .args(["--cflags", "--libs", "libjxl", "libjxl_cms"])
                .output()
                .ok()?;
            if !flags.status.success() {
                return None;
            }
            let binary =
                std::env::temp_dir().join(format!("jxl-wgpu-pass-oracle-{}", std::process::id()));
            let compiled = Command::new("cc")
                .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-O2"])
                .arg(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("test-data/decode_progressive.c"),
                )
                .args(
                    std::str::from_utf8(&flags.stdout)
                        .unwrap()
                        .split_whitespace(),
                )
                .arg("-o")
                .arg(&binary)
                .output()
                .unwrap();
            assert!(
                compiled.status.success(),
                "{}",
                String::from_utf8_lossy(&compiled.stderr)
            );
            Some(binary)
        })
        .as_ref()?;
    let directory = std::env::temp_dir().join(format!(
        "jxl-wgpu-pass-input-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&directory).unwrap();
    let input = directory.join("image.jxl");
    let prefix = directory.join("snapshot");
    std::fs::write(&input, encoded).unwrap();
    let mut command = Command::new(binary);
    command
        .arg(&input)
        .arg(encoded.len().to_string())
        .arg(&prefix);
    if linear {
        command.arg("linear");
    }
    if keep {
        command.arg("keep");
    }
    if flush_prefix {
        command.arg("prefix");
    }
    if !render_spots {
        command.arg("no-spots");
    }
    if extra_planes {
        command.arg("extras");
    }
    let decoded = command.output().unwrap();
    assert!(
        decoded.status.success(),
        "status {:?}\n{}\n{}",
        decoded.status,
        String::from_utf8_lossy(&decoded.stdout),
        String::from_utf8_lossy(&decoded.stderr)
    );
    let mut updates = Vec::new();
    let mut header = (0, 0, 0, false);
    for line in std::str::from_utf8(&decoded.stdout).unwrap().lines() {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields[0] == "frame" {
            header = (
                fields[1].parse::<usize>().unwrap(),
                fields[2].parse::<u32>().unwrap(),
                fields[3].parse::<u32>().unwrap(),
                fields[4] == "1",
            );
            continue;
        }
        if !matches!(fields[0], "progress" | "final") {
            continue;
        }
        let frame: usize = fields[1].parse().unwrap();
        assert_eq!(frame, header.0);
        assert_eq!(fields[5], "0", "nonfinite oracle output");
        let step = fields[2].parse().unwrap();
        updates.push(NativeUpdate {
            frame,
            duration: header.1,
            timecode: header.2,
            is_last: header.3,
            step,
            ratio: fields[3].parse().unwrap(),
            complete: fields[0] == "final",
            pixels: std::fs::read(directory.join(format!(
                "snapshot-frame{frame}-step{step}-{}.f32",
                fields[0]
            )))
            .unwrap(),
        });
    }
    std::fs::remove_dir_all(directory).unwrap();
    Some(updates)
}
