//! Independent libjpeg-turbo integer coefficient and MCU-padding oracle. No JXL decoder is used.

use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
};

pub struct JpegCoefficientOracle {
    directory: PathBuf,
    executable: PathBuf,
}

pub struct JpegCoefficientReference {
    pub extent: [u32; 2],
    pub components: Vec<JpegComponentReference>,
}

pub struct JpegComponentReference {
    pub id: u32,
    pub sampling: [u32; 2],
    pub blocks: [u32; 2],
    pub real_blocks: [u32; 2],
    pub quantization: Vec<u16>,
    pub coefficients: Vec<i16>,
}

fn checked(command: &mut Command) -> Output {
    let output = command
        .output()
        .expect("required native JPEG coefficient oracle tool");
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

impl JpegCoefficientOracle {
    pub fn compile() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "jxl-jpeg-coefficients-{}",
            format_args!(
                "{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            )
        ));
        fs::create_dir(&directory).unwrap();
        let executable = directory.join("coefficients");
        let version = checked(Command::new("pkg-config").args(["--modversion", "libjpeg"]));
        // Padded virtual-array access is qualified against this upstream implementation/version.
        assert_eq!(
            String::from_utf8(version.stdout).unwrap().trim(),
            "3.2.0",
            "the padded coefficient oracle requires libjpeg-turbo 3.2.0"
        );
        eprintln!("JPEG coefficient oracle: libjpeg-turbo 3.2.0, including padded MCU blocks");
        let flags = checked(Command::new("pkg-config").args(["--cflags", "--libs", "libjpeg"]));
        checked(
            Command::new(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()))
                .args(["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"])
                .arg(
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("test-data/jpeg_coefficients.cpp"),
                )
                .args(String::from_utf8(flags.stdout).unwrap().split_whitespace())
                .arg("-o")
                .arg(&executable),
        );
        Self {
            directory,
            executable,
        }
    }

    pub fn read(&self, jpeg: &[u8]) -> JpegCoefficientReference {
        let path = self.directory.join("source.jpg");
        fs::write(&path, jpeg).unwrap();
        let output = checked(Command::new(&self.executable).arg(path));
        let mut bytes = output.stdout.as_slice();
        fn word(bytes: &mut &[u8]) -> u32 {
            let (head, tail) = bytes.split_at(4);
            *bytes = tail;
            u32::from_le_bytes(head.try_into().unwrap())
        }
        assert_eq!(word(&mut bytes), u32::from_le_bytes(*b"JCO1"));
        let extent = [word(&mut bytes), word(&mut bytes)];
        let count = word(&mut bytes);
        assert!((1..=3).contains(&count));
        let mut components = Vec::new();
        for _ in 0..count {
            let id = word(&mut bytes);
            let sampling = [word(&mut bytes), word(&mut bytes)];
            let blocks = [word(&mut bytes), word(&mut bytes)];
            let real_blocks = [word(&mut bytes), word(&mut bytes)];
            let quantization = (0..64)
                .map(|_| u16::try_from(word(&mut bytes)).unwrap())
                .collect();
            let coefficients = (0..u64::from(blocks[0]) * u64::from(blocks[1]) * 64)
                .map(|_| i16::try_from(word(&mut bytes) as i32).unwrap())
                .collect();
            components.push(JpegComponentReference {
                id,
                sampling,
                blocks,
                real_blocks,
                quantization,
                coefficients,
            });
        }
        assert!(bytes.is_empty());
        JpegCoefficientReference { extent, components }
    }
}

impl Drop for JpegCoefficientOracle {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}
