//! Public libjxl original-profile/image metadata inspection and metadata-only input generation.
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Output, Stdio},
};

pub struct IccProfileOracle {
    directory: PathBuf,
    executable: PathBuf,
}
pub struct NativeIccProfile {
    pub input: Vec<u8>,
    pub profile: Vec<u8>,
}
#[derive(Debug)]
pub struct NativeImageInfo {
    pub size: (u32, u32),
    pub orientation: u32,
    pub intrinsic_size: (u32, u32),
    pub intensity_target: f32,
    pub min_nits: f32,
    pub relative_to_max_display: bool,
    pub linear_below: f32,
    pub preview: Option<(u32, u32)>,
    pub animation: bool,
}
fn checked(command: &mut Command) -> Output {
    let output = command
        .output()
        .expect("required native ICC profile oracle");
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}
impl IccProfileOracle {
    pub fn compile() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "jxl-icc-profile-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        fs::create_dir(&directory).unwrap();
        let version = checked(Command::new("pkg-config").args(["--modversion", "libjxl"]));
        assert_eq!(String::from_utf8(version.stdout).unwrap().trim(), "0.12.0");
        let flags = checked(Command::new("pkg-config").args(["--cflags", "--libs", "libjxl"]));
        let executable = directory.join("profile");
        checked(
            Command::new(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()))
                .args(["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"])
                .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data/icc_profile.cpp"))
                .args(String::from_utf8(flags.stdout).unwrap().split_whitespace())
                .arg("-o")
                .arg(&executable),
        );
        Self {
            directory,
            executable,
        }
    }
    /// Five public enum numbers followed by xy white/primaries and gamma; no expected bytes.
    pub fn create(&self, declaration: &str) -> NativeIccProfile {
        let mut process = Command::new(&self.executable)
            .arg("create")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        process
            .stdin
            .take()
            .unwrap()
            .write_all(declaration.as_bytes())
            .unwrap();
        let output = process.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{declaration}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        decode(output.stdout)
    }
    pub fn read(&self, input: &[u8]) -> NativeIccProfile {
        let path = self.directory.join("input.jxl");
        fs::write(&path, input).unwrap();
        decode(checked(Command::new(&self.executable).arg("read").arg(path)).stdout)
    }
    /// Raw image declarations with orientation kept; no decoding, resampling or tone mapping.
    pub fn image_info(&self, input: &[u8]) -> NativeImageInfo {
        let path = self.directory.join("input.jxl");
        fs::write(&path, input).unwrap();
        let bytes = checked(Command::new(&self.executable).arg("read-info").arg(path)).stdout;
        assert_eq!(bytes.len(), 14 * 4);
        let words = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect::<Vec<_>>();
        assert_eq!(words[0], 12000);
        for i in [8, 10, 13] {
            assert!(words[i] <= 1);
        }
        NativeImageInfo {
            size: (words[1], words[2]),
            orientation: words[3],
            intrinsic_size: (words[4], words[5]),
            intensity_target: f32::from_bits(words[6]),
            min_nits: f32::from_bits(words[7]),
            relative_to_max_display: words[8] != 0,
            linear_below: f32::from_bits(words[9]),
            preview: (words[10] != 0).then_some((words[11], words[12])),
            animation: words[13] != 0,
        }
    }
}
fn decode(bytes: Vec<u8>) -> NativeIccProfile {
    assert!(bytes.len() >= 8);
    let input = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let profile = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    assert_eq!(bytes.len(), 8 + input + profile);
    NativeIccProfile {
        input: bytes[8..8 + input].to_vec(),
        profile: bytes[8 + input..].to_vec(),
    }
}
impl Drop for IccProfileOracle {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}
