//! Required public libjxl frame-index producer. No production parser supplies its wire bytes.
use std::{fs, path::PathBuf, process::Command};

pub struct FrameIndexOracle {
    directory: PathBuf,
    executable: PathBuf,
}

fn checked(command: &mut Command) -> std::process::Output {
    let output = command
        .output()
        .expect("required native frame-index oracle");
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

impl FrameIndexOracle {
    pub fn compile() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "jxl-frame-index-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let version = checked(Command::new("pkg-config").args(["--modversion", "libjxl"]));
        assert_eq!(String::from_utf8(version.stdout).unwrap().trim(), "0.12.0");
        let flags = checked(Command::new("pkg-config").args(["--cflags", "--libs", "libjxl"]));
        let executable = directory.join("index");
        checked(
            Command::new(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()))
                .args(["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"])
                .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data/frame_index.cpp"))
                .args(String::from_utf8(flags.stdout).unwrap().split_whitespace())
                .arg("-o")
                .arg(&executable),
        );
        Self {
            directory,
            executable,
        }
    }

    /// Tiny still, all-indexed animation, or animation with every second frame indexed.
    pub fn encode(&self, mode: &str) -> Vec<u8> {
        checked(Command::new(&self.executable).arg(mode)).stdout
    }
}

impl Drop for FrameIndexOracle {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}
