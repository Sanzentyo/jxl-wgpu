use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

pub struct Oracle {
    directory: PathBuf,
    executable: PathBuf,
}

fn checked(command: &mut Command) -> Output {
    let output = command.output().expect("native oracle tool is required");
    assert!(
        output.status.success(),
        "{command:?}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

impl Oracle {
    pub fn compile() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "jxl-jbrd-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let executable = directory.join("native-reconstruct");
        // Oracle absence fails this test; there is no pixel fallback or silent skip.
        let version = checked(Command::new("pkg-config").args(["--modversion", "libjxl"]));
        eprintln!(
            "JPEG-only native oracle: libjxl {}",
            String::from_utf8_lossy(&version.stdout).trim()
        );
        let flags = checked(Command::new("pkg-config").args(["--cflags", "--libs", "libjxl"]));
        checked(
            Command::new(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()))
                .args(["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"])
                .arg(
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("test-data/jpeg_reconstruction_oracle/main.cpp"),
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

    fn run(&self, input: &[u8], limit: usize) -> (Output, PathBuf) {
        let source = self.directory.join("input.jxl");
        let destination = self.directory.join("output.jpg");
        if destination.exists() {
            fs::remove_file(&destination).unwrap();
        }
        fs::write(&source, input).unwrap();
        let output = Command::new(&self.executable)
            .arg(&source)
            .arg(&destination)
            .arg(limit.to_string())
            .output()
            .expect("run native JPEG oracle");
        (output, destination)
    }

    pub fn assert_exact(&self, input: &[u8], jpeg: &[u8]) {
        let (output, destination) = self.run(input, jpeg.len() + 1);
        assert!(
            output.status.success(),
            "JPEG-only reconstruction failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("without pixel fallback"));
        assert_eq!(fs::read(destination).unwrap(), jpeg);
    }

    pub fn assert_rejected(&self, input: &[u8], jpeg_bytes: usize) {
        let (output, destination) = self.run(input, jpeg_bytes + 1);
        assert!(
            !output.status.success(),
            "native oracle accepted malformed reconstruction metadata"
        );
        assert!(
            !destination.exists(),
            "failed JPEG reconstruction published output"
        );
    }
}

impl Drop for Oracle {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}
