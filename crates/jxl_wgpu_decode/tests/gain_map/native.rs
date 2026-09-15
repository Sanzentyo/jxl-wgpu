use std::{
    ffi::OsString,
    path::PathBuf,
    process::{Command, Output},
};

pub(super) struct Oracle {
    executable: OsString,
    directory: PathBuf,
}

impl Oracle {
    pub(super) fn new() -> Option<Self> {
        let Some(executable) = std::env::var_os("JXL_GAIN_MAP_ORACLE") else {
            assert!(
                std::env::var_os("JXL_REQUIRE_NATIVE_ORACLES").is_none(),
                "JXL_GAIN_MAP_ORACLE is required"
            );
            eprintln!("skipping live native gain-map oracle; set JXL_GAIN_MAP_ORACLE");
            return None;
        };
        let directory = std::env::temp_dir().join(format!(
            "jxl-gain-map-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        Some(Self {
            executable,
            directory,
        })
    }

    pub(super) fn run(&self, operation: &str, bytes: &[u8]) -> Output {
        let input = self.directory.join("input");
        std::fs::write(&input, bytes).unwrap();
        Command::new(&self.executable)
            .arg(operation)
            .arg(input)
            .arg(self.directory.join("output"))
            .output()
            .unwrap()
    }

    pub(super) fn roundtrip(&self, operation: &str, bytes: &[u8]) -> Vec<u8> {
        let output = self.run(operation, bytes);
        assert!(
            output.status.success(),
            "{operation}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::read(self.directory.join("output")).unwrap()
    }

    pub(super) fn apply(
        &self,
        iso: &[u8],
        base: &[f64],
        map: &[f64],
        extent: [usize; 2],
        map_extent: [usize; 2],
        headroom: f64,
    ) -> Vec<f64> {
        let input = self.directory.join("iso");
        let base_path = self.directory.join("base");
        let map_path = self.directory.join("gain");
        std::fs::write(&input, iso).unwrap();
        for (path, values) in [(&base_path, base), (&map_path, map)] {
            std::fs::write(
                path,
                values
                    .iter()
                    .flat_map(|v| (*v as f32).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
        let result = Command::new(&self.executable)
            .arg("apply")
            .arg(input)
            .arg(base_path)
            .arg(map_path)
            .args(extent.into_iter().chain(map_extent).map(|v| v.to_string()))
            .arg(headroom.to_string())
            .arg(self.directory.join("output"))
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "native application: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let bytes = std::fs::read(self.directory.join("output")).unwrap();
        let (words, tail) = bytes.as_chunks::<4>();
        assert!(tail.is_empty());
        words
            .iter()
            .map(|word| f64::from(f32::from_le_bytes(*word)))
            .collect()
    }
}

impl Drop for Oracle {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}
