//! Reproduce independently decoded lossy Modular color and every-plane references offline.
use std::path::PathBuf;
#[path = "support/offline.rs"]
mod offline;

fn main() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| source.join("lossy_modular"));
    std::fs::create_dir_all(&output).unwrap();
    let temporary =
        std::env::temp_dir().join(format!("jxl-wgpu-lossy-modular-{}", std::process::id()));
    std::fs::create_dir_all(&temporary).unwrap();
    offline::generate_extras(&source, &output, &temporary, "lossy");
    offline::extra_references(&source, &output, &temporary, "lossy");
    for entry in std::fs::read_dir(&output).unwrap() {
        let path = entry.unwrap().path();
        if path.to_str().unwrap().ends_with(".jxl.hex") {
            let text = std::fs::read_to_string(path).unwrap();
            assert_eq!(offline::hex(&offline::unhex(&text)), text);
        }
    }
    std::fs::remove_dir_all(temporary).unwrap();
    eprintln!("Regenerated lossy Modular fixtures in {}", output.display());
}
