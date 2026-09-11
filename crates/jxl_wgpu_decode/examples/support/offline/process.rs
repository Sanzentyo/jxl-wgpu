use std::path::Path;
use std::process::{Command, Output};

pub fn run(command: &mut Command) -> Output {
    let output = command.output().expect("run offline fixture tool");
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

pub fn compile(source: &Path, binary: &Path, libraries: &[&str]) {
    let flags = run(Command::new("pkg-config")
        .args(["--cflags", "--libs"])
        .args(libraries));
    run(Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror"])
        .arg(source)
        .args(
            std::str::from_utf8(&flags.stdout)
                .unwrap()
                .split_whitespace(),
        )
        .arg("-o")
        .arg(binary));
}
