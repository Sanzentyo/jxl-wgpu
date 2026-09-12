use std::fmt::Write as _;

pub fn hex(bytes: &[u8]) -> String {
    let mut text = String::new();
    for line in bytes.chunks(32) {
        for byte in line {
            write!(text, "{byte:02x}").unwrap();
        }
        text.push('\n');
    }
    text
}

pub fn unhex(text: &str) -> Vec<u8> {
    let hex = text.split_whitespace().collect::<String>();
    assert!(hex.len().is_multiple_of(2));
    hex.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}
