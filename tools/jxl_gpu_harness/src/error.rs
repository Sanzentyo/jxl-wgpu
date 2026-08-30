use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("capture is truncated while reading {0}")]
    Truncated(&'static str),
    #[error("invalid capture magic")]
    InvalidMagic,
    #[error("capture schema version {found} is unsupported; expected {expected}")]
    UnsupportedSchema { found: u16, expected: u16 },
    #[error("capture flags 0x{0:04x} are unsupported")]
    UnsupportedFlags(u16),
    #[error("capture header is too large: {actual} bytes (limit {limit})")]
    HeaderTooLarge { actual: u64, limit: u64 },
    #[error("capture payload is too large: {actual} bytes (limit {limit})")]
    PayloadTooLarge { actual: u64, limit: u64 },
    #[error("capture length overflow")]
    LengthOverflow,
    #[error("capture has {actual} trailing bytes")]
    TrailingBytes { actual: u64 },
    #[error("capture header hash mismatch")]
    HeaderHashMismatch,
    #[error("capture section {section_id} hash mismatch")]
    SectionHashMismatch { section_id: u32 },
    #[error("capture section {0} is declared more than once")]
    DuplicateSection(u32),
    #[error("capture section {0} is missing")]
    MissingSection(u32),
    #[error("capture section layout is invalid: {0}")]
    InvalidSection(String),
    #[error("capture metadata is invalid: {0}")]
    InvalidMetadata(String),
    #[error("tensor data is invalid: {0}")]
    InvalidTensor(String),
    #[error("operation is unsupported by {backend}: {operation}")]
    UnsupportedOperation {
        backend: &'static str,
        operation: String,
    },
    #[error("backend is unavailable: {0}")]
    BackendUnavailable(String),
    #[error("verification failed: {0}")]
    Verification(String),
    #[error("JPEG XL decode failed for {path}: {message}")]
    Decode { path: PathBuf, message: String },
    #[error("configuration is invalid: {0}")]
    InvalidConfig(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),
}

impl Error {
    pub fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.as_ref().to_path_buf(),
            source,
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
