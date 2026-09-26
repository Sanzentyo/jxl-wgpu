//! The shared JPEG XL name syntax for frames and extra channels.
use std::sync::Arc;

use jxl_gpu_bitstream::BitWriter;

use crate::EncodeError;

/// A UTF-8 frame or extra-channel name within JPEG XL's 1071-byte wire limit.
/// Length is measured in bytes; embedded NULs are preserved. Clones share storage.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CodestreamName(Arc<str>);

impl CodestreamName {
    pub const MAX_BYTES: usize = 1071;

    pub fn new(bytes: impl AsRef<[u8]>) -> Result<Self, EncodeError> {
        let bytes = bytes.as_ref();
        if bytes.len() > Self::MAX_BYTES {
            return Err(EncodeError::InvalidConfiguration(
                "codestream name exceeds 1071 UTF-8 bytes",
            ));
        }
        let name = std::str::from_utf8(bytes).map_err(|_| {
            EncodeError::InvalidConfiguration("codestream name must be valid UTF-8")
        })?;
        Ok(Self(Arc::from(name)))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn encoded_bits(&self) -> usize {
        self.0.len() * 8
            + match self.0.len() {
                0 => 2,
                1..=15 => 6,
                16..=47 => 7,
                _ => 12,
            }
    }

    pub(crate) fn write(&self, output: &mut BitWriter) -> Result<(), EncodeError> {
        let length = self.0.len() as u64;
        match length {
            0 => output.write_bits(0, 2)?,
            1..=15 => {
                output.write_bits(1, 2)?;
                output.write_bits(length, 4)?;
            }
            16..=47 => {
                output.write_bits(2, 2)?;
                output.write_bits(length - 16, 5)?;
            }
            _ => {
                output.write_bits(3, 2)?;
                output.write_bits(length - 48, 10)?;
            }
        }
        for byte in self.0.bytes() {
            output.write_bits(u64::from(byte), 8)?;
        }
        Ok(())
    }
}
