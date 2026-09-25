//! Modular image options and ownership metadata; shared source color syntax is codec-independent.
use super::types::AlphaAssociation;
use crate::{ImageColorOptions, source_color::SourceColorEncoding};

#[derive(Clone, Debug)]
pub(super) struct ModularImageMetadata {
    pub(super) encoding: SourceColorEncoding,
    pub(super) options: ImageColorOptions,
    pub(super) alpha: AlphaAssociation,
    pub(super) max_icc_profile_bytes: u64,
}

impl Default for ModularImageMetadata {
    fn default() -> Self {
        Self {
            encoding: SourceColorEncoding::default(),
            options: ImageColorOptions::default(),
            alpha: AlphaAssociation::default(),
            max_icc_profile_bytes: super::icc::DEFAULT_PROFILE_LIMIT,
        }
    }
}

impl ModularImageMetadata {
    pub(super) fn new(
        encoding: SourceColorEncoding,
        options: ImageColorOptions,
        alpha: AlphaAssociation,
        max_icc_profile_bytes: u64,
    ) -> Self {
        Self {
            encoding,
            options,
            alpha,
            max_icc_profile_bytes,
        }
    }
}
