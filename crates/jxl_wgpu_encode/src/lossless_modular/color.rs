//! Modular image options and ownership metadata; shared source color syntax is codec-independent.
use super::types::AlphaAssociation;
use crate::{ImageOptions, source_color::SourceColorEncoding};

#[derive(Clone, Debug)]
pub(super) struct ModularImageMetadata {
    pub(super) encoding: SourceColorEncoding,
    pub(super) options: ImageOptions,
    pub(super) alpha: AlphaAssociation,
    pub(super) max_icc_profile_bytes: u64,
    pub(super) extra_channels: Vec<crate::ExtraChannel>,
    pub(super) max_extra_channel_metadata_bytes: u64,
}

impl Default for ModularImageMetadata {
    fn default() -> Self {
        Self {
            encoding: SourceColorEncoding::default(),
            options: ImageOptions::default(),
            alpha: AlphaAssociation::default(),
            max_icc_profile_bytes: crate::source_color::icc::DEFAULT_PROFILE_LIMIT,
            extra_channels: Vec::new(),
            max_extra_channel_metadata_bytes: 1 << 20,
        }
    }
}

impl ModularImageMetadata {
    pub(super) fn new(
        encoding: SourceColorEncoding,
        options: ImageOptions,
        alpha: AlphaAssociation,
        max_icc_profile_bytes: u64,
    ) -> Self {
        Self {
            encoding,
            options,
            alpha,
            max_icc_profile_bytes,
            extra_channels: Vec::new(),
            max_extra_channel_metadata_bytes: 1 << 20,
        }
    }

    pub(super) fn with_inputs(mut self, config: &super::types::LosslessModularConfig) -> Self {
        self.extra_channels.clone_from(&config.extra_channels);
        self.max_extra_channel_metadata_bytes = config.max_extra_channel_metadata_bytes;
        self
    }
}
