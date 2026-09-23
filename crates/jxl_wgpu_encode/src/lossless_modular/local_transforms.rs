use std::sync::Arc;

use crate::{
    EncodeError, LosslessModularRctType, LosslessModularSqueeze, LosslessModularSqueezeStep,
};

/// One ordered operation on the current group-local image channels, excluding Palette metadata.
/// RCT may include any three equal-geometry image channels, including alpha or Squeeze residuals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LosslessModularTransform {
    Rct {
        begin_channel: u32,
        rct_type: LosslessModularRctType,
    },
    Squeeze(LosslessModularSqueezeStep),
}

/// Local transforms after source color transformation and optional Palette.
///
/// Named/separable Squeeze policies retain their optimized lowering and byte identity.
/// An explicit program applies RCT and Squeeze in declaration order to the evolving topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LosslessModularLocalTransforms {
    selection: Selection,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Selection {
    Squeeze(LosslessModularSqueeze),
    Sequence(Arc<[LosslessModularTransform]>),
}

impl Default for LosslessModularLocalTransforms {
    fn default() -> Self {
        LosslessModularSqueeze::None.into()
    }
}

impl From<LosslessModularSqueeze> for LosslessModularLocalTransforms {
    fn from(squeeze: LosslessModularSqueeze) -> Self {
        Self {
            selection: Selection::Squeeze(squeeze),
        }
    }
}

impl LosslessModularLocalTransforms {
    /// Each entry emits one wire transform; 1–273 entries are representable.
    /// The complete header, including preceding local RCT/Palette, must also fit 273 entries.
    /// Actual channel bounds and equal-geometry RCT inputs are checked before GPU admission.
    pub fn sequence(
        operations: impl Into<Arc<[LosslessModularTransform]>>,
    ) -> Result<Self, EncodeError> {
        let operations = operations.into();
        if !(1..=273).contains(&operations.len()) {
            return Err(EncodeError::InvalidModularTransformCount {
                count: operations.len(),
            });
        }
        for operation in operations.iter() {
            if let LosslessModularTransform::Rct { begin_channel, .. } = *operation
                && begin_channel > 9287
            {
                return Err(EncodeError::InvalidModularRctBegin {
                    begin: begin_channel,
                });
            }
        }
        Ok(Self {
            selection: Selection::Sequence(operations),
        })
    }

    #[must_use]
    pub fn operations(&self) -> Option<&[LosslessModularTransform]> {
        match &self.selection {
            Selection::Squeeze(_) => None,
            Selection::Sequence(operations) => Some(operations),
        }
    }

    #[must_use]
    pub fn squeeze_policy(&self) -> Option<&LosslessModularSqueeze> {
        match &self.selection {
            Selection::Squeeze(squeeze) => Some(squeeze),
            Selection::Sequence(_) => None,
        }
    }

    pub(super) fn uses_program(&self) -> bool {
        self.operations().is_some()
            || self
                .squeeze_policy()
                .is_some_and(|policy| policy.steps().is_some())
    }

    pub(super) fn uses_squeeze(&self) -> bool {
        self.squeeze_policy()
            .is_some_and(LosslessModularSqueeze::enabled)
            || self.operations().is_some_and(|operations| {
                operations
                    .iter()
                    .any(|operation| matches!(operation, LosslessModularTransform::Squeeze(_)))
            })
    }
}
