use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::capture::{OperationKind, PrecisionMode};
use crate::compare::AccuracyThreshold;
use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThresholdConfig {
    pub version: u16,
    #[serde(default)]
    pub default: AccuracyThreshold,
    #[serde(default)]
    pub operations: BTreeMap<String, AccuracyThreshold>,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            version: 1,
            default: AccuracyThreshold::default(),
            operations: BTreeMap::new(),
        }
    }
}

impl ThresholdConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path).map_err(|source| Error::io(path, source))?;
        let config: Self = toml::from_str(&source)?;
        if config.version != 1 {
            return Err(Error::InvalidConfig(format!(
                "threshold config version {} is unsupported",
                config.version
            )));
        }
        Ok(config)
    }

    pub fn for_operation(&self, operation: &OperationKind) -> &AccuracyThreshold {
        self.for_name(operation.as_str())
    }

    pub fn for_name(&self, operation: &str) -> &AccuracyThreshold {
        self.operations.get(operation).unwrap_or(&self.default)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CorpusConfig {
    pub version: u16,
    pub cases: Vec<SyntheticCaseConfig>,
}

impl CorpusConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path).map_err(|source| Error::io(path, source))?;
        let config: Self = toml::from_str(&source)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != 1 {
            return Err(Error::InvalidConfig(format!(
                "corpus config version {} is unsupported",
                self.version
            )));
        }
        if self.cases.is_empty() {
            return Err(Error::InvalidConfig(
                "corpus must contain at least one case".into(),
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        for case in &self.cases {
            case.validate()?;
            if !names.insert(&case.name) {
                return Err(Error::InvalidConfig(format!(
                    "duplicate corpus case {}",
                    case.name
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticCaseConfig {
    pub name: String,
    pub operation: OperationKind,
    pub width: u32,
    pub height: u32,
    #[serde(default = "default_channels")]
    pub channels: u16,
    pub seed: u64,
    #[serde(default = "default_precision")]
    pub precision: PrecisionMode,
    #[serde(default)]
    pub parameters: BTreeMap<String, f64>,
}

const fn default_channels() -> u16 {
    3
}

const fn default_precision() -> PrecisionMode {
    PrecisionMode::F32
}

impl SyntheticCaseConfig {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Error::InvalidConfig("case name must not be empty".into()));
        }
        if self.width == 0 || self.height == 0 || self.channels == 0 {
            return Err(Error::InvalidConfig(format!(
                "case {} has an empty dimension",
                self.name
            )));
        }
        if self.channels > u16::from(u8::MAX) {
            return Err(Error::InvalidConfig(format!(
                "case {} has {} channels, but at most {} are supported",
                self.name,
                self.channels,
                u8::MAX
            )));
        }
        let elements = u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .and_then(|value| value.checked_mul(u64::from(self.channels)))
            .ok_or(Error::LengthOverflow)?;
        if elements > 256 * 1024 * 1024 {
            return Err(Error::InvalidConfig(format!(
                "case {} is too large: {elements} scalar elements",
                self.name
            )));
        }
        match self.operation {
            OperationKind::Epf if self.channels != 3 => Err(Error::InvalidConfig(format!(
                "case {} requires exactly three EPF channels",
                self.name
            ))),
            OperationKind::YcbcrToRgb if self.channels != 3 => Err(Error::InvalidConfig(format!(
                "case {} requires exactly three YCbCr channels",
                self.name
            ))),
            OperationKind::PremultiplyAlpha if self.channels < 2 => Err(Error::InvalidConfig(
                format!("case {} requires color and alpha channels", self.name),
            )),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_override_is_selected_by_operation() {
        let mut config = ThresholdConfig::default();
        let exact = AccuracyThreshold {
            require_exact: true,
            ..AccuracyThreshold::default()
        };
        config.operations.insert("copy".into(), exact.clone());
        assert_eq!(config.for_operation(&OperationKind::Copy), &exact);
        assert_eq!(
            config.for_operation(&OperationKind::Gaborish),
            &config.default
        );
    }
}
