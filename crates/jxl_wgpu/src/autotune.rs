// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Serializable kernel choices keyed to a concrete adapter and driver.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const PROFILE_VERSION: u32 = 1;

/// Stable-enough adapter identity for rejecting stale tuning data.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AdapterFingerprint {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
}

impl AdapterFingerprint {
    pub fn from_adapter_info(info: &wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name.clone(),
            vendor: info.vendor,
            device: info.device,
            device_type: format!("{:?}", info.device_type),
            backend: format!("{:?}", info.backend),
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
        }
    }
}

/// Workgroup shapes considered by the portable kernels.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum KernelVariant {
    Scalar,
    Tile8x8,
    Tile16x8,
    #[default]
    Tile16x16,
    Tile32x4,
}

impl KernelVariant {
    pub const ALL: [Self; 5] = [
        Self::Scalar,
        Self::Tile8x8,
        Self::Tile16x8,
        Self::Tile16x16,
        Self::Tile32x4,
    ];

    pub const fn workgroup_size(self) -> (u32, u32) {
        match self {
            Self::Scalar => (1, 1),
            Self::Tile8x8 => (8, 8),
            Self::Tile16x8 => (16, 8),
            Self::Tile16x16 => (16, 16),
            Self::Tile32x4 => (32, 4),
        }
    }

    pub const fn invocations(self) -> u32 {
        let (x, y) = self.workgroup_size();
        x * y
    }
}

/// Measured choice for one stable kernel key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TunedKernel {
    pub kernel: String,
    pub variant: KernelVariant,
    pub median_ns: u64,
    pub p95_ns: u64,
    pub samples: u32,
}

impl TunedKernel {
    pub fn from_samples(
        kernel: impl Into<String>,
        variant: KernelVariant,
        samples: &[u64],
    ) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mut ordered = samples.to_vec();
        ordered.sort_unstable();
        let middle = ordered.len() / 2;
        let median_ns = if ordered.len().is_multiple_of(2) {
            ordered[middle - 1].saturating_add(ordered[middle]) / 2
        } else {
            ordered[middle]
        };
        let p95_index = ordered.len().saturating_mul(95).div_ceil(100) - 1;
        Some(Self {
            kernel: kernel.into(),
            variant,
            median_ns,
            p95_ns: ordered[p95_index],
            samples: u32::try_from(samples.len()).unwrap_or(u32::MAX),
        })
    }
}

/// On-disk collection of the fastest measured variants for one adapter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutotuneProfile {
    pub version: u32,
    pub adapter: AdapterFingerprint,
    pub kernels: BTreeMap<String, TunedKernel>,
}

impl AutotuneProfile {
    pub fn new(adapter: AdapterFingerprint) -> Self {
        Self {
            version: PROFILE_VERSION,
            adapter,
            kernels: BTreeMap::new(),
        }
    }

    pub fn best(&self, kernel: &str) -> Option<&TunedKernel> {
        self.kernels.get(kernel)
    }

    pub fn best_variant(&self, kernel: &str) -> Option<KernelVariant> {
        self.best(kernel).map(|tuned| tuned.variant)
    }

    /// Records a result, retaining the lowest median for a kernel key.
    pub fn record(&mut self, tuned: TunedKernel) -> bool {
        let replace = self
            .kernels
            .get(&tuned.kernel)
            .is_none_or(|current| tuned.median_ns < current.median_ns);
        if replace {
            self.kernels.insert(tuned.kernel.clone(), tuned);
        }
        replace
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let profile: Self = serde_json::from_str(json)?;
        if profile.version != PROFILE_VERSION {
            return Err(Error::Unsupported(format!(
                "autotune profile version {} is unsupported (expected {PROFILE_VERSION})",
                profile.version
            )));
        }
        Ok(profile)
    }

    pub fn to_json(&self) -> Result<String> {
        let mut json = serde_json::to_string_pretty(self)?;
        json.push('\n');
        Ok(json)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, self.to_json()?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint() -> AdapterFingerprint {
        AdapterFingerprint {
            name: "test adapter".into(),
            vendor: 1,
            device: 2,
            device_type: "DiscreteGpu".into(),
            backend: "Vulkan".into(),
            driver: "test".into(),
            driver_info: "1.0".into(),
        }
    }

    #[test]
    fn computes_robust_sample_summary() {
        let tuned =
            TunedKernel::from_samples("gaborish", KernelVariant::Tile16x8, &[100, 10, 30, 20])
                .unwrap();
        assert_eq!(tuned.median_ns, 25);
        assert_eq!(tuned.p95_ns, 100);
        assert_eq!(tuned.samples, 4);
    }

    #[test]
    fn keeps_fastest_variant() {
        let mut profile = AutotuneProfile::new(fingerprint());
        assert!(profile.record(
            TunedKernel::from_samples("copy", KernelVariant::Tile8x8, &[20, 22, 24],).unwrap()
        ));
        assert!(!profile.record(
            TunedKernel::from_samples("copy", KernelVariant::Tile16x16, &[30, 32, 34],).unwrap()
        ));
        assert_eq!(profile.best_variant("copy"), Some(KernelVariant::Tile8x8));
    }

    #[test]
    fn json_round_trip_is_versioned() {
        let mut profile = AutotuneProfile::new(fingerprint());
        profile
            .record(TunedKernel::from_samples("copy", KernelVariant::Scalar, &[5, 6, 7]).unwrap());
        let json = profile.to_json().unwrap();
        assert_eq!(AutotuneProfile::from_json(&json).unwrap(), profile);

        let incompatible = json.replacen("\"version\": 1", "\"version\": 2", 1);
        assert!(matches!(
            AutotuneProfile::from_json(&incompatible),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn variants_report_valid_dimensions() {
        for variant in KernelVariant::ALL {
            let (x, y) = variant.workgroup_size();
            assert_eq!(variant.invocations(), x * y);
            assert_ne!(x, 0);
            assert_ne!(y, 0);
        }
    }
}
