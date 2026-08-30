use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub sample_count: u64,
    pub finite_sample_count: u64,
    pub exact_match_count: u64,
    pub max_abs: f64,
    pub mean_abs: f64,
    pub rmse: f64,
    pub max_rel: f64,
    pub max_ulp: u32,
    pub p99_ulp: u32,
    pub psnr_db: Option<f64>,
    pub nan_mismatches: u64,
    pub infinity_mismatches: u64,
    pub first_mismatch_index: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AccuracyThreshold {
    pub max_abs: f64,
    pub max_rel: f64,
    pub max_rmse: f64,
    pub max_ulp: u32,
    pub max_p99_ulp: u32,
    pub min_psnr_db: Option<f64>,
    #[serde(default)]
    pub require_exact: bool,
}

impl Default for AccuracyThreshold {
    fn default() -> Self {
        Self {
            max_abs: 2.0e-5,
            max_rel: 2.0e-5,
            max_rmse: 2.0e-6,
            max_ulp: 32,
            max_p99_ulp: 8,
            min_psnr_db: Some(90.0),
            require_exact: false,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThresholdEvaluation {
    pub passed: bool,
    pub failures: Vec<String>,
}

impl AccuracyThreshold {
    pub fn evaluate(&self, metrics: &AccuracyMetrics) -> ThresholdEvaluation {
        let mut failures = self.common_failures(metrics);
        let exceeds_pointwise_alternatives = metrics.max_abs > self.max_abs
            && metrics.max_rel > self.max_rel
            && metrics.max_ulp > self.max_ulp;
        if exceeds_pointwise_alternatives {
            failures.push(format!(
                "pointwise error exceeds all alternatives: max_abs {} > {}, max_rel {} > {}, max_ulp {} > {}",
                metrics.max_abs,
                self.max_abs,
                metrics.max_rel,
                self.max_rel,
                metrics.max_ulp,
                self.max_ulp
            ));
        }
        if metrics.max_abs > self.max_abs
            && metrics.max_rel > self.max_rel
            && metrics.p99_ulp > self.max_p99_ulp
        {
            failures.push(format!(
                "p99_ulp {} exceeds {} while absolute and relative limits are also exceeded",
                metrics.p99_ulp, self.max_p99_ulp,
            ));
        }
        ThresholdEvaluation {
            passed: failures.is_empty(),
            failures,
        }
    }

    pub fn evaluate_f32(
        &self,
        reference: &[f32],
        actual: &[f32],
        metrics: &AccuracyMetrics,
    ) -> Result<ThresholdEvaluation> {
        if reference.len() != actual.len() {
            return Err(Error::Verification(format!(
                "length mismatch while evaluating thresholds: reference has {}, actual has {}",
                reference.len(),
                actual.len()
            )));
        }
        let mut failures = self.common_failures(metrics);
        let mut first_max_violation = None;
        let mut p99_violations = 0_usize;
        let mut finite_samples = 0_usize;
        for (index, (&expected, &found)) in reference.iter().zip(actual).enumerate() {
            if !expected.is_finite() || !found.is_finite() {
                continue;
            }
            finite_samples += 1;
            let absolute = f64::from((expected - found).abs());
            let combined_limit = self
                .max_rel
                .mul_add(f64::from(expected.abs()), self.max_abs);
            if absolute <= combined_limit {
                continue;
            }
            let ulp = ulp_distance(expected, found);
            if ulp > self.max_ulp && first_max_violation.is_none() {
                first_max_violation = Some((index, absolute, combined_limit, ulp));
            }
            if ulp > self.max_p99_ulp {
                p99_violations += 1;
            }
        }
        if let Some((index, absolute, combined_limit, ulp)) = first_max_violation {
            failures.push(format!(
                "sample {index} exceeds combined abs/rel and ULP limits: {absolute} > {combined_limit}, {ulp} ULP > {}",
                self.max_ulp
            ));
        }
        let p99_outlier_budget = finite_samples.div_ceil(100);
        if p99_violations > p99_outlier_budget {
            failures.push(format!(
                "{p99_violations} samples exceed combined abs/rel and p99 ULP limits; at most {p99_outlier_budget} are allowed"
            ));
        }
        Ok(ThresholdEvaluation {
            passed: failures.is_empty(),
            failures,
        })
    }

    fn common_failures(&self, metrics: &AccuracyMetrics) -> Vec<String> {
        let mut failures = Vec::new();
        if metrics.nan_mismatches != 0 {
            failures.push(format!(
                "{} NaN classification mismatches",
                metrics.nan_mismatches
            ));
        }
        if metrics.infinity_mismatches != 0 {
            failures.push(format!(
                "{} infinity classification mismatches",
                metrics.infinity_mismatches
            ));
        }
        if self.require_exact && metrics.exact_match_count != metrics.sample_count {
            failures.push(format!(
                "only {}/{} samples matched exactly",
                metrics.exact_match_count, metrics.sample_count
            ));
        }
        if metrics.rmse > self.max_rmse {
            failures.push(format!("rmse {} exceeds {}", metrics.rmse, self.max_rmse));
        }
        if let (Some(minimum), Some(actual)) = (self.min_psnr_db, metrics.psnr_db)
            && actual < minimum
        {
            failures.push(format!("PSNR {actual} dB is below {minimum} dB"));
        }
        failures
    }
}

pub fn compare_f32(reference: &[f32], actual: &[f32], peak: f64) -> Result<AccuracyMetrics> {
    if reference.len() != actual.len() {
        return Err(Error::Verification(format!(
            "length mismatch: reference has {}, actual has {}",
            reference.len(),
            actual.len()
        )));
    }
    if !peak.is_finite() || peak <= 0.0 {
        return Err(Error::InvalidConfig(format!(
            "comparison peak must be finite and positive, found {peak}"
        )));
    }

    let mut metrics = AccuracyMetrics {
        sample_count: u64::try_from(reference.len()).unwrap_or(u64::MAX),
        ..AccuracyMetrics::default()
    };
    let mut sum_abs = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    let mut ulps = Vec::with_capacity(reference.len());

    for (index, (&expected, &found)) in reference.iter().zip(actual).enumerate() {
        if expected.to_bits() == found.to_bits() {
            metrics.exact_match_count += 1;
        } else if metrics.first_mismatch_index.is_none() {
            metrics.first_mismatch_index = Some(u64::try_from(index).unwrap_or(u64::MAX));
        }

        if expected.is_nan() || found.is_nan() {
            if !(expected.is_nan() && found.is_nan()) {
                metrics.nan_mismatches += 1;
            }
            continue;
        }
        if expected.is_infinite() || found.is_infinite() {
            if expected != found {
                metrics.infinity_mismatches += 1;
            }
            continue;
        }

        metrics.finite_sample_count += 1;
        let expected = f64::from(expected);
        let found = f64::from(found);
        let absolute = (expected - found).abs();
        let relative = absolute / expected.abs().max(f64::from(f32::MIN_POSITIVE));
        metrics.max_abs = metrics.max_abs.max(absolute);
        metrics.max_rel = metrics.max_rel.max(relative);
        sum_abs += absolute;
        sum_squared = absolute.mul_add(absolute, sum_squared);
        let ulp = ulp_distance(reference[index], actual[index]);
        metrics.max_ulp = metrics.max_ulp.max(ulp);
        ulps.push(ulp);
    }

    if metrics.finite_sample_count != 0 {
        let count = metrics.finite_sample_count as f64;
        metrics.mean_abs = sum_abs / count;
        metrics.rmse = (sum_squared / count).sqrt();
        metrics.psnr_db = if metrics.rmse == 0.0 {
            Some(f64::INFINITY)
        } else {
            Some(20.0 * (peak / metrics.rmse).log10())
        };
    }
    if !ulps.is_empty() {
        ulps.sort_unstable();
        let p99_index = (ulps.len() - 1) * 99 / 100;
        metrics.p99_ulp = ulps[p99_index];
    }
    Ok(metrics)
}

pub fn compare_u8(reference: &[u8], actual: &[u8]) -> Result<AccuracyMetrics> {
    compare_integer(
        reference.iter().copied().map(i64::from),
        actual.iter().copied().map(i64::from),
        reference.len(),
        255.0,
    )
}

pub fn compare_u16(reference: &[u16], actual: &[u16]) -> Result<AccuracyMetrics> {
    compare_integer(
        reference.iter().copied().map(i64::from),
        actual.iter().copied().map(i64::from),
        reference.len(),
        65_535.0,
    )
}

fn compare_integer(
    reference: impl Iterator<Item = i64>,
    actual: impl Iterator<Item = i64>,
    expected_len: usize,
    peak: f64,
) -> Result<AccuracyMetrics> {
    let reference = reference.collect::<Vec<_>>();
    let actual = actual.collect::<Vec<_>>();
    if reference.len() != expected_len || actual.len() != expected_len {
        return Err(Error::Verification(format!(
            "integer length mismatch: expected {expected_len}, got {}/{}",
            reference.len(),
            actual.len()
        )));
    }
    let mut metrics = AccuracyMetrics {
        sample_count: u64::try_from(expected_len).unwrap_or(u64::MAX),
        finite_sample_count: u64::try_from(expected_len).unwrap_or(u64::MAX),
        ..AccuracyMetrics::default()
    };
    let mut sum_abs = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    for (index, (&expected, &found)) in reference.iter().zip(&actual).enumerate() {
        let delta = expected.abs_diff(found);
        if delta == 0 {
            metrics.exact_match_count += 1;
        } else if metrics.first_mismatch_index.is_none() {
            metrics.first_mismatch_index = Some(u64::try_from(index).unwrap_or(u64::MAX));
        }
        let delta = delta as f64;
        metrics.max_abs = metrics.max_abs.max(delta);
        metrics.max_ulp = metrics
            .max_ulp
            .max(u32::try_from(delta as u64).unwrap_or(u32::MAX));
        sum_abs += delta;
        sum_squared = delta.mul_add(delta, sum_squared);
    }
    if expected_len != 0 {
        let count = expected_len as f64;
        metrics.mean_abs = sum_abs / count;
        metrics.rmse = (sum_squared / count).sqrt();
        metrics.psnr_db = if metrics.rmse == 0.0 {
            Some(f64::INFINITY)
        } else {
            Some(20.0 * (peak / metrics.rmse).log10())
        };
    }
    metrics.p99_ulp = metrics.max_ulp;
    Ok(metrics)
}

pub fn ulp_distance(left: f32, right: f32) -> u32 {
    if left.is_nan() || right.is_nan() {
        return u32::MAX;
    }
    ordered_f32(left)
        .abs_diff(ordered_f32(right))
        .min(u64::from(u32::MAX)) as u32
}

fn ordered_f32(value: f32) -> i64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        i64::from(bits | 0x8000_0000)
    } else {
        i64::from(!bits)
    }
}

pub fn total_order_f64(left: f64, right: f64) -> Ordering {
    left.total_cmp(&right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_values_have_zero_error() {
        let metrics = compare_f32(&[0.0, -0.0, 1.0], &[0.0, -0.0, 1.0], 1.0).unwrap();
        assert_eq!(metrics.exact_match_count, 3);
        assert_eq!(metrics.max_abs, 0.0);
        assert_eq!(metrics.max_ulp, 0);
        assert_eq!(metrics.psnr_db, Some(f64::INFINITY));
    }

    #[test]
    fn adjacent_float_is_one_ulp() {
        let one = 1.0_f32;
        let next = f32::from_bits(one.to_bits() + 1);
        assert_eq!(ulp_distance(one, next), 1);
        assert_eq!(ulp_distance(-one, -next), 1);
        assert_eq!(ulp_distance(-0.0, 0.0), 1);
    }

    #[test]
    fn classification_mismatches_are_counted() {
        let metrics = compare_f32(
            &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            &[0.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
            1.0,
        )
        .unwrap();
        assert_eq!(metrics.nan_mismatches, 1);
        assert_eq!(metrics.infinity_mismatches, 1);
    }

    #[test]
    fn strict_threshold_rejects_difference() {
        let metrics = compare_f32(&[1.0], &[1.1], 1.0).unwrap();
        let evaluation = AccuracyThreshold {
            max_abs: 0.01,
            max_rel: 0.01,
            max_rmse: 0.01,
            max_ulp: 1,
            max_p99_ulp: 1,
            min_psnr_db: Some(80.0),
            require_exact: true,
        }
        .evaluate(&metrics);
        assert!(!evaluation.passed);
        assert!(!evaluation.failures.is_empty());
    }

    #[test]
    fn combined_tolerance_accepts_small_absolute_error_near_zero() {
        let expected = [1.0e-8_f32];
        let actual = [2.0e-8_f32];
        let metrics = compare_f32(&expected, &actual, 1.0).unwrap();
        let evaluation = AccuracyThreshold {
            max_abs: 2.0e-8,
            max_rel: 1.0e-6,
            max_rmse: 2.0e-8,
            max_ulp: 0,
            max_p99_ulp: 0,
            min_psnr_db: None,
            require_exact: false,
        }
        .evaluate_f32(&expected, &actual, &metrics)
        .unwrap();
        assert!(evaluation.passed, "{:?}", evaluation.failures);
    }

    #[test]
    fn combined_tolerance_rejects_large_error() {
        let expected = [1.0_f32];
        let actual = [1.1_f32];
        let metrics = compare_f32(&expected, &actual, 1.0).unwrap();
        let evaluation = AccuracyThreshold {
            max_abs: 0.01,
            max_rel: 0.01,
            max_rmse: 1.0,
            max_ulp: 1,
            max_p99_ulp: 1,
            min_psnr_db: None,
            require_exact: false,
        }
        .evaluate_f32(&expected, &actual, &metrics)
        .unwrap();
        assert!(!evaluation.passed);
        assert!(
            evaluation
                .failures
                .iter()
                .any(|failure| failure.contains("combined abs/rel"))
        );
    }

    #[test]
    fn integer_metrics_use_lsb_units() {
        let metrics = compare_u8(&[0, 10, 255], &[0, 11, 253]).unwrap();
        assert_eq!(metrics.max_abs, 2.0);
        assert_eq!(metrics.exact_match_count, 1);
    }
}
