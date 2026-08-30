// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Accuracy gates and phase-level timing summaries used by validation harnesses.

use std::time::Duration;

use jxl_gpu_protocol::PrecisionContract;

/// Full-reference error measurements for one output plane or image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccuracyMetrics {
    pub samples: u64,
    pub mismatched_samples: u64,
    pub non_finite_mismatches: u64,
    pub max_absolute_error: f64,
    pub mean_absolute_error: f64,
    pub max_relative_error: f64,
    pub rmse: f64,
    pub psnr_db: f64,
    pub max_lsb_error: u16,
    pub signal_peak: f64,
}

impl AccuracyMetrics {
    pub fn compare_f32(reference: &[f32], actual: &[f32], signal_peak: f32) -> Option<Self> {
        if reference.len() != actual.len()
            || reference.is_empty()
            || !signal_peak.is_finite()
            || signal_peak <= 0.0
        {
            return None;
        }

        let mut accumulator = AccuracyAccumulator::new(f64::from(signal_peak));
        for (&reference, &actual) in reference.iter().zip(actual) {
            if reference.to_bits() == actual.to_bits() {
                accumulator.push(0.0, 0.0, false, 0);
            } else if reference.is_finite() && actual.is_finite() {
                let reference = f64::from(reference);
                let actual = f64::from(actual);
                let absolute = (reference - actual).abs();
                let denominator = reference.abs().max(actual.abs()).max(f64::EPSILON);
                accumulator.push(absolute, absolute / denominator, false, u16::MAX);
            } else {
                accumulator.push(f64::INFINITY, f64::INFINITY, true, u16::MAX);
            }
        }
        Some(accumulator.finish())
    }

    pub fn compare_u16(reference: &[u16], actual: &[u16], signal_peak: u16) -> Option<Self> {
        if reference.len() != actual.len() || reference.is_empty() || signal_peak == 0 {
            return None;
        }

        let mut accumulator = AccuracyAccumulator::new(f64::from(signal_peak));
        for (&reference, &actual) in reference.iter().zip(actual) {
            let absolute = reference.abs_diff(actual);
            let denominator = f64::from(reference.max(actual)).max(1.0);
            accumulator.push(
                f64::from(absolute),
                f64::from(absolute) / denominator,
                false,
                absolute,
            );
        }
        Some(accumulator.finish())
    }

    pub fn meets(self, contract: PrecisionContract) -> bool {
        match contract {
            PrecisionContract::Exact => self.mismatched_samples == 0,
            PrecisionContract::Float {
                absolute,
                relative,
                rmse,
            } => {
                self.non_finite_mismatches == 0
                    && self.max_absolute_error <= f64::from(absolute)
                    && self.max_relative_error <= f64::from(relative)
                    && self.rmse <= f64::from(rmse)
            }
            PrecisionContract::Perceptual { max_lsb, min_psnr } => {
                self.non_finite_mismatches == 0
                    && self.max_lsb_error <= max_lsb
                    && self.psnr_db >= f64::from(min_psnr)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct AccuracyAccumulator {
    samples: u64,
    mismatched_samples: u64,
    non_finite_mismatches: u64,
    max_absolute_error: f64,
    absolute_error_sum: f64,
    max_relative_error: f64,
    squared_error_sum: f64,
    max_lsb_error: u16,
    signal_peak: f64,
}

impl AccuracyAccumulator {
    const fn new(signal_peak: f64) -> Self {
        Self {
            samples: 0,
            mismatched_samples: 0,
            non_finite_mismatches: 0,
            max_absolute_error: 0.0,
            absolute_error_sum: 0.0,
            max_relative_error: 0.0,
            squared_error_sum: 0.0,
            max_lsb_error: 0,
            signal_peak,
        }
    }

    fn push(&mut self, absolute: f64, relative: f64, non_finite: bool, lsb: u16) {
        self.samples += 1;
        if absolute != 0.0 {
            self.mismatched_samples += 1;
        }
        self.non_finite_mismatches += u64::from(non_finite);
        self.max_absolute_error = self.max_absolute_error.max(absolute);
        self.absolute_error_sum += absolute;
        self.max_relative_error = self.max_relative_error.max(relative);
        self.squared_error_sum += absolute * absolute;
        self.max_lsb_error = self.max_lsb_error.max(lsb);
    }

    fn finish(self) -> AccuracyMetrics {
        let samples = self.samples as f64;
        let rmse = (self.squared_error_sum / samples).sqrt();
        let psnr_db = if rmse == 0.0 {
            f64::INFINITY
        } else {
            20.0 * (self.signal_peak / rmse).log10()
        };
        AccuracyMetrics {
            samples: self.samples,
            mismatched_samples: self.mismatched_samples,
            non_finite_mismatches: self.non_finite_mismatches,
            max_absolute_error: self.max_absolute_error,
            mean_absolute_error: self.absolute_error_sum / samples,
            max_relative_error: self.max_relative_error,
            rmse,
            psnr_db,
            max_lsb_error: self.max_lsb_error,
            signal_peak: self.signal_peak,
        }
    }
}

/// Host and GPU time attributable to each submission phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimingBreakdown {
    pub planning_ns: u64,
    pub upload_ns: u64,
    pub encoding_ns: u64,
    pub execution_ns: u64,
    pub readback_ns: u64,
    pub total_ns: u64,
}

impl TimingBreakdown {
    pub fn from_durations(
        planning: Duration,
        upload: Duration,
        encoding: Duration,
        execution: Duration,
        readback: Duration,
        total: Duration,
    ) -> Self {
        Self {
            planning_ns: duration_ns(planning),
            upload_ns: duration_ns(upload),
            encoding_ns: duration_ns(encoding),
            execution_ns: duration_ns(execution),
            readback_ns: duration_ns(readback),
            total_ns: duration_ns(total),
        }
    }

    pub fn measured_phase_ns(self) -> u64 {
        self.planning_ns
            .saturating_add(self.upload_ns)
            .saturating_add(self.encoding_ns)
            .saturating_add(self.execution_ns)
            .saturating_add(self.readback_ns)
    }

    pub fn unaccounted_ns(self) -> u64 {
        self.total_ns.saturating_sub(self.measured_phase_ns())
    }

    pub fn accumulate(&mut self, other: Self) {
        self.planning_ns = self.planning_ns.saturating_add(other.planning_ns);
        self.upload_ns = self.upload_ns.saturating_add(other.upload_ns);
        self.encoding_ns = self.encoding_ns.saturating_add(other.encoding_ns);
        self.execution_ns = self.execution_ns.saturating_add(other.execution_ns);
        self.readback_ns = self.readback_ns.saturating_add(other.readback_ns);
        self.total_ns = self.total_ns.saturating_add(other.total_ns);
    }

    pub fn averaged(self, samples: u64) -> Option<Self> {
        (samples != 0).then(|| Self {
            planning_ns: self.planning_ns / samples,
            upload_ns: self.upload_ns / samples,
            encoding_ns: self.encoding_ns / samples,
            execution_ns: self.execution_ns / samples,
            readback_ns: self.readback_ns / samples,
            total_ns: self.total_ns / samples,
        })
    }
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_float_comparison_passes_exact_contract() {
        let metrics =
            AccuracyMetrics::compare_f32(&[0.0, 0.5, 1.0], &[0.0, 0.5, 1.0], 1.0).unwrap();
        assert_eq!(metrics.mismatched_samples, 0);
        assert_eq!(metrics.rmse, 0.0);
        assert!(metrics.psnr_db.is_infinite());
        assert!(metrics.meets(PrecisionContract::Exact));
    }

    #[test]
    fn float_metrics_apply_all_thresholds() {
        let metrics = AccuracyMetrics::compare_f32(&[0.0, 1.0], &[0.01, 0.98], 1.0).unwrap();
        assert!((metrics.max_absolute_error - 0.02).abs() < 1.0e-6);
        assert!(metrics.meets(PrecisionContract::Float {
            absolute: 0.021,
            relative: 1.0,
            rmse: 0.02,
        }));
        assert!(!metrics.meets(PrecisionContract::Float {
            absolute: 0.01,
            relative: 1.0,
            rmse: 0.02,
        }));
    }

    #[test]
    fn integer_metrics_track_lsb_error() {
        let metrics = AccuracyMetrics::compare_u16(&[0, 100, 200], &[1, 98, 200], 255).unwrap();
        assert_eq!(metrics.max_lsb_error, 2);
        assert!(metrics.meets(PrecisionContract::Perceptual {
            max_lsb: 2,
            min_psnr: 30.0,
        }));
        assert!(!metrics.meets(PrecisionContract::Perceptual {
            max_lsb: 1,
            min_psnr: 30.0,
        }));
    }

    #[test]
    fn non_finite_mismatch_fails_float_contract() {
        let metrics = AccuracyMetrics::compare_f32(&[f32::INFINITY], &[0.0], 1.0).unwrap();
        assert_eq!(metrics.non_finite_mismatches, 1);
        assert!(!metrics.meets(PrecisionContract::Float {
            absolute: f32::MAX,
            relative: f32::MAX,
            rmse: f32::MAX,
        }));
    }

    #[test]
    fn timings_accumulate_and_average() {
        let mut timings = TimingBreakdown {
            planning_ns: 1,
            upload_ns: 2,
            encoding_ns: 3,
            execution_ns: 4,
            readback_ns: 5,
            total_ns: 20,
        };
        timings.accumulate(timings);
        assert_eq!(timings.unaccounted_ns(), 10);
        assert_eq!(timings.averaged(2).unwrap().execution_ns, 4);
        assert_eq!(timings.averaged(0), None);
    }
}
