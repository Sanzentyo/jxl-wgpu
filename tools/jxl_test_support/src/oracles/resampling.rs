//! Independent F64 interpolation and arithmetic bounds shared by codec conformance tests.
use jxl_gpu_bitstream::{SampleBitDepth, UpsamplingWeightsInventory};

#[derive(Clone, Copy)]
pub enum Arithmetic {
    Rust,
    Native,
    Wgsl,
}

impl Arithmetic {
    fn unit(self) -> f64 {
        // WGSL 15.7.4 permits either adjacent representable result. CPU oracles
        // use round-to-nearest. https://www.w3.org/TR/WGSL/#floating-point-accuracy
        2_f64.powi(if matches!(self, Self::Wgsl) { -23 } else { -24 })
    }

    fn normalization_error(self) -> f64 {
        let unit = self.unit();
        match self {
            Self::Rust => unit,
            // libjxl dec_modular.cc: F64 reciprocal -> F32 factor -> F32 multiply.
            Self::Native => (2.0 * unit + f64::EPSILON) / (1.0 - 2.0 * unit - f64::EPSILON),
            // WGSL F32 division permits 2.5 ULP; both integer operands here are
            // exactly representable. ULP(x) <= 2^-23 * abs(x) for normal x.
            Self::Wgsl => 2.5 * unit,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub value: f64,
    pub error: f64,
}

impl Sample {
    pub fn decoded(word: u32, depth: SampleBitDepth, arithmetic: Arithmetic) -> Self {
        match depth {
            SampleBitDepth::Integer { bits_per_sample } => {
                let value = f64::from(word) / ((1_u64 << bits_per_sample) - 1) as f64;
                Self {
                    value,
                    error: value.abs() * arithmetic.normalization_error(),
                }
            }
            SampleBitDepth::Float {
                bits_per_sample: 32,
                exponent_bits_per_sample: 8,
            } => Self {
                value: f64::from(f32::from_bits(word)),
                error: 0.0,
            },
            _ => panic!("unexpected fixture precision"),
        }
    }

    pub fn check(self, value: f32, context: &str, index: usize) {
        assert!(
            value.is_finite() && (f64::from(value) - self.value).abs() <= self.error,
            "{context} sample {index}: {value} vs {} +/- {}",
            self.value,
            self.error
        );
    }
}

pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub samples: Vec<Sample>,
}

impl Plane {
    pub fn reconstruct(
        mut self,
        factor: u32,
        width: usize,
        height: usize,
        weights: &UpsamplingWeightsInventory,
        arithmetic: Arithmetic,
    ) -> Self {
        let mut remaining = factor;
        while remaining > 1 {
            let step = remaining.min(8);
            self = self.filter(step, weights, arithmetic);
            remaining /= step;
        }
        self.crop(width, height)
    }

    pub fn crop(self, width: usize, height: usize) -> Self {
        assert!(width <= self.width && height <= self.height);
        let samples = self
            .samples
            .chunks_exact(self.width)
            .take(height)
            .flat_map(|row| row[..width].iter().copied())
            .collect();
        Self {
            width,
            height,
            samples,
        }
    }

    pub fn filter(
        &self,
        factor: u32,
        weights: &UpsamplingWeightsInventory,
        arithmetic: Arithmetic,
    ) -> Self {
        let compact: Vec<_> = match factor {
            2 => weights.up2.iter().collect(),
            4 => weights.up4.iter().collect(),
            8 => weights.up8.iter().collect(),
            _ => panic!("filter factor"),
        };
        let factor = factor as usize;
        let side = factor / 2 * 5;
        let mut symmetric = vec![vec![0.0; side]; side];
        let mut values = compact.into_iter();
        for (y, row) in symmetric.iter_mut().enumerate() {
            for value in row.iter_mut().skip(y) {
                *value = f64::from(values.next().unwrap().to_f32());
            }
        }
        for y in 0..side {
            let (above, remaining) = symmetric.split_at_mut(y);
            for (value, row) in remaining[0][..y].iter_mut().zip(above.iter()) {
                *value = row[y];
            }
        }
        assert!(values.next().is_none());
        let width = self.width * factor;
        let height = self.height * factor;
        let unit = arithmetic.unit();
        let gamma = 50.0 * unit / (1.0 - 50.0 * unit);
        let mut samples = Vec::with_capacity(width * height);
        let mirror = |coordinate: isize, size: usize| {
            let coordinate = coordinate.rem_euclid(2 * size as isize) as usize;
            if coordinate < size {
                coordinate
            } else {
                2 * size - coordinate - 1
            }
        };
        for y in 0..height {
            for x in 0..width {
                let phase_x = x % factor;
                let phase_y = y % factor;
                let mut sum = 0.0;
                let mut error = 0.0;
                let mut magnitude = 0.0;
                let mut minimum = [f64::INFINITY; 3];
                let mut maximum = [f64::NEG_INFINITY; 3];
                for dy in 0..5 {
                    for dx in 0..5 {
                        let sx = mirror((x / factor) as isize + dx as isize - 2, self.width);
                        let sy = mirror((y / factor) as isize + dy as isize - 2, self.height);
                        let sample = self.samples[sy * self.width + sx];
                        let row = phase_y.min(factor - phase_y - 1) * 5
                            + if phase_y < factor / 2 { dy } else { 4 - dy };
                        let col = phase_x.min(factor - phase_x - 1) * 5
                            + if phase_x < factor / 2 { dx } else { 4 - dx };
                        let weight = symmetric[row][col];
                        sum += sample.value * weight;
                        error += sample.error * weight.abs();
                        magnitude += (sample.value.abs() + sample.error) * weight.abs();
                        for (index, value) in [
                            sample.value - sample.error,
                            sample.value,
                            sample.value + sample.error,
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            minimum[index] = minimum[index].min(value);
                            maximum[index] = maximum[index].max(value);
                        }
                    }
                }
                // 25 products and 25 additions; covers any summation order or
                // FMA contraction, with a separate allowance for WGSL flushing.
                error += gamma * magnitude + 50.0 * f64::from(f32::MIN_POSITIVE);
                let value = sum.clamp(minimum[1], maximum[1]);
                let low = (sum - error).clamp(minimum[0], maximum[0]);
                let high = (sum + error).clamp(minimum[2], maximum[2]);
                samples.push(Sample {
                    value,
                    error: (value - low).max(high - value),
                });
            }
        }
        Self {
            width,
            height,
            samples,
        }
    }
}
