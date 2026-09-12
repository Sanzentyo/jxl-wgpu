//! Offline f64 LF presentation oracle. Native libjxl supplies restored small-image RGB;
//! invert its opsin transform, expand codec components, then transform for presentation.
//! This deliberately does not use the production renderer's phase tables or GPU helpers.
use jxl_gpu_bitstream::ImageHeaderInventory;

pub struct Planes {
    pub width: usize,
    pub height: usize,
    pub channels: Vec<Vec<f64>>,
}

fn product(matrix: [[f64; 3]; 3], values: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row.into_iter().zip(values).map(|(a, b)| a * b).sum())
}

fn inverse(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut rows: [[f64; 6]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if j < 3 {
                matrix[i][j]
            } else {
                f64::from(j - 3 == i)
            }
        })
    });
    for i in 0..3 {
        let pivot = (i..3)
            .max_by(|&a, &b| rows[a][i].abs().total_cmp(&rows[b][i].abs()))
            .unwrap();
        rows.swap(i, pivot);
        let divisor = rows[i][i];
        assert!(divisor.abs() > 1e-12, "invertible oracle opsin matrix");
        for value in &mut rows[i] {
            *value /= divisor;
        }
        let pivot = rows[i];
        for (k, row) in rows.iter_mut().enumerate() {
            if k == i {
                continue;
            }
            let scale = row[i];
            for (value, pivot) in row.iter_mut().zip(pivot) {
                *value -= scale * pivot;
            }
        }
    }
    rows.map(|row| row[3..].try_into().unwrap())
}

fn opsin(image: &ImageHeaderInventory) -> ([[f64; 3]; 3], [f64; 3], f64) {
    let opsin = image.opsin_inverse_matrix.unwrap();
    (
        opsin
            .inverse_matrix
            .map(|r| r.map(|v| f64::from(v.to_f32()))),
        opsin.opsin_bias.map(|v| f64::from(v.to_f32())),
        f64::from(image.tone_mapping.intensity_target.to_f32()) / 255.0,
    )
}

impl Planes {
    /// Packed linear RGBA followed by each full-resolution scalar extra plane.
    pub fn from_native(native: &[f32], image: &ImageHeaderInventory, extent: [u32; 2]) -> Self {
        let [width, height] = extent.map(|v| v as usize);
        let pixels = width * height;
        assert_eq!(native.len(), pixels * (4 + image.extra_channels.len()));
        let (matrix, bias, intensity) = opsin(image);
        let forward = inverse(matrix);
        let mut channels = vec![Vec::with_capacity(pixels); 3 + image.extra_channels.len()];
        for i in 0..pixels {
            let lms = product(
                forward,
                std::array::from_fn(|c| f64::from(native[i * 4 + c])),
            );
            let mixed: [f64; 3] =
                std::array::from_fn(|c| (lms[c] * intensity - bias[c]).cbrt() + bias[c].cbrt());
            for (plane, value) in channels.iter_mut().zip([
                (mixed[0] - mixed[1]) / 2.0,
                (mixed[0] + mixed[1]) / 2.0,
                mixed[2],
            ]) {
                plane.push(value);
            }
            for (c, plane) in channels[3..].iter_mut().enumerate() {
                plane.push(f64::from(native[(4 + c) * pixels + i]));
            }
        }
        Self {
            width,
            height,
            channels,
        }
    }

    /// Clip the consumer-local LF rectangle before every recursive 8x expansion.
    pub fn expand(&mut self, image: &ImageHeaderInventory, extent: [u32; 2], level: u8) {
        let [crop_w, crop_h] = extent.map(|v| v.div_ceil(1 << (3 * level)) as usize);
        assert!(crop_w <= self.width && crop_h <= self.height);
        self.channels = self
            .channels
            .iter()
            .map(|v| {
                v.chunks_exact(self.width)
                    .take(crop_h)
                    .flat_map(|row| row[..crop_w].iter().copied())
                    .collect()
            })
            .collect();
        self.width = crop_w;
        self.height = crop_h;
        for stage in (0..level).rev() {
            let [out_w, out_h] = extent.map(|v| v.div_ceil(1 << (3 * stage)) as usize);
            self.channels = self
                .channels
                .iter()
                .map(|v| up8(v, self.width, self.height, out_w, out_h, image))
                .collect();
            self.width = out_w;
            self.height = out_h;
        }
    }

    pub fn into_linear(mut self, image: &ImageHeaderInventory) -> Vec<Vec<f64>> {
        let (matrix, bias, intensity) = opsin(image);
        for i in 0..self.width * self.height {
            let mixed = [
                self.channels[1][i] + self.channels[0][i],
                self.channels[1][i] - self.channels[0][i],
                self.channels[2][i],
            ];
            let lms = std::array::from_fn(|c| {
                ((mixed[c] - bias[c].cbrt()).powi(3) + bias[c]) / intensity
            });
            for (plane, value) in self.channels.iter_mut().zip(product(matrix, lms)) {
                plane[i] = value;
            }
        }
        self.channels
    }
}

pub fn srgb(value: f64) -> f64 {
    let a = value.abs();
    value.signum()
        * if a <= 0.0031308 {
            a * 12.92
        } else {
            1.055 * a.powf(1.0 / 2.4) - 0.055
        }
}

fn up8(
    input: &[f64],
    width: usize,
    height: usize,
    out_w: usize,
    out_h: usize,
    image: &ImageHeaderInventory,
) -> Vec<f64> {
    let mirror = |v: i64, size: usize| {
        let p = size as i64 * 2;
        let v = v.rem_euclid(p);
        v.min(p - 1 - v) as usize
    };
    (0..out_h)
        .flat_map(|y| {
            (0..out_w).map(move |x| {
                let mut sum = 0.0;
                let mut low = f64::INFINITY;
                let mut high = f64::NEG_INFINITY;
                for row in 0..5 {
                    for col in 0..5 {
                        let value = input[mirror((y / 8) as i64 + row - 2, height) * width
                            + mirror((x / 8) as i64 + col - 2, width)];
                        let phase = |p: usize, tap: i64| {
                            p.min(7 - p) as i64 * 5 + if p < 4 { tap } else { 4 - tap }
                        };
                        let (a, b) = (phase(x % 8, col), phase(y % 8, row));
                        let (i, j) = (a.min(b), a.max(b));
                        sum += value
                            * f64::from(
                                image.upsampling_weights.up8[(i * (41 - i) / 2 + j - i) as usize]
                                    .to_f32(),
                            );
                        low = low.min(value);
                        high = high.max(value);
                    }
                }
                sum.clamp(low, high)
            })
        })
        .collect()
}
