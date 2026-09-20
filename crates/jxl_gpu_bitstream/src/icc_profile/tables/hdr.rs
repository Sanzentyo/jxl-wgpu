// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Fixed 9x9x9 ICC metadata LUT from libjxl 0.12.0, BSD-3-Clause.
// This generates a profile table; no image or decoded sample is processed here.
use super::{Encoding, Result, Transfer, Writer, hlg_decode, pq_decode, pq_encode};

fn tone(rgb: &mut [f32; 3], weights: [f32; 3]) {
    let pq_min = pq_encode(1.0, 0.0);
    let pq_max = pq_encode(1.0, 10000.0);
    let range = pq_max - pq_min;
    let inv_range = 1.0 / range;
    let min_lum = (pq_encode(1.0, 0.0) - pq_min) * inv_range;
    let max_lum = (pq_encode(1.0, 250.0) - pq_min) * inv_range;
    let ks = 1.5 * max_lum - 0.5;
    let luminance = 10000.0 * (weights[0] * rgb[0] + weights[1] * rgb[1] + weights[2] * rgb[2]);
    let normalized = ((pq_encode(1.0, f64::from(luminance)) - pq_min) * inv_range).min(1.0);
    let e2 = if normalized < ks {
        normalized
    } else {
        let t = (normalized - ks) * (1.0 / (1.0 - ks).max(1e-6));
        let t2 = t * t;
        let t3 = t2 * t;
        (2.0 * t3 - 3.0 * t2 + 1.0) * ks
            + (t3 - 2.0 * t2 + t) * (1.0 - ks)
            + (-2.0 * t3 + 3.0 * t2) * max_lum
    };
    let one = 1.0 - e2;
    let square = one * one;
    let e3 = min_lum * (square * square) + e2;
    let d4 = pq_decode(1.0, f64::from(e3 * range + pq_min)).clamp(0.0, 250.0);
    let multiplier = (d4 / luminance.max(1e-6)) * (10000.0 / 250.0);
    for value in rgb {
        *value = if luminance <= 1e-6 {
            d4 * (1.0 / 250.0)
        } else {
            *value * multiplier
        };
    }
}

fn gamut(rgb: &mut [f32; 3], weights: [f32; 3]) {
    let luminance = weights[0] * rgb[0] + weights[1] * rgb[1] + weights[2] * rgb[2];
    let (mut saturation, mut light) = (0.0f32, 0.0f32);
    for value in *rgb {
        let delta = value - luminance;
        let reciprocal = 1.0 / if delta == 0.0 { 1.0 } else { delta };
        let divided = value * reciprocal;
        if delta < 0.0 {
            saturation = saturation.max(divided);
        }
        light = light.max(if delta <= 0.0 {
            saturation
        } else {
            divided - reciprocal
        });
    }
    let mix = (0.3 * (saturation - light) + light).clamp(0.0, 1.0);
    for value in &mut *rgb {
        *value = mix * (luminance - *value) + *value;
    }
    let reciprocal = 1.0 / rgb.iter().copied().fold(1.0f32, f32::max);
    for value in rgb {
        *value *= reciprocal;
    }
}

fn lab(value: f32) -> f32 {
    let delta = (6.0f64 / 29.0) as f32;
    if value <= delta * delta * delta {
        value * (1.0 / (3.0 * delta * delta)) + 4.0 / 29.0
    } else {
        value.cbrt()
    }
}

pub(in super::super) fn write(w: &mut Writer, e: &Encoding) -> Result<()> {
    w.append(b"mft1");
    w.u32(0);
    w.append(&[3, 3, 9, 0]);
    for r in 0..3 {
        for c in 0..3 {
            w.fixed(if r == c { 1.0 } else { 0.0 })?;
        }
    }
    for _ in 0..3 {
        for value in 0..=255 {
            w.u8(value);
        }
    }
    let weights = e.original_primaries[1];
    let hlg_exponent = 1.111f32.powf((80.0f32 / 300.0).log2()) - 1.0;
    for x in 0..9 {
        for y in 0..9 {
            for z in 0..9 {
                let input = [x, y, z].map(|v| f64::from(v as f32 * (1.0 / 8.0)));
                let mut rgb = input.map(|v| {
                    if e.transfer == Transfer::Pq {
                        pq_decode(10000.0, v)
                    } else {
                        hlg_decode(v)
                    }
                });
                if e.transfer == Transfer::Pq {
                    tone(&mut rgb, weights);
                } else {
                    let luminance = weights[0] * rgb[0] + weights[1] * rgb[1] + weights[2] * rgb[2];
                    let ratio = luminance.powf(hlg_exponent).min(1e9);
                    for value in &mut rgb {
                        *value *= ratio;
                    }
                }
                gamut(&mut rgb, weights);
                let xyz = e.primaries.map(|row| {
                    row.into_iter()
                        .zip(rgb)
                        .fold(0.0f32, |sum, (a, b)| sum + a * b)
                });
                let [fx, fy, fz] = [lab(xyz[0] / 0.964212), lab(xyz[1]), lab(xyz[2] / 0.825188)];
                w.u8((255.0 * (1.16 * fy - 0.16).clamp(0.0, 1.0)).round() as u8);
                w.u8((128.0 + (500.0 * (fx - fy)).clamp(-128.0, 127.0)).round() as u8);
                w.u8((128.0 + (200.0 * (fy - fz)).clamp(-128.0, 127.0)).round() as u8);
            }
        }
    }
    for _ in 0..3 {
        for value in 0..=255 {
            w.u8(value);
        }
    }
    Ok(())
}
