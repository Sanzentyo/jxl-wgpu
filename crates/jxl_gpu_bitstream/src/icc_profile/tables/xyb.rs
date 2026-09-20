// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Scaled-XYB profile metadata from libjxl 0.12.0, BSD-3-Clause.
use super::{Result, Writer};

pub(in super::super) fn write(w: &mut Writer) -> Result<()> {
    w.append(b"mAB ");
    w.u32(0);
    w.append(&[3, 3, 0, 0]);
    for offset in [32, 244, 148, 80, 32] {
        w.u32(offset);
    }
    for _ in 0..3 {
        w.para(0, &[1.0])?;
    }
    w.append(&[2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]);
    let offset = [0.015_386_134_f32, 0.0, 0.277_704_6];
    let scale = [22.995_789_f32, 1.183_000_1, 1.502_141_4];
    let offsets = [
        offset[0] + offset[1],
        offset[1] - offset[0] + 1.0 / scale[0],
        offset[1] + offset[2],
    ];
    let scales = [
        (scale[0] * scale[1]) / (scale[0] + scale[1]),
        (scale[0] * scale[1]) / (scale[0] + scale[1]),
        (scale[1] * scale[2]) / (scale[1] + scale[2]),
    ];
    for x in 0..2 {
        for y in 0..2 {
            for b in 0..2 {
                let input = [x, y, b];
                let corner: [f32; 3] =
                    std::array::from_fn(|i| input[i] as f32 / scale[i] - offset[i]);
                let values = [
                    corner[1] + corner[0],
                    corner[1] - corner[0],
                    corner[2] + corner[1],
                ];
                for i in 0..3 {
                    w.u16((65535.0 * ((values[i] + offsets[i]) * scales[i])).round() as u16);
                }
            }
        }
    }
    let bias = -0.003_793_073_4_f32;
    for i in 0..3 {
        let b = -offsets[i] - bias.cbrt();
        w.para(
            3,
            &[3.0, 1.0 / scales[i], b, 0.0, (-b * scales[i]).max(0.0)],
        )?;
    }
    let matrix = [
        [1.5170095f64, -1.1065225, 0.071623],
        [-0.050022, 0.5683655, -0.018344],
        [-1.387676, 1.1145555, 0.6857255],
    ];
    for value in matrix.into_iter().flatten() {
        w.fixed(value as f32)?;
    }
    for row in matrix {
        let intercept = row.into_iter().fold(0.0f32, |sum, v| {
            (f64::from(sum) + v * f64::from(bias)) as f32
        });
        w.fixed(intercept)?;
    }
    Ok(())
}
