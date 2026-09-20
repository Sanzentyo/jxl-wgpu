// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Fixed-size ICC metadata tables from libjxl 0.12.0, BSD-3-Clause.
use super::{Result, encoding::Encoding, writer::Writer};
use crate::TransferFunctionInventory as Transfer;

mod hdr;
mod xyb;
pub(super) use hdr::write as hdr;
pub(super) use xyb::write as xyb;

pub(super) fn pq_decode(display: f32, value: f64) -> f32 {
    if value == 0.0 {
        return 0.0;
    }
    let xp = value.powf(1.0 / (2523.0 / 32.0));
    let num = (xp - 3424.0 / 4096.0).max(0.0);
    let den = 2413.0 / 128.0 - 2392.0 / 128.0 * xp;
    ((num / den).powf(16384.0 / 2610.0) * f64::from(10000.0 / display)) as f32
}
pub(super) fn pq_encode(display: f32, value: f64) -> f32 {
    if value == 0.0 {
        return 0.0;
    }
    let xp = (value * f64::from(display * (1.0 / 10000.0))).powf(2610.0 / 16384.0);
    let num = 3424.0 / 4096.0 + xp * (2413.0 / 128.0);
    let den = 1.0 + xp * (2392.0 / 128.0);
    (num / den).powf(2523.0 / 32.0) as f32
}
pub(super) fn hlg_decode(value: f64) -> f32 {
    if value <= 0.5 {
        (value * value * (1.0 / 3.0)) as f32
    } else {
        ((((value - 0.5599107295) * (1.0 / 0.17883277)).exp() + (1.0 - 4.0 * 0.17883277))
            * (1.0 / 12.0)) as f32
    }
}

pub(super) fn curve(w: &mut Writer, e: &Encoding) -> Result<()> {
    match e.transfer {
        Transfer::Gamma { .. } => w.para(0, &[(1.0 / e.gamma) as f32]),
        Transfer::Srgb => w.para(
            3,
            &[
                2.4,
                (1.0f64 / 1.055) as f32,
                (0.055f64 / 1.055) as f32,
                (1.0f64 / 12.92) as f32,
                0.04045,
            ],
        ),
        Transfer::Bt709 => w.para(
            3,
            &[
                (1.0f64 / 0.45) as f32,
                (1.0f64 / 1.099) as f32,
                (0.099f64 / 1.099) as f32,
                (1.0f64 / 4.5) as f32,
                0.081,
            ],
        ),
        Transfer::Dci => w.para(3, &[2.6, 1.0, 0.0, 1.0, 0.0]),
        Transfer::Linear => w.para(3, &[1.0, 1.0, 0.0, 1.0, 0.0]),
        Transfer::Pq | Transfer::Hlg => {
            w.append(b"curv");
            w.u32(0);
            w.u32(64);
            for i in 0..64 {
                let x = f64::from(i as f32 / 63.0);
                let y = if e.transfer == Transfer::Pq {
                    pq_decode(10000.0, x)
                } else {
                    hlg_decode(x)
                };
                w.u16(((f64::from(y).clamp(0.0, 1.0) * 65535.0) as f32).round() as u16);
            }
            Ok(())
        }
        Transfer::Unknown => unreachable!("validated transfer"),
    }
}

pub(super) fn reverse(w: &mut Writer) -> Result<()> {
    w.append(b"mBA ");
    w.u32(0);
    w.append(&[3, 3, 0, 0]);
    for offset in [32, 0, 0, 0, 0] {
        w.u32(offset);
    }
    for _ in 0..3 {
        w.para(0, &[1.0])?;
    }
    Ok(())
}
