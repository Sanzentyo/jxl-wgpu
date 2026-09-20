// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// libjxl 0.12.0 profile descriptions and metadata normalization, BSD-3-Clause.
use std::fmt::Write;

use crate::{
    ChromaticityInventory as Point, ColourEncodingInventory, ColourSpaceInventory as Space,
    PrimariesInventory as Primaries, RenderingIntentInventory as Intent,
    TransferFunctionInventory as Transfer, WhitePointInventory as White,
};

use super::{IccProfileError as Error, Result, math};

pub(super) struct Text {
    bytes: [u8; 256],
    len: usize,
}
impl Text {
    fn new() -> Self {
        Self {
            bytes: [0; 256],
            len: 0,
        }
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
    fn as_str(&self) -> &str {
        std::str::from_utf8(self.as_bytes()).expect("ASCII metadata")
    }
}
impl Write for Text {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let end = self.len.checked_add(text.len()).ok_or(std::fmt::Error)?;
        self.bytes
            .get_mut(self.len..end)
            .ok_or(std::fmt::Error)?
            .copy_from_slice(text.as_bytes());
        self.len = end;
        Ok(())
    }
}

/// C's locale-independent %g convention, six significant digits. Inputs are finite metadata.
fn number(text: &mut Text, value: f64) -> std::fmt::Result {
    let mut scientific = Text::new();
    write!(scientific, "{value:.5e}")?;
    let (mantissa, exponent) = scientific.as_str().split_once('e').ok_or(std::fmt::Error)?;
    let exponent: i32 = exponent.parse().map_err(|_| std::fmt::Error)?;
    if !(-4..6).contains(&exponent) {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        write!(text, "{mantissa}e{exponent:+03}")
    } else {
        let mut fixed = Text::new();
        let precision = (5 - exponent).max(0) as usize;
        write!(fixed, "{value:.precision$}")?;
        let value = fixed.as_str();
        text.write_str(if value.contains('.') {
            value.trim_end_matches('0').trim_end_matches('.')
        } else {
            value
        })
    }
}

pub(super) struct Encoding {
    pub space: Space,
    pub intent: Intent,
    pub transfer: Transfer,
    pub gamma: f64,
    pub description: Text,
    pub white: [f32; 3],
    pub adaptation: math::Matrix,
    pub primaries: math::Matrix,
    pub original_primaries: math::Matrix,
    pub cicp: Option<[u8; 4]>,
    pub hdr: bool,
}

fn point(point: Point) -> Result<[f64; 2]> {
    if [point.x, point.y]
        .into_iter()
        .any(|v| !(-0x200000..=0x1fffff).contains(&v))
    {
        return Err(Error::Invalid(
            "chromaticity outside JPEG XL coordinate range",
        ));
    }
    Ok([f64::from(point.x) * 1e-6, f64::from(point.y) * 1e-6])
}

impl Encoding {
    pub fn new(colour: ColourEncodingInventory) -> Result<Self> {
        let ColourEncodingInventory::Enumerated {
            colour_space: space,
            white_point: white,
            primaries,
            transfer_function: transfer,
            rendering_intent: intent,
        } = colour
        else {
            return Err(Error::Unsupported(
                "embedded-profile declaration without bytes",
            ));
        };
        // XYB has implicit white, primaries and transfer; Gray has no primaries.
        let (white, primaries, transfer) = if space == Space::Xyb {
            (
                White::D65,
                Primaries::Srgb,
                Transfer::Gamma {
                    scaled_gamma: 3_333_333,
                    inverted: true,
                },
            )
        } else if space == Space::Grey {
            (white, Primaries::Srgb, transfer)
        } else {
            (white, primaries, transfer)
        };
        if space == Space::Unknown || transfer == Transfer::Unknown {
            return Err(Error::Unsupported(
                "unknown color space or transfer function",
            ));
        }
        if space == Space::Xyb && intent != Intent::Perceptual {
            return Err(Error::Unsupported("non-perceptual XYB profile"));
        }
        let wp = match white {
            White::D65 => [0.3127, 0.3290],
            White::E => [1.0 / 3.0, 1.0 / 3.0],
            White::Dci => [0.314, 0.351],
            White::Custom(p) => point(p)?,
        };
        let rgb = match primaries {
            Primaries::Srgb => [
                [0.639998686, 0.330010138],
                [0.300003784, 0.600003357],
                [0.150002046, 0.059997204],
            ],
            Primaries::Bt2100 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
            Primaries::P3 => [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
            Primaries::Custom { red, green, blue } => [point(red)?, point(green)?, point(blue)?],
        };
        let gamma = match transfer {
            Transfer::Gamma {
                scaled_gamma,
                inverted,
            } => {
                if scaled_gamma == 0 {
                    return Err(Error::Invalid("zero gamma"));
                }
                let value = f64::from(scaled_gamma) * 1e-7;
                let gamma = if inverted { value } else { 1.0 / value };
                if !(1.0 / 8192.0..=1.0).contains(&gamma) {
                    return Err(Error::Invalid("gamma outside JPEG XL range"));
                }
                gamma
            }
            _ => 0.0,
        };
        let description = description(space, white, primaries, transfer, intent, wp, rgb, gamma)
            .map_err(|_| Error::Invalid("profile description"))?;
        let adaptation = if space == Space::Grey {
            [[0.0; 3]; 3]
        } else {
            math::adaptation(wp)?
        };
        let original_primaries = if space == Space::Rgb {
            math::primaries(rgb, wp)?
        } else {
            [[0.0; 3]; 3]
        };
        let white_value = if space == Space::Grey {
            math::grey_white(wp)?
        } else {
            [0.964203, 1.0, 0.824905]
        };
        let cp = match (white, primaries) {
            (White::D65, Primaries::Srgb) => Some(1),
            (White::D65, Primaries::Bt2100) => Some(9),
            (White::D65, Primaries::P3) => Some(12),
            (White::Dci, Primaries::P3) => Some(11),
            _ => None,
        };
        let ct = match transfer {
            Transfer::Bt709 => Some(1),
            Transfer::Linear => Some(8),
            Transfer::Srgb => Some(13),
            Transfer::Pq => Some(16),
            Transfer::Dci => Some(17),
            Transfer::Hlg => Some(18),
            _ => None,
        };
        let cicp = if space == Space::Rgb {
            cp.zip(ct).map(|(p, t)| [p, t, 0, 1])
        } else {
            None
        };
        Ok(Self {
            space,
            intent,
            transfer,
            gamma,
            description,
            white: white_value,
            adaptation,
            primaries: math::product(adaptation, original_primaries),
            original_primaries,
            cicp,
            hdr: cicp.is_some() && matches!(transfer, Transfer::Pq | Transfer::Hlg),
        })
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "finite color declaration fields form one profile description"
)]
fn description(
    space: Space,
    white: White,
    primaries: Primaries,
    transfer: Transfer,
    intent: Intent,
    wp: [f64; 2],
    rgb: [[f64; 2]; 3],
    gamma: f64,
) -> std::result::Result<Text, std::fmt::Error> {
    let mut text = Text::new();
    let close = |a: f64, b: f64| (a - b).abs() < 3e-5;
    let close_rgb = |target: [[f64; 2]; 3]| {
        rgb.into_iter()
            .flatten()
            .zip(target.into_iter().flatten())
            .all(|(a, b)| close(a, b))
    };
    let short = if space == Space::Rgb {
        match (white, primaries, transfer) {
            (White::D65, Primaries::Srgb, Transfer::Srgb) => Some("sRGB"),
            (White::D65, Primaries::P3, Transfer::Srgb) => Some("DisplayP3"),
            (White::D65, Primaries::Bt2100, Transfer::Pq) => Some("Rec2100PQ"),
            (White::D65, Primaries::Bt2100, Transfer::Hlg) => Some("Rec2100HLG"),
            (White::D65, Primaries::Custom { .. }, Transfer::Gamma { .. })
                if close_rgb([[0.64, 0.33], [0.21, 0.71], [0.15, 0.06]])
                    && close(gamma, 256.0 / 563.0) =>
            {
                Some("Adobe98")
            }
            (White::Custom(_), Primaries::Custom { .. }, Transfer::Gamma { .. })
                if close(wp[0], 0.345669)
                    && close(wp[1], 0.358496)
                    && close_rgb([
                        [0.734699, 0.265301],
                        [0.159597, 0.840403],
                        [0.036598, 0.000105],
                    ])
                    && close(gamma, 1.0 / 1.8) =>
            {
                Some("ProPhoto")
            }
            _ => None,
        }
    } else {
        None
    };
    if let Some(short) = short {
        text.write_str(short)?;
        return Ok(text);
    }
    text.write_str(match space {
        Space::Rgb => "RGB",
        Space::Grey => "Gra",
        Space::Xyb => "XYB",
        Space::Unknown => "CS?",
    })?;
    if space != Space::Xyb {
        text.write_char('_')?;
        match white {
            White::Custom(_) => {
                number(&mut text, wp[0])?;
                text.write_char(';')?;
                number(&mut text, wp[1])?;
            }
            White::D65 => text.write_str("D65")?,
            White::E => text.write_str("EER")?,
            White::Dci => text.write_str("DCI")?,
        }
    }
    if space == Space::Rgb {
        text.write_char('_')?;
        match primaries {
            Primaries::Custom { .. } => {
                for (i, value) in rgb.into_iter().flatten().enumerate() {
                    if i != 0 {
                        text.write_char(';')?;
                    }
                    number(&mut text, value)?;
                }
            }
            Primaries::Srgb => text.write_str("SRG")?,
            Primaries::Bt2100 => text.write_str("202")?,
            Primaries::P3 => text.write_str("DCI")?,
        }
    }
    text.write_char('_')?;
    text.write_str(match intent {
        Intent::Perceptual => "Per",
        Intent::Relative => "Rel",
        Intent::Saturation => "Sat",
        Intent::Absolute => "Abs",
    })?;
    if space != Space::Xyb {
        text.write_char('_')?;
        match transfer {
            Transfer::Gamma { .. } => {
                text.write_char('g')?;
                number(&mut text, gamma)?;
            }
            Transfer::Bt709 => text.write_str("709")?,
            Transfer::Linear => text.write_str("Lin")?,
            Transfer::Srgb => text.write_str("SRG")?,
            Transfer::Pq => text.write_str("PeQ")?,
            Transfer::Dci => text.write_str("DCI")?,
            Transfer::Hlg => text.write_str("HLG")?,
            Transfer::Unknown => text.write_str("TF?")?,
        }
    }
    Ok(text)
}
