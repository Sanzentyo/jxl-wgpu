//! Bounded source-color metadata lowering; no image samples enter this module.

use jxl_gpu_bitstream::{
    BitWriter, ChromaticityInventory, PrimariesInventory, TransferFunctionInventory,
    WhitePointInventory,
};
use jxl_gpu_formats::{
    ColorModel, ColorRange, ColorSpecification, PixelFormat, TransferFunction, YcbcrEncoding,
};
use jxl_gpu_protocol::{
    Chromaticity, ColorMatrix, RgbChromaticities, RgbColorSpace, WhitePointAdaptation,
    icc::{IccProfile, IccRenderingIntent},
};

use crate::ColorChannels;
use crate::{EncodeError, UnsupportedFeature};

pub(crate) mod icc;
mod options;
pub use options::ImageColorOptions;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EnumeratedColorEncoding {
    white: WhitePointInventory,
    primaries: PrimariesInventory,
    transfer: TransferFunctionInventory,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SourceColorEncoding {
    Enumerated(EnumeratedColorEncoding),
    Icc(IccProfile),
}

impl Default for SourceColorEncoding {
    fn default() -> Self {
        Self::Enumerated(EnumeratedColorEncoding::default())
    }
}

impl SourceColorEncoding {
    pub(crate) fn from_format(format: &PixelFormat) -> Result<Self, EncodeError> {
        if let ColorSpecification::Icc(profile) = &format.color_spec {
            if matches!(
                (format.model, &profile.header().device_space.0),
                (ColorModel::Rgb | ColorModel::IccDevice, b"RGB ")
                    | (ColorModel::Gray | ColorModel::IccDevice, b"GRAY")
            ) {
                return Ok(Self::Icc(profile.clone()));
            }
            return Err(UnsupportedFeature::InputFormat.into());
        }
        let spec = match (&format.color_spec, format.model) {
            (ColorSpecification::Undefined, ColorModel::NonColor)
            | (
                ColorSpecification::Default | ColorSpecification::Undefined,
                ColorModel::Rgb | ColorModel::Gray,
            ) => return Ok(Self::default()),
            (ColorSpecification::Defined(spec), ColorModel::Rgb | ColorModel::Gray) => spec,
            _ => return Err(UnsupportedFeature::InputFormat.into()),
        };
        if spec.range != ColorRange::Full || spec.encoding != YcbcrEncoding::Undefined {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        let mut coordinates = spec
            .space
            .rgb_space()
            .and_then(RgbColorSpace::chromaticities)
            .ok_or(UnsupportedFeature::InputFormat)?;
        if format.model == ColorModel::Gray {
            coordinates = RgbChromaticities {
                white: coordinates.white,
                ..RgbChromaticities::BT709
            };
        }
        validate_geometry(coordinates)?;
        let white = match coordinates.white {
            Chromaticity::D65 => WhitePointInventory::D65,
            Chromaticity::E => WhitePointInventory::E,
            Chromaticity::DCI => WhitePointInventory::Dci,
            value => WhitePointInventory::Custom(quantize_xy(value)?),
        };
        let rgb = |value: RgbChromaticities| (value.red, value.green, value.blue);
        let primaries = if rgb(coordinates) == rgb(RgbChromaticities::BT709) {
            PrimariesInventory::Srgb
        } else if rgb(coordinates) == rgb(RgbChromaticities::BT2020) {
            PrimariesInventory::Bt2100
        } else if rgb(coordinates) == rgb(RgbChromaticities::DISPLAY_P3) {
            PrimariesInventory::P3
        } else {
            PrimariesInventory::Custom {
                red: quantize_xy(coordinates.red)?,
                green: quantize_xy(coordinates.green)?,
                blue: quantize_xy(coordinates.blue)?,
            }
        };
        // Quantization must not turn an otherwise valid declaration into a singular matrix.
        if let WhitePointInventory::Custom(value) = white {
            coordinates.white = expand_xy(value);
        }
        if let PrimariesInventory::Custom { red, green, blue } = primaries {
            coordinates.red = expand_xy(red);
            coordinates.green = expand_xy(green);
            coordinates.blue = expand_xy(blue);
        }
        validate_geometry(coordinates)?;
        let transfer = match spec.transfer {
            TransferFunction::Linear => TransferFunctionInventory::Linear,
            TransferFunction::Srgb | TransferFunction::Sycc => TransferFunctionInventory::Srgb,
            TransferFunction::Bt709 => TransferFunctionInventory::Bt709,
            TransferFunction::Pq => TransferFunctionInventory::Pq,
            TransferFunction::Hlg => TransferFunctionInventory::Hlg,
            TransferFunction::Dci => TransferFunctionInventory::Dci,
            TransferFunction::Gamma(value) => {
                let gamma = f64::from(value.value());
                if !(1.0 / 8192.0..=1.0).contains(&gamma) {
                    return Err(UnsupportedFeature::InputFormat.into());
                }
                TransferFunctionInventory::Gamma {
                    scaled_gamma: (gamma * 10_000_000.0).round() as u32,
                    inverted: true,
                }
            }
            _ => return Err(UnsupportedFeature::InputFormat.into()),
        };
        Ok(Self::Enumerated(EnumeratedColorEncoding {
            white,
            primaries,
            transfer,
        }))
    }

    pub(crate) fn icc_profile(&self) -> Option<&IccProfile> {
        match self {
            Self::Icc(profile) => Some(profile),
            Self::Enumerated(_) => None,
        }
    }

    pub(crate) fn write(
        &self,
        output: &mut BitWriter,
        channels: ColorChannels,
        intent: IccRenderingIntent,
    ) -> Result<(), EncodeError> {
        match self {
            Self::Icc(profile) => {
                if profile.header().rendering_intent != intent {
                    return Err(EncodeError::InvalidConfiguration(
                        "encoder intent must match the unchanged embedded ICC profile",
                    ));
                }
                output.write_bits(0, 1)?; // explicit color encoding
                output.write_bits(1, 1)?; // embedded ICC
                write_enum(output, u32::from(channels == ColorChannels::Gray))
            }
            Self::Enumerated(encoding) => encoding.write(output, channels, intent),
        }
    }
}

impl EnumeratedColorEncoding {
    /// Expand the serialized declaration, including rounded custom xy/gamma values.
    /// GPU conversion must use these values rather than the caller's unrounded metadata.
    pub(crate) fn rgb_encoding(self) -> jxl_gpu_protocol::RgbColorEncoding {
        use jxl_gpu_protocol::{GammaExponent, RgbColorEncoding, TransferFunction as Transfer};
        let mut coordinates = match self.primaries {
            PrimariesInventory::Srgb => RgbChromaticities::BT709,
            PrimariesInventory::Bt2100 => RgbChromaticities::BT2020,
            PrimariesInventory::P3 => RgbChromaticities::DISPLAY_P3,
            PrimariesInventory::Custom { red, green, blue } => RgbChromaticities {
                red: expand_xy(red),
                green: expand_xy(green),
                blue: expand_xy(blue),
                white: Chromaticity::D65,
            },
        };
        coordinates.white = match self.white {
            WhitePointInventory::D65 => Chromaticity::D65,
            WhitePointInventory::E => Chromaticity::E,
            WhitePointInventory::Dci => Chromaticity::DCI,
            WhitePointInventory::Custom(value) => expand_xy(value),
        };
        let space = match coordinates {
            RgbChromaticities::BT709 => RgbColorSpace::Bt709,
            RgbChromaticities::BT2020 => RgbColorSpace::Bt2020,
            RgbChromaticities::DISPLAY_P3 => RgbColorSpace::DisplayP3,
            value => RgbColorSpace::Custom(value),
        };
        let transfer = match self.transfer {
            TransferFunctionInventory::Linear => Transfer::Linear,
            TransferFunctionInventory::Srgb => Transfer::Srgb,
            TransferFunctionInventory::Bt709 => Transfer::Bt709,
            TransferFunctionInventory::Pq => Transfer::Pq,
            TransferFunctionInventory::Hlg => Transfer::Hlg,
            TransferFunctionInventory::Dci => Transfer::Dci,
            TransferFunctionInventory::Gamma {
                scaled_gamma,
                inverted: true,
            } => Transfer::Gamma(
                GammaExponent::new(scaled_gamma as f32 / 10_000_000.0)
                    .expect("checked positive wire gamma"),
            ),
            _ => unreachable!("source color only constructs supported wire transfers"),
        };
        RgbColorEncoding { space, transfer }
    }

    pub(crate) fn write(
        self,
        output: &mut BitWriter,
        channels: ColorChannels,
        intent: IccRenderingIntent,
    ) -> Result<(), EncodeError> {
        let gray = channels == ColorChannels::Gray;
        let compact = !gray && self == Self::default() && intent == IccRenderingIntent::Relative;
        output.write_bits(u64::from(compact), 1)?;
        if compact {
            return Ok(());
        }
        output.write_bits(0, 1)?; // enumerated, no embedded ICC
        write_enum(output, u32::from(gray))?;
        match self.white {
            WhitePointInventory::D65 => write_enum(output, 1)?,
            WhitePointInventory::E => write_enum(output, 10)?,
            WhitePointInventory::Dci => write_enum(output, 11)?,
            WhitePointInventory::Custom(value) => {
                write_enum(output, 2)?;
                write_xy(output, value)?;
            }
        }
        if !gray {
            match self.primaries {
                PrimariesInventory::Srgb => write_enum(output, 1)?,
                PrimariesInventory::Bt2100 => write_enum(output, 9)?,
                PrimariesInventory::P3 => write_enum(output, 11)?,
                PrimariesInventory::Custom { red, green, blue } => {
                    write_enum(output, 2)?;
                    for value in [red, green, blue] {
                        write_xy(output, value)?;
                    }
                }
            }
        }
        output.write_bits(
            u64::from(matches!(
                self.transfer,
                TransferFunctionInventory::Gamma { .. }
            )),
            1,
        )?;
        match self.transfer {
            TransferFunctionInventory::Gamma { scaled_gamma, .. } => {
                output.write_bits(u64::from(scaled_gamma), 24)?;
            }
            transfer => write_enum(
                output,
                match transfer {
                    TransferFunctionInventory::Bt709 => 1,
                    TransferFunctionInventory::Unknown => 2,
                    TransferFunctionInventory::Linear => 8,
                    TransferFunctionInventory::Srgb => 13,
                    TransferFunctionInventory::Pq => 16,
                    TransferFunctionInventory::Dci => 17,
                    TransferFunctionInventory::Hlg => 18,
                    TransferFunctionInventory::Gamma { .. } => unreachable!(),
                },
            )?,
        }
        write_enum(output, intent as u32)
    }
}

fn validate_geometry(coordinates: RgbChromaticities) -> Result<(), EncodeError> {
    ColorMatrix::between_rgb(
        RgbColorSpace::Custom(coordinates),
        RgbColorSpace::Bt709,
        WhitePointAdaptation::Bradford,
    )
    .map(|_| ())
    .map_err(|_| UnsupportedFeature::InputFormat.into())
}

fn quantize_xy(value: Chromaticity) -> Result<ChromaticityInventory, EncodeError> {
    let component = |value: f64| -> Result<i32, EncodeError> {
        let scaled = (value * 1_000_000.0).round();
        if !(-2_097_152.0..=2_097_151.0).contains(&scaled) {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        Ok(scaled as i32)
    };
    Ok(ChromaticityInventory {
        x: component(value.x())?,
        y: component(value.y())?,
    })
}

fn expand_xy(value: ChromaticityInventory) -> Chromaticity {
    Chromaticity::new(
        f64::from(value.x) / 1_000_000.0,
        f64::from(value.y) / 1_000_000.0,
    )
    .expect("bounded integer chromaticities are finite")
}

fn write_enum(output: &mut BitWriter, value: u32) -> Result<(), EncodeError> {
    let (selector, offset, bits) = match value {
        0 => (0, 0, 0),
        1 => (1, 1, 0),
        2..=17 => (2, 2, 4),
        _ => (3, 18, 6),
    };
    output.write_bits(selector, 2)?;
    output.write_bits(u64::from(value - offset), bits)?;
    Ok(())
}

fn write_xy(output: &mut BitWriter, value: ChromaticityInventory) -> Result<(), EncodeError> {
    for coordinate in [value.x, value.y] {
        let packed = crate::frame_header::pack_signed(coordinate);
        let (selector, offset, bits) = match packed {
            0..524_288 => (0, 0, 19),
            524_288..1_048_576 => (1, 524_288, 19),
            1_048_576..2_097_152 => (2, 1_048_576, 20),
            _ => (3, 2_097_152, 21),
        };
        output.write_bits(selector, 2)?;
        output.write_bits(u64::from(packed - offset), bits)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
