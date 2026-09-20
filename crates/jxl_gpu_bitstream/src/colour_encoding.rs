//! Shared image/gain-map ColorEncoding grammar with implicit XYB fields.
//! Primitive enums and coordinate bundles remain owned by jxl-image.

use jxl_bitstream::{Bitstream, BitstreamResult, Error};
use jxl_image::color::{
    ColourEncoding, ColourSpace, EnumColourEncoding, Primaries, TransferFunction, WhitePoint,
};
use jxl_oxide_common::Bundle;

pub(crate) fn parse(reader: &mut Bitstream<'_>) -> BitstreamResult<ColourEncoding> {
    if reader.read_bool()? {
        return Ok(ColourEncoding::default());
    }
    let want_icc = reader.read_bool()?;
    let colour_space = reader.read_enum::<ColourSpace>()?;
    if want_icc {
        return Ok(ColourEncoding::IccProfile(colour_space));
    }
    let white_point = if colour_space == ColourSpace::Xyb {
        WhitePoint::D65
    } else {
        WhitePoint::parse(reader, ())?
    };
    let primaries = if matches!(colour_space, ColourSpace::Xyb | ColourSpace::Grey) {
        Primaries::Srgb
    } else {
        Primaries::parse(reader, ())?
    };
    let tf = if colour_space == ColourSpace::Xyb {
        // No transfer-function bits occur in an XYB ColorEncoding. The implicit
        // 1/3 exponent uses the same 1e7 quantization as the other gamma values.
        TransferFunction::Gamma {
            g: 3_333_333,
            inverted: true,
        }
    } else {
        let transfer = TransferFunction::parse(reader, ())?;
        if let TransferFunction::Gamma { g, .. } = transfer
            && (g > 10_000_000 || u64::from(g) * 8192 < 10_000_000)
        {
            return Err(Error::ValidationFailed("Invalid color gamma"));
        }
        transfer
    };
    Ok(ColourEncoding::Enum(EnumColourEncoding {
        colour_space,
        white_point,
        primaries,
        tf,
        rendering_intent: reader.read_enum()?,
    }))
}

#[cfg(test)]
mod tests;
