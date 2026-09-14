//! Integer LUT methods, lowered to explicit unit-domain stages and physical PCS XYZ.
use super::profile::{Reader, invalid, limit};
use super::{
    IccAffine, IccClutInterpolation, IccDirection, IccError, IccLimits, IccProfile, IccProgram,
    IccSignature, IccStage,
};

mod data;
mod multi;
mod tables;

pub(super) fn parse(
    profile: &IccProfile,
    tag: IccSignature,
    direction: IccDirection,
    inputs: usize,
    outputs: usize,
) -> Result<IccProgram, IccError> {
    let data = profile.required(tag)?;
    let kind = data.signature(0)?;
    if !matches!(&kind.0, b"mft1" | b"mft2" | b"mAB " | b"mBA ") {
        return Err(IccError::TagType { tag, kind });
    }
    let limits = profile.limits();
    let pair = data.slice(8, 10, "LUT channels")?;
    if (usize::from(pair[0]), usize::from(pair[1])) != (inputs, outputs) {
        return invalid("LUT profile channels", 8);
    }
    for count in [inputs, outputs] {
        limit(
            "processing channels",
            count as u64,
            u64::from(limits.max_processing_channels),
        )?;
    }
    let reverse = direction == IccDirection::PcsToDevice;
    let lab = profile.header().pcs == IccSignature(*b"Lab ");
    // ICC permits choosing the interpolation method. Match the existing CMM policy:
    // legacy Lab-indexed output LUTs use multilinear interpolation; other LUTs use
    // tetrahedra in the last three dimensions, linear interpolation in earlier ones.
    let interpolation = if reverse && lab {
        IccClutInterpolation::Multilinear
    } else {
        IccClutInterpolation::Tetrahedral
    };
    let mut stages = match &kind.0 {
        b"mft1" | b"mft2" => tables::parse(
            data,
            inputs,
            outputs,
            limits,
            reverse && !lab,
            interpolation,
        )?,
        b"mAB " | b"mBA " => {
            if profile.header().version < 0x0400_0000 {
                return invalid("LUT A/B requires v4", 8);
            }
            if (kind == IccSignature(*b"mBA ")) != reverse {
                return Err(IccError::TagType { tag, kind });
            }
            multi::parse(data, tag, inputs, outputs, limits, reverse, interpolation)?
        }
        _ => return Err(IccError::TagType { tag, kind }),
    };
    let connection = pcs(lab, kind == IccSignature(*b"mft2"), reverse)?;
    if reverse {
        stages.splice(..0, connection);
    } else {
        stages.extend(connection);
    }
    IccProgram::new(inputs, outputs, stages)
}

fn diagonal(scale: [f64; 3], offset: [f64; 3], clamp: bool) -> Result<IccStage, IccError> {
    Ok(IccStage::Matrix(IccAffine::new(
        3,
        vec![scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, scale[2]],
        offset.to_vec(),
        clamp,
    )?))
}

fn pcs(lab: bool, legacy_lab: bool, reverse: bool) -> Result<Vec<IccStage>, IccError> {
    if !lab {
        // lut8 PCSXYZ is implementation-defined. Use the same normalized XYZ
        // convention as lut16/lutA/B and Little CMS: one is encoded by 0x8000.
        let scale = if reverse {
            32768.0 / 65535.0
        } else {
            65535.0 / 32768.0
        };
        return Ok(vec![diagonal([scale; 3], [0.0; 3], reverse)?]);
    }
    if !legacy_lab {
        return if reverse {
            Ok(vec![
                IccStage::XyzToLab,
                diagonal(
                    [0.01, 1.0 / 255.0, 1.0 / 255.0],
                    [0.0, 128.0 / 255.0, 128.0 / 255.0],
                    true,
                )?,
            ])
        } else {
            Ok(vec![
                diagonal([100.0, 255.0, 255.0], [0.0, -128.0, -128.0], false)?,
                IccStage::LabToXyz,
            ])
        };
    }
    // lut16 uses the legacy Lab encoding even in a v4 profile. L*=100 is 0xff00;
    // a*/b*=0 is 0x8000. Preserve the valid legacy a*/b* range above +127 while
    // enforcing L* in [0,100]. These boundaries cannot be fused away.
    if reverse {
        Ok(vec![
            IccStage::XyzToLab,
            diagonal(
                [0.01, 256.0 / 65535.0, 256.0 / 65535.0],
                [0.0, 32768.0 / 65535.0, 32768.0 / 65535.0],
                true,
            )?,
            diagonal([65280.0 / 65535.0, 1.0, 1.0], [0.0; 3], false)?,
        ])
    } else {
        Ok(vec![
            diagonal([65535.0 / 65280.0, 1.0, 1.0], [0.0; 3], true)?,
            diagonal(
                [100.0, 65535.0 / 256.0, 65535.0 / 256.0],
                [0.0, -128.0, -128.0],
                false,
            )?,
            IccStage::LabToXyz,
        ])
    }
}

fn processing_count(count: usize, limits: IccLimits) -> Result<(), IccError> {
    limit(
        "processing elements",
        count as u64,
        u64::from(limits.max_processing_elements),
    )
}
