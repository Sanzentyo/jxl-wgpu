use crate::capture::{
    CaptureFile, DataType, OperationKind, OperationSpec, SectionKind, TensorShape, encode_f32,
    encode_i32,
};
use crate::config::SyntheticCaseConfig;
use crate::error::{Error, Result};
use crate::reference::{
    epf_uses_variable_sigma, execute_affine_i32, execute_epf_with_sigma, execute_operation,
};

const INPUT_SECTION_ID: u32 = 1;
const EXPECTED_SECTION_ID: u32 = 2;

pub fn generate_case(config: &SyntheticCaseConfig) -> Result<CaptureFile> {
    config.validate()?;
    let operation = operation_spec(config)?;
    let input_shape = TensorShape::planar(config.width, config.height, config.channels)?;
    let output_shape = output_shape(config)?;
    let input_len =
        usize::try_from(input_shape.minimum_elements()?).map_err(|_| Error::LengthOverflow)?;
    let mut random = SplitMix64::new(config.seed);
    if config.operation == OperationKind::Affine {
        let input = (0..input_len)
            .map(|_| random.next_i32_symmetric(2048))
            .collect::<Vec<_>>();
        let expected = execute_affine_i32(&operation, &input, &input_shape, &output_shape)?;
        let mut capture = base_capture(config, operation);
        capture.add_section(
            INPUT_SECTION_ID,
            "input",
            SectionKind::Input,
            DataType::I32,
            Some(input_shape),
            encode_i32(&input),
        )?;
        capture.add_section(
            EXPECTED_SECTION_ID,
            "expected",
            SectionKind::Expected,
            DataType::F32,
            Some(output_shape),
            encode_f32(&expected),
        )?;
        return Ok(capture);
    }
    let mut input = (0..input_len)
        .map(|_| random.next_f32_signed())
        .collect::<Vec<_>>();
    if config.operation == OperationKind::PremultiplyAlpha {
        let plane_len =
            usize::try_from(input_shape.channel_stride).map_err(|_| Error::LengthOverflow)?;
        let alpha_channel = config.channels - 1;
        let start = usize::from(alpha_channel)
            .checked_mul(plane_len)
            .ok_or(Error::LengthOverflow)?;
        input[start..start + plane_len]
            .iter_mut()
            .for_each(|value| *value = value.abs().min(1.0));
    }
    if config.operation == OperationKind::Epf {
        let sigma = if epf_uses_variable_sigma(&operation)? {
            let shape =
                TensorShape::planar(config.width.div_ceil(8), config.height.div_ceil(8), 1)?;
            let values = variable_epf_sigma(&shape)?;
            Some((values, shape))
        } else {
            None
        };
        let expected = execute_epf_with_sigma(
            &operation,
            &input,
            &input_shape,
            &output_shape,
            sigma
                .as_ref()
                .map(|(values, shape)| (values.as_slice(), shape)),
        )?;
        let mut capture = base_capture(config, operation);
        capture.add_section(
            INPUT_SECTION_ID,
            "input",
            SectionKind::Input,
            DataType::F32,
            Some(input_shape),
            encode_f32(&input),
        )?;
        capture.add_section(
            EXPECTED_SECTION_ID,
            "expected",
            SectionKind::Expected,
            DataType::F32,
            Some(output_shape),
            encode_f32(&expected),
        )?;
        if let Some((values, shape)) = sigma {
            capture.add_section(
                3,
                "sigma",
                SectionKind::Parameter,
                DataType::F32,
                Some(shape),
                encode_f32(&values),
            )?;
        }
        return Ok(capture);
    }
    let expected = execute_operation(&operation, &input, &input_shape, &output_shape)?;
    let mut capture = base_capture(config, operation);
    capture.add_section(
        INPUT_SECTION_ID,
        "input",
        SectionKind::Input,
        DataType::F32,
        Some(input_shape),
        encode_f32(&input),
    )?;
    capture.add_section(
        EXPECTED_SECTION_ID,
        "expected",
        SectionKind::Expected,
        DataType::F32,
        Some(output_shape),
        encode_f32(&expected),
    )?;
    Ok(capture)
}

fn base_capture(config: &SyntheticCaseConfig, operation: OperationSpec) -> CaptureFile {
    let mut capture = CaptureFile::new(
        config.name.clone(),
        operation,
        config.seed,
        config.precision,
    );
    capture
        .metadata
        .tags
        .insert("generator".into(), "splitmix64-v1".into());
    capture
}

fn operation_spec(config: &SyntheticCaseConfig) -> Result<OperationSpec> {
    let mut parameters = config.parameters.clone();
    match config.operation {
        OperationKind::Affine => {
            parameters.entry("scale".into()).or_insert(0.75);
            parameters.entry("bias".into()).or_insert(0.125);
        }
        OperationKind::Gaborish => {
            parameters.entry("weight1".into()).or_insert(0.115_169_525);
            parameters.entry("weight2".into()).or_insert(0.061_248_592);
        }
        OperationKind::Epf => {
            let pass = parameters.get("pass").copied().unwrap_or(1.0);
            if !pass.is_finite() || pass.fract() != 0.0 || !(0.0..=2.0).contains(&pass) {
                return Err(Error::InvalidConfig(format!(
                    "case {} has invalid EPF pass {pass}",
                    config.name
                )));
            }
            parameters.entry("pass".into()).or_insert(pass);
            parameters.entry("sigma".into()).or_insert(-0.58);
            parameters.entry("variable_sigma".into()).or_insert(0.0);
            parameters
                .entry("sigma_scale".into())
                .or_insert(match pass as u8 {
                    0 => 0.9,
                    1 => 1.0,
                    2 => 6.5,
                    _ => unreachable!(),
                });
            parameters
                .entry("border_sad_mul".into())
                .or_insert(2.3 / 3.0);
            parameters.entry("channel_scale_x".into()).or_insert(40.0);
            parameters.entry("channel_scale_y".into()).or_insert(5.0);
            parameters.entry("channel_scale_b".into()).or_insert(3.5);
            if !matches!(parameters["variable_sigma"], 0.0 | 1.0) {
                return Err(Error::InvalidConfig(format!(
                    "case {} has invalid EPF variable_sigma {}; use 0 or 1",
                    config.name, parameters["variable_sigma"]
                )));
            }
        }
        OperationKind::Upsample => {
            parameters.entry("factor".into()).or_insert(2.0);
        }
        OperationKind::ChromaUpsample => {
            parameters.entry("axis".into()).or_insert(0.0);
        }
        OperationKind::PremultiplyAlpha => {
            parameters
                .entry("alpha_channel".into())
                .or_insert(f64::from(config.channels - 1));
        }
        OperationKind::Copy | OperationKind::YcbcrToRgb => {}
    }
    if parameters.values().any(|value| !value.is_finite()) {
        return Err(Error::InvalidConfig(format!(
            "case {} contains a non-finite operation parameter",
            config.name
        )));
    }
    Ok(OperationSpec {
        kind: config.operation.clone(),
        parameters,
    })
}

fn output_shape(config: &SyntheticCaseConfig) -> Result<TensorShape> {
    match config.operation {
        OperationKind::Upsample => {
            let factor = config.parameters.get("factor").copied().unwrap_or(2.0);
            if !factor.is_finite() || factor.fract() != 0.0 || !matches!(factor as u32, 2 | 4 | 8) {
                return Err(Error::InvalidConfig(format!(
                    "case {} has invalid upsample factor {factor}",
                    config.name
                )));
            }
            let factor = factor as u32;
            TensorShape::planar(
                config
                    .width
                    .checked_mul(factor)
                    .ok_or(Error::LengthOverflow)?,
                config
                    .height
                    .checked_mul(factor)
                    .ok_or(Error::LengthOverflow)?,
                config.channels,
            )
        }
        OperationKind::ChromaUpsample => {
            let axis = config.parameters.get("axis").copied().unwrap_or(0.0);
            if !matches!(axis, 0.0 | 1.0) {
                return Err(Error::InvalidConfig(format!(
                    "case {} has invalid chroma axis {axis}; use 0 for horizontal or 1 for vertical",
                    config.name
                )));
            }
            let full_width = config.width.checked_mul(2).ok_or(Error::LengthOverflow)?;
            let full_height = config.height.checked_mul(2).ok_or(Error::LengthOverflow)?;
            let (width, height) = if axis == 0.0 {
                (
                    configured_chroma_extent(config, "output_width", full_width)?,
                    config.height,
                )
            } else {
                (
                    config.width,
                    configured_chroma_extent(config, "output_height", full_height)?,
                )
            };
            TensorShape::planar(width, height, config.channels)
        }
        _ => TensorShape::planar(config.width, config.height, config.channels),
    }
}

fn configured_chroma_extent(
    config: &SyntheticCaseConfig,
    parameter: &str,
    full_extent: u32,
) -> Result<u32> {
    let value = config
        .parameters
        .get(parameter)
        .copied()
        .unwrap_or(f64::from(full_extent));
    if !value.is_finite()
        || value.fract() != 0.0
        || value < 1.0
        || value > f64::from(full_extent)
        || (value as u32).div_ceil(2) != full_extent.div_ceil(2)
    {
        return Err(Error::InvalidConfig(format!(
            "case {} has invalid {parameter} {value}; expected {} or {full_extent}",
            config.name,
            full_extent.saturating_sub(1)
        )));
    }
    Ok(value as u32)
}

fn variable_epf_sigma(shape: &TensorShape) -> Result<Vec<f32>> {
    let length = usize::try_from(shape.minimum_elements()?).map_err(|_| Error::LengthOverflow)?;
    const VALUES: [f32; 5] = [-0.35, -4.25, -1.15, -0.72, -3.905_243];
    Ok((0..length)
        .map(|index| VALUES[index % VALUES.len()])
        .collect())
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_f32_signed(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 41) as u32;
        let unit = mantissa as f32 / ((1_u32 << 23) - 1) as f32;
        unit.mul_add(2.0, -1.0)
    }

    fn next_i32_symmetric(&mut self, magnitude: i32) -> i32 {
        let span = u64::try_from(i64::from(magnitude) * 2 + 1).unwrap_or(u64::MAX);
        i32::try_from(self.next_u64() % span).unwrap_or(i32::MAX) - magnitude
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::capture::PrecisionMode;

    use super::*;

    fn config(seed: u64) -> SyntheticCaseConfig {
        SyntheticCaseConfig {
            name: "deterministic".into(),
            operation: OperationKind::Gaborish,
            width: 17,
            height: 9,
            channels: 3,
            seed,
            precision: PrecisionMode::F32,
            parameters: BTreeMap::new(),
        }
    }

    #[test]
    fn same_seed_produces_identical_capture() {
        assert_eq!(
            generate_case(&config(42)).unwrap().to_bytes().unwrap(),
            generate_case(&config(42)).unwrap().to_bytes().unwrap()
        );
    }

    #[test]
    fn different_seed_changes_payload() {
        assert_ne!(
            generate_case(&config(42)).unwrap().to_bytes().unwrap(),
            generate_case(&config(43)).unwrap().to_bytes().unwrap()
        );
    }

    #[test]
    fn variable_epf_uses_existing_named_parameter_tensor_schema() {
        let capture = generate_case(&SyntheticCaseConfig {
            name: "epf-variable".into(),
            operation: OperationKind::Epf,
            width: 19,
            height: 11,
            channels: 3,
            seed: 99,
            precision: PrecisionMode::F32,
            parameters: BTreeMap::from([("pass".into(), 2.0), ("variable_sigma".into(), 1.0)]),
        })
        .unwrap();
        assert_eq!(
            capture.metadata.schema_version,
            crate::CAPTURE_SCHEMA_VERSION
        );
        let (descriptor, values) = capture
            .section_by_name(SectionKind::Parameter, "sigma")
            .unwrap()
            .expect("variable EPF capture has a sigma tensor");
        assert_eq!(descriptor.data_type, DataType::F32);
        assert_eq!(
            descriptor.tensor,
            Some(TensorShape::planar(3, 2, 1).unwrap())
        );
        assert_eq!(values.len(), 3 * 2 * std::mem::size_of::<f32>());

        let encoded = capture.to_bytes().unwrap();
        let decoded = CaptureFile::from_bytes(&encoded).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), encoded);
        assert_eq!(
            crate::reference::execute_capture(&decoded).unwrap(),
            crate::capture::decode_f32(decoded.section_by_kind(SectionKind::Expected).unwrap().1,)
                .unwrap()
        );
    }
}
