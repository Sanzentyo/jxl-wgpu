use crate::capture::{
    CaptureFile, DataType, OperationKind, OperationSpec, SectionKind, TensorShape,
};
use crate::error::{Error, Result};

pub fn execute_capture(capture: &CaptureFile) -> Result<Vec<f32>> {
    let (input_descriptor, input_bytes) = capture.section_by_kind(SectionKind::Input)?;
    let input_shape = input_descriptor
        .tensor
        .as_ref()
        .ok_or_else(|| Error::InvalidTensor("input section has no tensor shape".into()))?;
    let expected_descriptor = capture
        .metadata
        .sections
        .iter()
        .find(|descriptor| descriptor.kind == SectionKind::Expected)
        .ok_or_else(|| Error::InvalidMetadata("capture has no expected section".into()))?;
    let output_shape = expected_descriptor
        .tensor
        .as_ref()
        .ok_or_else(|| Error::InvalidTensor("expected section has no tensor shape".into()))?;
    match (&capture.metadata.operation.kind, input_descriptor.data_type) {
        (OperationKind::Affine, DataType::I32) => execute_affine_i32(
            &capture.metadata.operation,
            &crate::capture::decode_i32(input_bytes)?,
            input_shape,
            output_shape,
        ),
        (OperationKind::Epf, DataType::F32) => {
            let input = crate::capture::decode_f32(input_bytes)?;
            let sigma = epf_sigma_from_capture(capture)?;
            execute_epf_with_sigma(
                &capture.metadata.operation,
                &input,
                input_shape,
                output_shape,
                sigma
                    .as_ref()
                    .map(|(values, shape)| (values.as_slice(), shape)),
            )
        }
        (_, DataType::F32) => execute_operation(
            &capture.metadata.operation,
            &crate::capture::decode_f32(input_bytes)?,
            input_shape,
            output_shape,
        ),
        _ => Err(Error::UnsupportedOperation {
            backend: "reference",
            operation: format!("{:?} input", input_descriptor.data_type),
        }),
    }
}

pub fn execute_affine_i32(
    operation: &OperationSpec,
    input: &[i32],
    input_shape: &TensorShape,
    output_shape: &TensorShape,
) -> Result<Vec<f32>> {
    require_same_shape(input_shape, output_shape)?;
    let expected =
        usize::try_from(input_shape.minimum_elements()?).map_err(|_| Error::LengthOverflow)?;
    if input.len() != expected {
        return Err(Error::InvalidTensor(format!(
            "input has {} elements but shape requires {expected}",
            input.len()
        )));
    }
    let scale = operation.parameter("scale")? as f32;
    let bias = operation.parameter("bias")? as f32;
    Ok(input
        .iter()
        .map(|&value| (value as f32).mul_add(scale, bias))
        .collect())
}

pub fn execute_operation(
    operation: &OperationSpec,
    input: &[f32],
    input_shape: &TensorShape,
    output_shape: &TensorShape,
) -> Result<Vec<f32>> {
    validate_dense_planar(input, input_shape)?;
    match operation.kind {
        OperationKind::Copy => copy(input, input_shape, output_shape),
        OperationKind::Affine => affine(operation, input, input_shape, output_shape),
        OperationKind::Gaborish => gaborish(operation, input, input_shape, output_shape),
        OperationKind::Epf => {
            execute_epf_with_sigma(operation, input, input_shape, output_shape, None)
        }
        OperationKind::Upsample => upsample(operation, input, input_shape, output_shape),
        OperationKind::ChromaUpsample => {
            chroma_upsample(operation, input, input_shape, output_shape)
        }
        OperationKind::YcbcrToRgb => ycbcr_to_rgb(input, input_shape, output_shape),
        OperationKind::PremultiplyAlpha => {
            premultiply_alpha(operation, input, input_shape, output_shape)
        }
    }
}

fn copy(input: &[f32], input_shape: &TensorShape, output_shape: &TensorShape) -> Result<Vec<f32>> {
    require_same_shape(input_shape, output_shape)?;
    Ok(input.to_vec())
}

fn affine(
    operation: &OperationSpec,
    input: &[f32],
    input_shape: &TensorShape,
    output_shape: &TensorShape,
) -> Result<Vec<f32>> {
    require_same_shape(input_shape, output_shape)?;
    let scale = operation.parameter("scale")? as f32;
    let bias = operation.parameter("bias")? as f32;
    Ok(input
        .iter()
        .map(|&value| value.mul_add(scale, bias))
        .collect())
}

fn gaborish(
    operation: &OperationSpec,
    input: &[f32],
    input_shape: &TensorShape,
    output_shape: &TensorShape,
) -> Result<Vec<f32>> {
    require_same_shape(input_shape, output_shape)?;
    let weight1 = operation.parameter("weight1")? as f32;
    let weight2 = operation.parameter("weight2")? as f32;
    let weight_total = 1.0 + 4.0 * weight1 + 4.0 * weight2;
    if !weight_total.is_finite() || weight_total == 0.0 {
        return Err(Error::InvalidMetadata(format!(
            "invalid Gaborish weight total {weight_total}"
        )));
    }
    let weight0 = 1.0 / weight_total;
    let weight1 = weight1 / weight_total;
    let weight2 = weight2 / weight_total;
    let width = usize::try_from(input_shape.width).map_err(|_| Error::LengthOverflow)?;
    let height = usize::try_from(input_shape.height).map_err(|_| Error::LengthOverflow)?;
    let channels = usize::from(input_shape.channels);
    let mut output = vec![0.0_f32; input.len()];
    for channel in 0..channels {
        for y in 0..height {
            let top = y.saturating_sub(1);
            let bottom = (y + 1).min(height - 1);
            for x in 0..width {
                let left = x.saturating_sub(1);
                let right = (x + 1).min(width - 1);
                let center = sample(input, input_shape, channel, x, y)?;
                let cardinal = sample(input, input_shape, channel, x, top)?
                    + sample(input, input_shape, channel, left, y)?
                    + sample(input, input_shape, channel, x, bottom)?
                    + sample(input, input_shape, channel, right, y)?;
                let diagonal = sample(input, input_shape, channel, left, top)?
                    + sample(input, input_shape, channel, right, top)?
                    + sample(input, input_shape, channel, left, bottom)?
                    + sample(input, input_shape, channel, right, bottom)?;
                let value = weight1.mul_add(cardinal, weight2.mul_add(diagonal, center * weight0));
                let index = dense_planar_index(input_shape, channel, x, y)?;
                output[index] = value;
            }
        }
    }
    Ok(output)
}

const EPF_MIN_SIGMA: f32 = -3.905_243;
const EPF_DEFAULT_BORDER_SAD_MUL: f32 = 2.3 / 3.0;
const EPF_DEFAULT_CHANNEL_SCALE: [f32; 3] = [40.0, 5.0, 3.5];

#[derive(Clone, Copy, Debug)]
struct EpfReferenceParams {
    pass: u8,
    sigma_scale: f32,
    border_sad_mul: f32,
    channel_scale: [f32; 3],
    constant_sigma: f32,
    variable_sigma: bool,
}

fn epf_reference_params(operation: &OperationSpec) -> Result<EpfReferenceParams> {
    let pass = operation.parameters.get("pass").copied().unwrap_or(1.0);
    if !pass.is_finite() || pass.fract() != 0.0 || !(0.0..=2.0).contains(&pass) {
        return Err(Error::InvalidMetadata(format!(
            "EPF pass must be 0, 1, or 2; found {pass}"
        )));
    }
    let pass = pass as u8;
    let default_sigma_scale = match pass {
        0 => 0.9,
        1 => 1.0,
        2 => 6.5,
        _ => unreachable!(),
    };
    let variable_sigma = epf_uses_variable_sigma(operation)?;
    let params = EpfReferenceParams {
        pass,
        sigma_scale: operation
            .parameters
            .get("sigma_scale")
            .copied()
            .unwrap_or(default_sigma_scale) as f32,
        border_sad_mul: operation
            .parameters
            .get("border_sad_mul")
            .copied()
            .unwrap_or(f64::from(EPF_DEFAULT_BORDER_SAD_MUL)) as f32,
        channel_scale: [
            operation
                .parameters
                .get("channel_scale_x")
                .copied()
                .unwrap_or(f64::from(EPF_DEFAULT_CHANNEL_SCALE[0])) as f32,
            operation
                .parameters
                .get("channel_scale_y")
                .copied()
                .unwrap_or(f64::from(EPF_DEFAULT_CHANNEL_SCALE[1])) as f32,
            operation
                .parameters
                .get("channel_scale_b")
                .copied()
                .unwrap_or(f64::from(EPF_DEFAULT_CHANNEL_SCALE[2])) as f32,
        ],
        constant_sigma: operation.parameters.get("sigma").copied().unwrap_or(-0.58) as f32,
        variable_sigma,
    };
    if !params.sigma_scale.is_finite()
        || !params.border_sad_mul.is_finite()
        || !params.constant_sigma.is_finite()
        || params.channel_scale.iter().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidMetadata(
            "EPF parameters must be finite f32 values".into(),
        ));
    }
    Ok(params)
}

pub(crate) fn epf_uses_variable_sigma(operation: &OperationSpec) -> Result<bool> {
    match operation
        .parameters
        .get("variable_sigma")
        .copied()
        .unwrap_or(0.0)
    {
        0.0 => Ok(false),
        1.0 => Ok(true),
        value => Err(Error::InvalidMetadata(format!(
            "EPF variable_sigma must be 0 or 1; found {value}"
        ))),
    }
}

fn epf_sigma_from_capture(capture: &CaptureFile) -> Result<Option<(Vec<f32>, TensorShape)>> {
    let sigma = capture.section_by_name(SectionKind::Parameter, "sigma")?;
    if !epf_uses_variable_sigma(&capture.metadata.operation)? {
        if sigma.is_some() {
            return Err(Error::InvalidMetadata(
                "constant-sigma EPF capture must not contain a sigma parameter plane".into(),
            ));
        }
        return Ok(None);
    }
    let (descriptor, bytes) = sigma.ok_or_else(|| {
        Error::InvalidMetadata(
            "variable-sigma EPF capture is missing the named sigma parameter plane".into(),
        )
    })?;
    if descriptor.data_type != DataType::F32 {
        return Err(Error::InvalidTensor(
            "EPF sigma parameter plane must contain F32 values".into(),
        ));
    }
    let shape = descriptor
        .tensor
        .as_ref()
        .ok_or_else(|| Error::InvalidTensor("EPF sigma section has no tensor shape".into()))?;
    if shape != &TensorShape::planar(shape.width, shape.height, 1)? {
        return Err(Error::InvalidTensor(
            "EPF sigma parameter must be a dense one-channel planar tensor".into(),
        ));
    }
    Ok(Some((crate::capture::decode_f32(bytes)?, shape.clone())))
}

pub(crate) fn execute_epf_with_sigma(
    operation: &OperationSpec,
    input: &[f32],
    input_shape: &TensorShape,
    output_shape: &TensorShape,
    sigma_plane: Option<(&[f32], &TensorShape)>,
) -> Result<Vec<f32>> {
    require_same_shape(input_shape, output_shape)?;
    if input_shape.channels != 3 {
        return Err(Error::InvalidTensor(
            "EPF requires exactly three planar channels".into(),
        ));
    }
    let params = epf_reference_params(operation)?;
    let sigma_plane = match (params.variable_sigma, sigma_plane) {
        (false, None) => None,
        (true, Some((values, shape))) => {
            validate_dense_planar(values, shape)?;
            if shape.channels != 1
                || shape.width < input_shape.width.div_ceil(8)
                || shape.height < input_shape.height.div_ceil(8)
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(Error::InvalidTensor(format!(
                    "EPF sigma plane must be finite, one-channel, and cover at least {}x{} blocks",
                    input_shape.width.div_ceil(8),
                    input_shape.height.div_ceil(8)
                )));
            }
            Some((values, shape))
        }
        (false, Some(_)) => {
            return Err(Error::InvalidMetadata(
                "constant-sigma EPF received an unexpected sigma plane".into(),
            ));
        }
        (true, None) => {
            return Err(Error::InvalidMetadata(
                "variable-sigma EPF requires a sigma parameter plane".into(),
            ));
        }
    };

    const PLUS: [(i64, i64); 5] = [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)];
    const PASS0_OFFSETS: [(i64, i64); 12] = [
        (0, -2),
        (-1, -1),
        (0, -1),
        (1, -1),
        (-2, 0),
        (-1, 0),
        (1, 0),
        (2, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (0, 2),
    ];
    const PASS12_OFFSETS: [(i64, i64); 4] = [(0, -1), (-1, 0), (1, 0), (0, 1)];
    let offsets: &[(i64, i64)] = if params.pass == 0 {
        &PASS0_OFFSETS
    } else {
        &PASS12_OFFSETS
    };
    let mut output = vec![0.0; input.len()];
    for y in 0..input_shape.height {
        for x in 0..input_shape.width {
            let sigma = if let Some((values, shape)) = sigma_plane {
                sample(
                    values,
                    shape,
                    0,
                    usize::try_from(x / 8).map_err(|_| Error::LengthOverflow)?,
                    usize::try_from(y / 8).map_err(|_| Error::LengthOverflow)?,
                )?
            } else {
                params.constant_sigma
            };
            if EPF_MIN_SIGMA > sigma {
                for channel in 0..3 {
                    let index = dense_planar_index(
                        output_shape,
                        channel,
                        usize::try_from(x).map_err(|_| Error::LengthOverflow)?,
                        usize::try_from(y).map_err(|_| Error::LengthOverflow)?,
                    )?;
                    output[index] =
                        epf_sample(input, input_shape, channel, i64::from(x), i64::from(y))?;
                }
                continue;
            }

            let scaled_sigma = params.sigma_scale * 1.65;
            let block_border = matches!(x % 8, 0 | 7) || matches!(y % 8, 0 | 7);
            let sad_mul = if block_border {
                scaled_sigma * params.border_sad_mul
            } else {
                scaled_sigma
            };
            let inverse_sigma = sigma * sad_mul;
            let mut weights = Vec::with_capacity(offsets.len());
            for &(offset_x, offset_y) in offsets {
                let sad = if params.pass == 2 {
                    let center_x = epf_sample(input, input_shape, 0, i64::from(x), i64::from(y))?;
                    let candidate_x = epf_sample(
                        input,
                        input_shape,
                        0,
                        i64::from(x) + offset_x,
                        i64::from(y) + offset_y,
                    )?;
                    let center_y = epf_sample(input, input_shape, 1, i64::from(x), i64::from(y))?;
                    let candidate_y = epf_sample(
                        input,
                        input_shape,
                        1,
                        i64::from(x) + offset_x,
                        i64::from(y) + offset_y,
                    )?;
                    let center_b = epf_sample(input, input_shape, 2, i64::from(x), i64::from(y))?;
                    let candidate_b = epf_sample(
                        input,
                        input_shape,
                        2,
                        i64::from(x) + offset_x,
                        i64::from(y) + offset_y,
                    )?;
                    (candidate_x - center_x).abs().mul_add(
                        params.channel_scale[0],
                        (candidate_y - center_y).abs().mul_add(
                            params.channel_scale[1],
                            (candidate_b - center_b).abs() * params.channel_scale[2],
                        ),
                    )
                } else {
                    let mut total = 0.0;
                    for channel in 0..3 {
                        let mut channel_sad = 0.0;
                        for &(plus_x, plus_y) in &PLUS {
                            let center = epf_sample(
                                input,
                                input_shape,
                                channel,
                                i64::from(x) + plus_x,
                                i64::from(y) + plus_y,
                            )?;
                            let candidate = epf_sample(
                                input,
                                input_shape,
                                channel,
                                i64::from(x) + offset_x + plus_x,
                                i64::from(y) + offset_y + plus_y,
                            )?;
                            channel_sad += (candidate - center).abs();
                        }
                        total = channel_sad.mul_add(params.channel_scale[channel], total);
                    }
                    total
                };
                weights.push(sad.mul_add(inverse_sigma, 1.0).max(0.0));
            }
            let weight_sum = weights.iter().fold(1.0, |sum, weight| sum + weight);
            for channel in 0..3 {
                let center = epf_sample(input, input_shape, channel, i64::from(x), i64::from(y))?;
                let accumulate = |value: f32, (&(offset_x, offset_y), &weight)| {
                    epf_sample(
                        input,
                        input_shape,
                        channel,
                        i64::from(x) + offset_x,
                        i64::from(y) + offset_y,
                    )
                    .map(|sample| sample.mul_add(weight, value))
                };
                let value = if params.pass == 2 {
                    offsets.iter().zip(&weights).try_fold(center, accumulate)?
                } else {
                    offsets
                        .iter()
                        .zip(&weights)
                        .rev()
                        .try_fold(center, accumulate)?
                };
                let output_index = dense_planar_index(
                    output_shape,
                    channel,
                    usize::try_from(x).map_err(|_| Error::LengthOverflow)?,
                    usize::try_from(y).map_err(|_| Error::LengthOverflow)?,
                )?;
                output[output_index] = value / weight_sum;
            }
        }
    }
    Ok(output)
}

fn epf_sample(input: &[f32], shape: &TensorShape, channel: usize, x: i64, y: i64) -> Result<f32> {
    sample(
        input,
        shape,
        channel,
        mirror_coordinate(x, shape.width)?,
        mirror_coordinate(y, shape.height)?,
    )
}

fn upsample(
    operation: &OperationSpec,
    input: &[f32],
    input_shape: &TensorShape,
    output_shape: &TensorShape,
) -> Result<Vec<f32>> {
    let factor = operation.parameter("factor")?;
    if !factor.is_finite() || factor.fract() != 0.0 || !matches!(factor as u32, 2 | 4 | 8) {
        return Err(Error::InvalidMetadata(format!(
            "upsample factor must be 2, 4, or 8; found {factor}"
        )));
    }
    let factor = factor as u32;
    if output_shape.width != input_shape.width.saturating_mul(factor)
        || output_shape.height != input_shape.height.saturating_mul(factor)
        || output_shape.channels != input_shape.channels
    {
        return Err(Error::InvalidTensor(
            "upsample output shape does not match its factor".into(),
        ));
    }
    validate_dense_planar_shape(output_shape)?;
    let output_len =
        usize::try_from(output_shape.minimum_elements()?).map_err(|_| Error::LengthOverflow)?;
    let mut output = vec![0.0_f32; output_len];
    let width = usize::try_from(output_shape.width).map_err(|_| Error::LengthOverflow)?;
    let height = usize::try_from(output_shape.height).map_err(|_| Error::LengthOverflow)?;
    let factor = usize::try_from(factor).map_err(|_| Error::LengthOverflow)?;
    for channel in 0..usize::from(output_shape.channels) {
        for y in 0..height {
            for x in 0..width {
                let value = sample(input, input_shape, channel, x / factor, y / factor)?;
                let index = dense_planar_index(output_shape, channel, x, y)?;
                output[index] = value;
            }
        }
    }
    Ok(output)
}

fn chroma_upsample(
    operation: &OperationSpec,
    input: &[f32],
    input_shape: &TensorShape,
    output_shape: &TensorShape,
) -> Result<Vec<f32>> {
    let axis = operation.parameter("axis")?;
    if !matches!(axis, 0.0 | 1.0) {
        return Err(Error::InvalidMetadata(format!(
            "chroma axis must be 0 (horizontal) or 1 (vertical); found {axis}"
        )));
    }
    let horizontal = axis == 0.0;
    let valid_extent = if horizontal {
        output_shape.width.div_ceil(2) == input_shape.width
            && output_shape.height == input_shape.height
    } else {
        output_shape.width == input_shape.width
            && output_shape.height.div_ceil(2) == input_shape.height
    };
    if !valid_extent || output_shape.channels != input_shape.channels {
        return Err(Error::InvalidTensor(
            "chroma upsample output shape does not match its axis".into(),
        ));
    }
    validate_dense_planar_shape(output_shape)?;
    let output_len =
        usize::try_from(output_shape.minimum_elements()?).map_err(|_| Error::LengthOverflow)?;
    let mut output = vec![0.0; output_len];
    for channel in 0..usize::from(output_shape.channels) {
        for output_y in 0..output_shape.height {
            for output_x in 0..output_shape.width {
                let (source_x, source_y, neighbor_x, neighbor_y) = if horizontal {
                    let source_x = i64::from(output_x / 2);
                    let neighbor_x = source_x + if output_x & 1 == 0 { -1 } else { 1 };
                    (
                        source_x,
                        i64::from(output_y),
                        neighbor_x,
                        i64::from(output_y),
                    )
                } else {
                    let source_y = i64::from(output_y / 2);
                    let neighbor_y = source_y + if output_y & 1 == 0 { -1 } else { 1 };
                    (
                        i64::from(output_x),
                        source_y,
                        i64::from(output_x),
                        neighbor_y,
                    )
                };
                let source_x = mirror_coordinate(source_x, input_shape.width)?;
                let source_y = mirror_coordinate(source_y, input_shape.height)?;
                let neighbor_x = mirror_coordinate(neighbor_x, input_shape.width)?;
                let neighbor_y = mirror_coordinate(neighbor_y, input_shape.height)?;
                let current = sample(input, input_shape, channel, source_x, source_y)?;
                let neighbor = sample(input, input_shape, channel, neighbor_x, neighbor_y)?;
                let output_index = dense_planar_index(
                    output_shape,
                    channel,
                    usize::try_from(output_x).map_err(|_| Error::LengthOverflow)?,
                    usize::try_from(output_y).map_err(|_| Error::LengthOverflow)?,
                )?;
                output[output_index] = neighbor.mul_add(0.25, current * 0.75);
            }
        }
    }
    Ok(output)
}

fn mirror_coordinate(mut coordinate: i64, size: u32) -> Result<usize> {
    if size <= 1 {
        return Ok(0);
    }
    let size = i64::from(size);
    loop {
        if coordinate < 0 {
            coordinate = -coordinate - 1;
        } else if coordinate >= size {
            coordinate = size * 2 - coordinate - 1;
        } else {
            return usize::try_from(coordinate).map_err(|_| Error::LengthOverflow);
        }
    }
}

fn ycbcr_to_rgb(
    input: &[f32],
    input_shape: &TensorShape,
    output_shape: &TensorShape,
) -> Result<Vec<f32>> {
    require_same_shape(input_shape, output_shape)?;
    if input_shape.channels != 3 {
        return Err(Error::InvalidTensor(
            "YCbCr conversion requires three planar channels in Cb/Y/Cr order".into(),
        ));
    }
    let plane_len =
        usize::try_from(input_shape.channel_stride).map_err(|_| Error::LengthOverflow)?;
    let (cb, rest) = input.split_at(plane_len);
    let (y, cr) = rest.split_at(plane_len);
    let mut output = vec![0.0_f32; input.len()];
    let (red, rest) = output.split_at_mut(plane_len);
    let (green, blue) = rest.split_at_mut(plane_len);
    for index in 0..plane_len {
        let luminance = y[index] + 128.0 / 255.0;
        red[index] = cr[index].mul_add(1.402, luminance);
        green[index] = cr[index].mul_add(
            -0.299 * 1.402 / 0.587,
            cb[index].mul_add(-0.114 * 1.772 / 0.587, luminance),
        );
        blue[index] = cb[index].mul_add(1.772, luminance);
    }
    Ok(output)
}

fn premultiply_alpha(
    operation: &OperationSpec,
    input: &[f32],
    input_shape: &TensorShape,
    output_shape: &TensorShape,
) -> Result<Vec<f32>> {
    require_same_shape(input_shape, output_shape)?;
    let alpha_channel = operation
        .parameters
        .get("alpha_channel")
        .copied()
        .unwrap_or(f64::from(input_shape.channels - 1));
    if !alpha_channel.is_finite()
        || alpha_channel.fract() != 0.0
        || alpha_channel < 0.0
        || alpha_channel >= f64::from(input_shape.channels)
    {
        return Err(Error::InvalidMetadata(format!(
            "invalid alpha channel {alpha_channel}"
        )));
    }
    let alpha_channel = alpha_channel as usize;
    let plane_len =
        usize::try_from(input_shape.channel_stride).map_err(|_| Error::LengthOverflow)?;
    let alpha_start = alpha_channel
        .checked_mul(plane_len)
        .ok_or(Error::LengthOverflow)?;
    let alpha = input
        .get(alpha_start..alpha_start + plane_len)
        .ok_or_else(|| Error::InvalidTensor("alpha plane lies outside input".into()))?;
    let mut output = input.to_vec();
    for channel in 0..usize::from(input_shape.channels) {
        if channel == alpha_channel {
            continue;
        }
        let start = channel
            .checked_mul(plane_len)
            .ok_or(Error::LengthOverflow)?;
        for (value, alpha) in output[start..start + plane_len].iter_mut().zip(alpha) {
            *value *= *alpha;
        }
    }
    Ok(output)
}

fn sample(input: &[f32], shape: &TensorShape, channel: usize, x: usize, y: usize) -> Result<f32> {
    input
        .get(dense_planar_index(shape, channel, x, y)?)
        .copied()
        .ok_or_else(|| Error::InvalidTensor("sample index is outside input".into()))
}

fn dense_planar_index(shape: &TensorShape, channel: usize, x: usize, y: usize) -> Result<usize> {
    let channel_stride =
        usize::try_from(shape.channel_stride).map_err(|_| Error::LengthOverflow)?;
    let row_stride = usize::try_from(shape.row_stride).map_err(|_| Error::LengthOverflow)?;
    channel
        .checked_mul(channel_stride)
        .and_then(|value| value.checked_add(y.checked_mul(row_stride)?))
        .and_then(|value| value.checked_add(x))
        .ok_or(Error::LengthOverflow)
}

fn validate_dense_planar(input: &[f32], shape: &TensorShape) -> Result<()> {
    validate_dense_planar_shape(shape)?;
    let expected = usize::try_from(shape.minimum_elements()?).map_err(|_| Error::LengthOverflow)?;
    if input.len() != expected {
        return Err(Error::InvalidTensor(format!(
            "input has {} elements but shape requires {expected}",
            input.len()
        )));
    }
    Ok(())
}

fn validate_dense_planar_shape(shape: &TensorShape) -> Result<()> {
    let dense = TensorShape::planar(shape.width, shape.height, shape.channels)?;
    if shape != &dense {
        return Err(Error::UnsupportedOperation {
            backend: "reference",
            operation: "non-dense or non-planar tensor layout".into(),
        });
    }
    Ok(())
}

fn require_same_shape(left: &TensorShape, right: &TensorShape) -> Result<()> {
    validate_dense_planar_shape(left)?;
    validate_dense_planar_shape(right)?;
    if left != right {
        return Err(Error::InvalidTensor(
            "input and output tensor shapes differ".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn copy_is_exact() {
        let shape = TensorShape::planar(2, 1, 1).unwrap();
        let output = execute_operation(
            &OperationSpec {
                kind: OperationKind::Copy,
                parameters: BTreeMap::new(),
            },
            &[1.0, -2.0],
            &shape,
            &shape,
        )
        .unwrap();
        assert_eq!(output, [1.0, -2.0]);
    }

    #[test]
    fn gaborish_preserves_constant_field() {
        let shape = TensorShape::planar(3, 2, 1).unwrap();
        let output = execute_operation(
            &OperationSpec {
                kind: OperationKind::Gaborish,
                parameters: BTreeMap::from([
                    ("weight1".into(), 0.115169525),
                    ("weight2".into(), 0.061248592),
                ]),
            },
            &[0.25; 6],
            &shape,
            &shape,
        )
        .unwrap();
        assert!(output.iter().all(|value| (*value - 0.25).abs() < 1.0e-6));
    }

    #[test]
    fn nearest_upsample_replicates_source() {
        let input_shape = TensorShape::planar(2, 1, 1).unwrap();
        let output_shape = TensorShape::planar(4, 2, 1).unwrap();
        let output = execute_operation(
            &OperationSpec {
                kind: OperationKind::Upsample,
                parameters: BTreeMap::from([("factor".into(), 2.0)]),
            },
            &[1.0, 2.0],
            &input_shape,
            &output_shape,
        )
        .unwrap();
        assert_eq!(output, [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
    }
}
