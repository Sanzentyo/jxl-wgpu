use super::*;

pub(super) fn lower_program(
    transform: &IccTransform,
    memory: ResidentIccMemoryPlan,
) -> Result<Vec<u8>, ResidentIccError> {
    let curves = unique_curves(transform);
    let mut header = ProgramHeader::zeroed();
    for (r, row) in transform.matrix().iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            header.matrix[r][c] = *value as f32;
            if !header.matrix[r][c].is_finite() {
                return Err(ResidentIccError::Precision);
            }
        }
    }
    let mut bytes = Vec::with_capacity(memory.program_bytes as usize);
    bytes.resize(std::mem::size_of::<ProgramHeader>(), 0);
    let mut offsets = Vec::with_capacity(curves.len());
    for curve in &curves {
        offsets.push((bytes.len() / 4) as u32);
        let mut record = CurveParams::zeroed();
        record.selectors[3] =
            u32::from(curve.inverse_direction() == Ok(IccInverseDirection::Decreasing));
        match curve.kind() {
            IccCurveKind::Identity => {}
            IccCurveKind::Gamma(value) => {
                record.selectors[0] = 1;
                record.parameters[0][0] = f32::from(*value) / 256.0;
            }
            IccCurveKind::Sampled(samples) => {
                record.selectors[0] = 2;
                record.selectors[2] = samples.len() as u32;
            }
            IccCurveKind::Parametric {
                function,
                parameters,
            } => {
                let [g, a, b, _, d, _, _] = parameters.map(|v| f64::from(v) / 65536.0);
                let power_active = *function <= 2 || d <= 1.0;
                let largest_base = if *function == 0 {
                    1.0
                } else {
                    (a + b).max(0.0)
                };
                if power_active && largest_base.powf(g) > f64::from(f32::MAX) / 2.0 {
                    return Err(ResidentIccError::Precision);
                }
                record.selectors[0] = 3;
                record.selectors[1] = u32::from(*function);
                for (i, value) in parameters.iter().enumerate() {
                    record.parameters[i / 4][i % 4] = *value as f32 / 65536.0;
                }
            }
        }
        bytes.extend_from_slice(bytemuck::bytes_of(&record));
        if let IccCurveKind::Sampled(samples) = curve.kind() {
            for sample in &**samples {
                bytes.extend_from_slice(&u32::from(*sample).to_le_bytes());
            }
        }
    }
    for (target, endpoint) in [
        (&mut header.source_curves, transform.source()),
        (&mut header.target_curves, transform.target()),
    ] {
        target[3] = u32::from(matches!(endpoint, IccTransformEndpoint::LinearRgb(_)));
        for (i, curve) in endpoint.curves().iter().enumerate() {
            target[i] = offsets[curves
                .iter()
                .position(|v| *v == curve)
                .expect("enumerated ICC curve")];
        }
    }
    bytes[..std::mem::size_of::<ProgramHeader>()].copy_from_slice(bytemuck::bytes_of(&header));
    debug_assert_eq!(bytes.len() as u64, memory.program_bytes);
    Ok(bytes)
}
