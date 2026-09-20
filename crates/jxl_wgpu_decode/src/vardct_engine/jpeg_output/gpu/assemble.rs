use super::super::{GpuJpegFrame, runtime::Plan};
use super::*;

#[derive(Debug)]
pub(in super::super) struct Assembly {
    output: GpuBufferLease,
    byte_len: u32,
    expected: [u32; 4],
    pub operation: Operation,
}
impl Assembly {
    pub(in super::super) fn start(
        resources: &Resources,
        coefficients: &GpuJpegCoefficients,
        plan: &Plan,
        scans: &[(GpuBufferLease, u32)],
    ) -> crate::Result<Self> {
        let template = &plan.framing;
        if scans.len() != template.scan_offsets.len() {
            return Err(Error::Invalid("scan count").into());
        }
        let mut length = template.bytes.len() as u64;
        for (_, bytes) in scans {
            length = add(length, u64::from(*bytes))?;
        }
        check("JPEG output bytes", length, plan.limits.max_output_bytes)?;
        let byte_len = word(length)?;
        let output_bytes = storage_bytes(length)?;
        let template_bytes = storage_bytes(template.bytes.len() as u64)?;
        let dispatch_count = scans.len() as u64 * 2 + 1 + template.quantizers.len() as u64 + 1;
        let metadata_bytes = (12 + 192) * 4;
        let mut allocator = Allocator::new(
            resources,
            add(
                add(output_bytes, template_bytes)?,
                metadata_bytes + 16 + 16 + dispatch_count * 32,
            )?,
        )?;
        let output = allocator.buffer(output_bytes, STORAGE, &[])?;
        let framing = allocator.buffer(template_bytes, STORAGE, &[&template.bytes])?;
        let status = allocator.buffer(16, STORAGE, &[])?;
        let mut metadata = [0u32; 204];
        for (i, plane) in coefficients.layout().planes().iter().enumerate() {
            metadata[i * 4..i * 4 + 4].copy_from_slice(&[
                plane.coefficient_word_offset,
                plane.blocks_per_row * plane.block_rows * 64,
                12 + i as u32 * 64,
                0,
            ]);
            metadata[12 + i * 64..12 + (i + 1) * 64].copy_from_slice(&plan.scans.masks[i]);
        }
        let metadata =
            allocator.buffer(metadata_bytes, STORAGE, &[bytemuck::cast_slice(&metadata)])?;
        let mut commands = resources
            .backend
            .device()
            .create_command_encoder(&Default::default());
        let mut held = vec![
            output.clone(),
            framing.clone(),
            status.clone(),
            metadata.clone(),
            coefficients.buffer().clone(),
        ];
        let mut dispatch = |phase: usize,
                            source: &GpuBufferLease,
                            params: [u32; 8],
                            groups: u32|
         -> crate::Result<()> {
            let uniform = allocator.uniform(&params)?;
            // Empty gaps still have a charged parameter buffer; no zero-sized dispatch is submitted.
            if groups != 0 {
                resources.dispatch(
                    &mut commands,
                    &resources.pipelines.assembly[phase],
                    &resources.pipelines.assembly_layout,
                    &[
                        &uniform,
                        source,
                        coefficients.buffer(),
                        &metadata,
                        &output,
                        &status,
                    ],
                    groups,
                )?;
            }
            held.push(uniform);
            Ok(())
        };
        let (mut source_start, mut destination) = (0u32, 0u32);
        for (&offset, (scan, bytes)) in template.scan_offsets.iter().zip(scans) {
            let count = offset
                .checked_sub(source_start)
                .ok_or(Error::Invalid("template offsets"))?;
            dispatch(
                0,
                &framing,
                [
                    count,
                    source_start,
                    destination,
                    resources.width(),
                    0,
                    0,
                    0,
                    0,
                ],
                count.div_ceil(64),
            )?;
            destination += count;
            dispatch(
                0,
                scan,
                [*bytes, 0, destination, resources.width(), 0, 0, 0, 0],
                bytes.div_ceil(64),
            )?;
            destination += *bytes;
            source_start = offset;
        }
        let remaining = word(template.bytes.len() as u64)? - source_start;
        dispatch(
            0,
            &framing,
            [
                remaining,
                source_start,
                destination,
                resources.width(),
                0,
                0,
                0,
                0,
            ],
            remaining.div_ceil(64),
        )?;
        let mut patched = 0u64;
        for &[offset, source, mask, precision] in &template.quantizers {
            let mut destination = u64::from(offset);
            for (&scan_offset, (_, bytes)) in template.scan_offsets.iter().zip(scans) {
                if scan_offset <= offset {
                    destination = add(destination, u64::from(*bytes))?;
                }
            }
            dispatch(
                1,
                &framing,
                [
                    64,
                    source,
                    word(destination)?,
                    resources.width(),
                    mask,
                    precision,
                    0,
                    0,
                ],
                1,
            )?;
            patched = add(patched, 64 * (1 + u64::from(precision)))?;
        }
        let words = coefficients.layout().coefficient_words();
        dispatch(
            2,
            &framing,
            [
                words,
                0,
                0,
                resources.width(),
                0,
                0,
                coefficients.layout().planes().len() as u32,
                words,
            ],
            words.div_ceil(64),
        )?;
        held.extend(scans.iter().map(|(buffer, _)| buffer.clone()));
        let staging = allocator.staging()?;
        let operation = submit(resources, commands, &status, staging, held, coefficients)?;
        Ok(Self {
            output,
            byte_len,
            expected: [0, byte_len, word(patched)?, words],
            operation,
        })
    }
    pub(in super::super) fn finish(self, status: [u32; 4]) -> crate::Result<GpuJpegFrame> {
        if status != self.expected {
            return Err(Error::GpuStatus {
                stage: "assembly",
                status,
            }
            .into());
        }
        Ok(GpuJpegFrame {
            buffer: self.output,
            byte_len: u64::from(self.byte_len),
        })
    }
}
