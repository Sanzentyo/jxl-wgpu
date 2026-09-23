//! Shared resident/streamed upload of parameters and planned transform metadata.
use super::dispatch::{ModularDispatchBatch, ModularDispatchPlan};
use super::types::ModularParams;
use crate::{BackendError, EncodeError};

pub(super) struct TransformUpload {
    source_offset: u64,
    destination_offset: u64,
    bytes: u64,
}

impl TransformUpload {
    pub(super) fn record(
        &self,
        commands: &mut wgpu::CommandEncoder,
        parameters: &wgpu::Buffer,
        artifact: &wgpu::Buffer,
    ) {
        commands.copy_buffer_to_buffer(
            parameters,
            self.source_offset,
            artifact,
            self.destination_offset,
            self.bytes,
        );
    }
}

impl ModularDispatchPlan {
    pub(super) fn upload_parameters(
        &self,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        batch: ModularDispatchBatch,
    ) -> Result<Vec<TransformUpload>, EncodeError> {
        let range = batch.first_dispatch..batch.first_dispatch + batch.dispatch_count;
        let parameters = self
            .parameters
            .get(range.clone())
            .ok_or(BackendError::Invariant("invalid parameter upload range"))?;
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(parameters));
        let mut source_offset =
            batch.dispatch_count as u64 * std::mem::size_of::<ModularParams>() as u64;
        let mut uploads = Vec::new();
        for (group, params) in self.groups[range].iter().zip(parameters) {
            if group.transform_metadata_words == 0 {
                continue;
            }
            let topology = self.transforms.group(
                self.group_grid
                    .group(group.group_index)
                    .ok_or(BackendError::Invariant("transform upload group missing"))?,
            )?;
            let program = topology
                .transform_program
                .as_ref()
                .ok_or(BackendError::Invariant("missing transform upload program"))?;
            let words = program.metadata();
            let bytes = words.len() as u64 * 4;
            if words.len() != group.transform_metadata_words as usize
                || source_offset + bytes > batch.parameter_bytes
            {
                return Err(BackendError::Invariant(
                    "transform upload exceeds parameter allocation",
                )
                .into());
            }
            queue.write_buffer(buffer, source_offset, bytemuck::cast_slice(&words));
            uploads.push(TransformUpload {
                source_offset,
                destination_offset: u64::from(params.transform_program_word_offset) * 4,
                bytes,
            });
            source_offset += bytes;
        }
        let complete = if let Some(entropy) = batch.entropy {
            source_offset <= entropy.parameter_offset
                && entropy.parameter_offset + entropy.bytes(batch.dispatch_count)
                    == batch.parameter_bytes
        } else {
            source_offset == batch.parameter_bytes
        };
        if !complete {
            return Err(BackendError::Invariant("incomplete transform parameter upload").into());
        }
        Ok(uploads)
    }
}
