use super::{
    PacketCommands, VarDctDownstreamCommands, VarDctJobLifetime, VarDctPipelines, lock_unpoisoned,
    submit_vardct_downstream,
};
use crate::GpuCodestream;
use crate::vardct_engine::types::{PACKET_STATUS_BYTES, VarDctDecodeError};
use crate::vardct_engine::window_plan::{
    PacketStage, PacketWindowExecutionPlan, copy_stream_segment,
};
use crate::vardct_packet::{VarDctModularParams, VarDctPacketBuffers, VarDctPacketControl};
use jxl_wgpu::WgpuBackend;

struct PacketGroupUpload {
    group_index: usize,
    control: VarDctPacketControl,
    params: VarDctModularParams,
}

pub(super) struct PacketBatchSubmission {
    stream_upload: Box<[u8]>,
    groups: Box<[PacketGroupUpload]>,
    stage: PacketStage,
    stop_at_metadata: bool,
    copy_status: bool,
    setup: Option<wgpu::CommandBuffer>,
}

pub(super) fn prepare_packet_windows(
    plan: &PacketWindowExecutionPlan,
    source: &GpuCodestream,
    controls: &[VarDctPacketControl],
    status_boundary: bool,
    first_commands: Option<wgpu::CommandEncoder>,
) -> Result<Vec<PacketBatchSubmission>, VarDctDecodeError> {
    let mut setup = first_commands.map(wgpu::CommandEncoder::finish);
    let upload_len =
        usize::try_from(plan.stream_bytes).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
            field: "packet stream window host length",
        })?;
    let mut submissions = Vec::with_capacity(plan.batch_count());
    for (batch_index, batch) in plan.stream_batches.iter().enumerate() {
        if batch.group_count == 0 || batch.segments.is_empty() {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "packet batch contains no segment",
            });
        }
        let mut stream_upload = vec![0_u8; upload_len];
        let mut group_uploads = Vec::with_capacity(batch.group_count);
        for segment_index in batch.segments.clone() {
            let segment = *plan.stream_segments.get(segment_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "packet batch references an absent segment",
                },
            )?;
            let control = *controls.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "packet segment has no control record",
                },
            )?;
            let params = *plan.segment_params.get(segment_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "packet segment has no parameter record",
                },
            )?;
            copy_stream_segment(
                source,
                segment,
                &mut stream_upload,
                "packet segment exceeds the source or reusable upload",
            )?;
            group_uploads.push(PacketGroupUpload {
                group_index: segment.group_index,
                control,
                params,
            });
        }
        if group_uploads.len() != batch.group_count {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "packet batch group count disagrees with its segments",
            });
        }
        submissions.push(PacketBatchSubmission {
            stream_upload: stream_upload.into_boxed_slice(),
            groups: group_uploads.into_boxed_slice(),
            stage: plan.stage,
            stop_at_metadata: status_boundary,
            copy_status: status_boundary && batch_index + 1 == plan.batch_count(),
            setup: setup.take(),
        });
    }
    if submissions.is_empty() {
        return Err(VarDctDecodeError::EntropyWindowContract {
            detail: "windowed packet execution has no dispatch",
        });
    }
    Ok(submissions)
}

impl PacketBatchSubmission {
    fn record(
        &self,
        device: &wgpu::Device,
        pipelines: &VarDctPipelines,
        stream: &wgpu::Buffer,
        lifetime: &VarDctJobLifetime,
    ) -> Result<wgpu::CommandBuffer, VarDctDecodeError> {
        let metadata = lock_unpoisoned(&lifetime._modular_metadata);
        let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu bounded packet stream batch"),
        });
        for upload in &self.groups {
            let buffers = lifetime._groups.get(upload.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "packet window has no GPU group",
                },
            )?;
            let metadata_index = if self.stage == PacketStage::Combined {
                0
            } else {
                upload.group_index
            };
            let modular_metadata =
                metadata
                    .get(metadata_index)
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "packet window has no Modular metadata",
                    })?;
            let packet_buffers = VarDctPacketBuffers {
                codestream: stream,
                modular_metadata,
                reconstructed_lf: &buffers.reconstructed,
                raw_hf_metadata: &buffers.raw_metadata,
                coefficients: &buffers.coefficients,
                status: &buffers.packet_status,
                control: &buffers.packet_control,
                modular_params: &buffers.modular_params,
            };
            match self.stage {
                PacketStage::Lf => {
                    pipelines
                        .packet
                        .encode_lf(device, &mut commands, packet_buffers)
                }
                PacketStage::Hf if self.stop_at_metadata => {
                    pipelines
                        .packet
                        .encode_hf_metadata(device, &mut commands, packet_buffers)
                }
                PacketStage::Hf => {
                    pipelines
                        .packet
                        .encode_hf(device, &mut commands, packet_buffers)
                }
                PacketStage::Combined => {
                    pipelines
                        .packet
                        .encode(device, &mut commands, packet_buffers)
                }
            }
        }
        if self.copy_status {
            for (index, group) in lifetime._groups.iter().enumerate() {
                commands.copy_buffer_to_buffer(
                    &group.packet_status,
                    0,
                    &lifetime.status_staging,
                    index as u64 * PACKET_STATUS_BYTES,
                    PACKET_STATUS_BYTES,
                );
            }
        }
        Ok(commands.finish())
    }
}

fn write_packet_batch(
    queue: &wgpu::Queue,
    stream: &wgpu::Buffer,
    batch: &PacketBatchSubmission,
    lifetime: &VarDctJobLifetime,
) -> Result<(), VarDctDecodeError> {
    queue.write_buffer(stream, 0, &batch.stream_upload);
    for upload in &batch.groups {
        let group = lifetime._groups.get(upload.group_index).ok_or(
            VarDctDecodeError::EntropyWindowContract {
                detail: "windowed packet batch references an absent group",
            },
        )?;
        queue.write_buffer(
            &group.packet_control,
            0,
            bytemuck::bytes_of(&upload.control),
        );
        queue.write_buffer(&group.modular_params, 0, bytemuck::bytes_of(&upload.params));
    }
    Ok(())
}

pub(super) fn submit_packet_commands(
    backend: &WgpuBackend,
    pipelines: &VarDctPipelines,
    commands: PacketCommands,
    lifetime: &VarDctJobLifetime,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    match commands {
        PacketCommands::Whole(commands) => Ok(backend.queue().submit([commands])),
        PacketCommands::Windowed(batches) => {
            submit_packet_batches(backend, pipelines, batches, None, lifetime)
        }
    }
}

pub(super) fn submit_packet_batches(
    backend: &WgpuBackend,
    pipelines: &VarDctPipelines,
    mut batches: Vec<PacketBatchSubmission>,
    downstream: Option<VarDctDownstreamCommands>,
    lifetime: &VarDctJobLifetime,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    let stream = lifetime._packet_stream_window.as_ref().ok_or(
        VarDctDecodeError::EntropyWindowContract {
            detail: "windowed packet commands have no retained stream upload",
        },
    )?;
    let mut final_batch = batches
        .pop()
        .ok_or(VarDctDecodeError::EntropyWindowContract {
            detail: "windowed packet execution has no batches",
        })?;
    let queue = backend.queue();
    for mut batch in batches {
        write_packet_batch(queue, stream, &batch, lifetime)?;
        let commands = batch.record(backend.device(), pipelines, stream, lifetime)?;
        queue.submit(batch.setup.take().into_iter().chain([commands]));
    }
    write_packet_batch(queue, stream, &final_batch, lifetime)?;
    let commands = final_batch.record(backend.device(), pipelines, stream, lifetime)?;
    let commands = final_batch
        .setup
        .take()
        .into_iter()
        .chain([commands])
        .collect::<Vec<_>>();
    if let Some(downstream) = downstream {
        submit_vardct_downstream(queue, commands, downstream, lifetime)
    } else {
        Ok(queue.submit(commands))
    }
}
