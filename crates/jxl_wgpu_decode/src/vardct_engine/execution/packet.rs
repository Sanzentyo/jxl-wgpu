use std::sync::Arc;

use super::{
    PacketCommands, VarDctDownstreamCommands, VarDctJobLifetime, VarDctPipelines, lock_unpoisoned,
    submit_vardct_downstream,
};
use crate::GpuCodestream;
use crate::entropy_window::StreamBatch;
use crate::vardct_engine::types::{PACKET_STATUS_BYTES, VarDctDecodeError};
use crate::vardct_engine::window_plan::{
    PacketStage, PacketWindowExecutionPlan, copy_stream_segment,
};
use crate::vardct_packet::{VarDctPacketBuffers, VarDctPacketControl};
use jxl_wgpu::WgpuBackend;

/// Retains the source and per-group geometry until submission; no per-window host copies.
pub(super) struct PacketWindowCommands {
    plan: PacketWindowExecutionPlan,
    source: GpuCodestream,
    controls: Box<[VarDctPacketControl]>,
    status_boundary: bool,
    setup: Option<wgpu::CommandBuffer>,
}

pub(super) fn prepare_packet_windows(
    plan: &PacketWindowExecutionPlan,
    source: &GpuCodestream,
    controls: &[VarDctPacketControl],
    status_boundary: bool,
    first_commands: Option<wgpu::CommandEncoder>,
) -> Result<PacketWindowCommands, VarDctDecodeError> {
    if plan.batch_count() == 0 {
        return Err(VarDctDecodeError::EntropyWindowContract {
            detail: "windowed packet execution has no dispatch",
        });
    }
    if controls.len() != plan.group_count() {
        return Err(VarDctDecodeError::GroupPlanCount {
            component: "packet window control",
            expected: plan.group_count(),
            actual: controls.len(),
        });
    }
    Ok(PacketWindowCommands {
        plan: plan.clone(),
        source: source.clone(),
        controls: controls.into(),
        status_boundary,
        setup: first_commands.map(wgpu::CommandEncoder::finish),
    })
}

impl PacketWindowCommands {
    pub(super) fn batch_count(&self) -> usize {
        self.plan.batch_count()
    }

    fn record(
        &self,
        batch: &StreamBatch<'_>,
        copy_status: bool,
        device: &wgpu::Device,
        pipelines: &VarDctPipelines,
        stream: &wgpu::Buffer,
        lifetime: &VarDctJobLifetime,
    ) -> Result<wgpu::CommandBuffer, VarDctDecodeError> {
        let metadata = lock_unpoisoned(&lifetime._modular_metadata);
        let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu bounded packet stream batch"),
        });
        for segment in batch.segments() {
            let buffers = lifetime._groups.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "packet window has no GPU group",
                },
            )?;
            let metadata_index = if self.plan.stage == PacketStage::Combined {
                0
            } else {
                segment.group_index
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
            match self.plan.stage {
                PacketStage::Lf => {
                    pipelines
                        .packet
                        .encode_lf(device, &mut commands, packet_buffers)
                }
                PacketStage::Hf if self.status_boundary => {
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
        if copy_status {
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

pub(super) fn submit_packet_commands(
    backend: &WgpuBackend,
    pipelines: &VarDctPipelines,
    commands: PacketCommands,
    lifetime: &Arc<VarDctJobLifetime>,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    match commands {
        PacketCommands::Whole(commands) => Ok(backend.queue().submit([commands])),
        PacketCommands::Windowed(commands) => {
            submit_packet_windows(backend, pipelines, commands, None, lifetime)
        }
    }
}

pub(super) fn submit_packet_windows(
    backend: &WgpuBackend,
    pipelines: &VarDctPipelines,
    mut windows: PacketWindowCommands,
    downstream: Option<VarDctDownstreamCommands>,
    lifetime: &Arc<VarDctJobLifetime>,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    let stream = lifetime._packet_stream_window.as_ref().ok_or(
        VarDctDecodeError::EntropyWindowContract {
            detail: "windowed packet commands have no retained stream upload",
        },
    )?;
    let upload_len = usize::try_from(windows.plan.stream_bytes).map_err(|_| {
        VarDctDecodeError::ArithmeticOverflow {
            field: "packet stream window host length",
        }
    })?;
    let mut upload = vec![0; upload_len];
    let queue = backend.queue();
    for index in 0..windows.batch_count() {
        let batch =
            windows
                .plan
                .streams
                .batch(index)
                .ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "packet window batch is missing",
                })?;
        upload.fill(0);
        for &segment in batch.segments() {
            copy_stream_segment(
                &windows.source,
                segment,
                &mut upload,
                "packet segment exceeds the source or reusable upload",
            )?;
            let group = lifetime._groups.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "windowed packet batch references an absent group",
                },
            )?;
            let control = windows.controls.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "packet segment has no control record",
                },
            )?;
            let params = windows.plan.params_for_segment(segment)?;
            queue.write_buffer(&group.packet_control, 0, bytemuck::bytes_of(control));
            queue.write_buffer(&group.modular_params, 0, bytemuck::bytes_of(&params));
        }
        queue.write_buffer(stream, 0, &upload);
        let final_batch = index + 1 == windows.batch_count();
        let commands = windows.record(
            &batch,
            windows.status_boundary && final_batch,
            backend.device(),
            pipelines,
            stream,
            lifetime,
        )?;
        let prefix = windows.setup.take().into_iter().chain([commands]);
        if final_batch {
            return if let Some(downstream) = downstream {
                submit_vardct_downstream(queue, prefix.collect(), downstream, lifetime)
            } else {
                Ok(queue.submit(prefix))
            };
        }
        queue.submit(prefix);
    }
    Err(VarDctDecodeError::EntropyWindowContract {
        detail: "windowed packet execution has no batches",
    })
}
