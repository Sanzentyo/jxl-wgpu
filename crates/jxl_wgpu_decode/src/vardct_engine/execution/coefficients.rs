use super::{HfCoefficientJobBuffers, VarDctGroupJobBuffers, VarDctPipelines};
use crate::GpuCodestream;
use crate::vardct_engine::types::VarDctDecodeError;
use crate::vardct_engine::window_plan::copy_stream_segment;
use crate::vardct_pass_group::{HfCoefficientBuffers, HfCoefficientExecutionPlan};

/// Host uploads and dispatch geometry only. Recording all command buffers before the first
/// submit can exhaust Metal's command resources when a small cap produces thousands of windows.
pub(super) struct HfCoefficientBatchSubmission {
    pub(super) stream_upload: Box<[u8]>,
    pub(super) params_upload: Box<[u8]>,
    group_index: usize,
    group_count: u32,
}

pub(super) fn prepare_hf_batches(
    source: &GpuCodestream,
    plan: &HfCoefficientExecutionPlan,
) -> Result<Vec<HfCoefficientBatchSubmission>, VarDctDecodeError> {
    let upload_len = usize::try_from(plan.stream_window_bytes()).map_err(|_| {
        VarDctDecodeError::ArithmeticOverflow {
            field: "HF stream window host length",
        }
    })?;
    let mut batches = Vec::with_capacity(plan.stream_batch_count());
    for (group_index, group) in plan.groups.iter().enumerate() {
        for batch in &group.stream_batches {
            let mut stream_upload = vec![0_u8; upload_len];
            for segment in &group.stream_segments[batch.segments.clone()] {
                copy_stream_segment(
                    source,
                    *segment,
                    &mut stream_upload,
                    "HF stream segment exceeds the source or reusable upload",
                )?;
            }
            batches.push(HfCoefficientBatchSubmission {
                stream_upload: stream_upload.into_boxed_slice(),
                params_upload: bytemuck::cast_slice(&group.segment_params[batch.segments.clone()])
                    .to_vec()
                    .into_boxed_slice(),
                group_index,
                group_count: u32::try_from(batch.group_count).map_err(|_| {
                    VarDctDecodeError::ArithmeticOverflow {
                        field: "HF stream batch dispatch count",
                    }
                })?,
            });
        }
    }
    Ok(batches)
}

impl HfCoefficientBatchSubmission {
    pub(super) fn record(
        &self,
        device: &wgpu::Device,
        pipelines: &VarDctPipelines,
        buffers: &HfCoefficientJobBuffers,
        groups: &[VarDctGroupJobBuffers],
    ) -> Result<wgpu::CommandBuffer, VarDctDecodeError> {
        let group =
            groups
                .get(self.group_index)
                .ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "HF coefficient batch has no GPU group",
                })?;
        let hf = buffers.groups.get(self.group_index).ok_or(
            VarDctDecodeError::EntropyWindowContract {
                detail: "HF coefficient batch has no status group",
            },
        )?;
        let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu bounded HF coefficient stream batch"),
        });
        pipelines.hf_coefficients.encode(
            device,
            &mut commands,
            HfCoefficientBuffers {
                codestream: buffers.stream_window.as_ref().ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "HF batch has no stream upload",
                    },
                )?,
                entropy_bundle: &buffers.entropy_bundle,
                reconstruction: &group.reconstructed,
                params: buffers.params_window.as_ref().ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "HF batch has no parameter upload",
                    },
                )?,
                status: &hf.status,
                artifact: &group.artifact,
                order_table: &buffers.order_table,
                coefficients: &group.coefficients,
                sink_params: &hf.sink_params,
            },
            self.group_count,
        );
        Ok(commands.finish())
    }
}
