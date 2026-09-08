use super::{HfCoefficientJobBuffers, VarDctGroupJobBuffers, VarDctPipelines};
use crate::GpuCodestream;
use crate::vardct_engine::types::VarDctDecodeError;
use crate::vardct_engine::window_plan::copy_stream_segment;
use crate::vardct_pass_group::{
    HfCoefficientBuffers, HfCoefficientExecutionPlan, HfCoefficientGroupExecutionPlan,
    HfCoefficientPassParams,
};

/// Per-group parameter snapshots and immutable stream geometry. Host and GPU work for each
/// window is created only when submitted; the compressed source remains leased until then.
pub(super) struct HfCoefficientWindowCommands {
    source: GpuCodestream,
    groups: Box<[HfCoefficientGroupExecutionPlan]>,
    stream_bytes: usize,
    max_group_count: usize,
    batch_count: usize,
}

pub(super) fn prepare_hf_windows(
    source: &GpuCodestream,
    plan: &HfCoefficientExecutionPlan,
) -> Result<HfCoefficientWindowCommands, VarDctDecodeError> {
    let stream_bytes = usize::try_from(plan.stream_window_bytes()).map_err(|_| {
        VarDctDecodeError::ArithmeticOverflow {
            field: "HF stream window host length",
        }
    })?;
    let batch_count = plan.stream_batch_count();
    let max_group_count = plan
        .groups
        .iter()
        .map(|group| group.streams.max_group_count())
        .max()
        .unwrap_or(0);
    if batch_count == 0 || max_group_count == 0 {
        return Err(VarDctDecodeError::EntropyWindowContract {
            detail: "windowed HF coefficient plan has no batches",
        });
    }
    u32::try_from(max_group_count).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
        field: "HF stream batch dispatch count",
    })?;
    Ok(HfCoefficientWindowCommands {
        source: source.clone(),
        groups: plan.groups.clone().into_boxed_slice(),
        stream_bytes,
        max_group_count,
        batch_count,
    })
}

impl HfCoefficientWindowCommands {
    pub(super) fn submit(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipelines: &VarDctPipelines,
        buffers: &HfCoefficientJobBuffers,
        groups: &[VarDctGroupJobBuffers],
        mut before: Option<wgpu::CommandBuffer>,
    ) -> Result<usize, VarDctDecodeError> {
        let stream =
            buffers
                .stream_window
                .as_ref()
                .ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "HF batch has no stream upload",
                })?;
        let params_buffer =
            buffers
                .params_window
                .as_ref()
                .ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "HF batch has no parameter upload",
                })?;
        let mut stream_upload = vec![0; self.stream_bytes];
        let mut params_upload = Vec::<HfCoefficientPassParams>::with_capacity(self.max_group_count);
        for (group_index, plan) in self.groups.iter().enumerate() {
            let group =
                groups
                    .get(group_index)
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "HF coefficient batch has no GPU group",
                    })?;
            let hf = buffers.groups.get(group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "HF coefficient batch has no status group",
                },
            )?;
            for batch in plan.streams.batches() {
                stream_upload.fill(0);
                params_upload.clear();
                for &segment in batch.segments() {
                    copy_stream_segment(
                        &self.source,
                        segment,
                        &mut stream_upload,
                        "HF stream segment exceeds the source or reusable upload",
                    )?;
                    params_upload.push(plan.params_for_segment(segment).ok_or(
                        VarDctDecodeError::EntropyWindowContract {
                            detail: "HF coefficient segment has no base parameter record",
                        },
                    )?);
                }
                queue.write_buffer(stream, 0, &stream_upload);
                queue.write_buffer(params_buffer, 0, bytemuck::cast_slice(&params_upload));
                let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu bounded HF coefficient stream batch"),
                });
                pipelines.hf_coefficients.encode(
                    device,
                    &mut commands,
                    HfCoefficientBuffers {
                        codestream: stream,
                        entropy_bundle: &buffers.entropy_bundle,
                        reconstruction: &group.reconstructed,
                        params: params_buffer,
                        status: &hf.status,
                        artifact: &group.artifact,
                        order_table: &buffers.order_table,
                        coefficients: &group.coefficients,
                        sink_params: &hf.sink_params,
                    },
                    batch.group_count() as u32,
                );
                queue.submit(before.take().into_iter().chain([commands.finish()]));
            }
        }
        Ok(self.batch_count)
    }
}
