use super::{HfCoefficientJobBuffers, VarDctGroupJobBuffers, VarDctPipelines};
use crate::GpuCodestream;
use crate::vardct_engine::types::VarDctDecodeError;
use crate::vardct_engine::window_plan::copy_stream_segment;
use crate::vardct_pass_group::{
    HfCoefficientBuffers, HfCoefficientExecutionPlan, HfCoefficientGroupExecutionPlan,
    HfCoefficientPassParams,
};

#[derive(Clone, Copy)]
pub(super) struct HfCoefficientPassBuffers<'a> {
    pub(super) source: &'a wgpu::Buffer,
    pub(super) plan: &'a HfCoefficientExecutionPlan,
    pub(super) jobs: &'a HfCoefficientJobBuffers,
    pub(super) groups: &'a [VarDctGroupJobBuffers],
}

pub(super) fn record_hf_passes(
    device: &wgpu::Device,
    pipelines: &VarDctPipelines,
    inputs: HfCoefficientPassBuffers<'_>,
) -> Result<Vec<wgpu::CommandBuffer>, VarDctDecodeError> {
    (0..inputs.plan.pass_count())
        .map(|pass| {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu image-wide HF coefficient pass"),
            });
            encode_hf_pass(device, &mut encoder, pipelines, inputs, pass)?;
            Ok(encoder.finish())
        })
        .collect()
}

/// One image-wide accumulation stage, with original per-group validation/state offsets.
pub(super) fn encode_hf_pass(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &VarDctPipelines,
    inputs: HfCoefficientPassBuffers<'_>,
    pass_index: usize,
) -> Result<(), VarDctDecodeError> {
    if inputs.plan.groups.len() != inputs.jobs.groups.len()
        || inputs.plan.groups.len() != inputs.groups.len()
    {
        return Err(VarDctDecodeError::EntropyWindowContract {
            detail: "HF pass has inconsistent LF-group resources",
        });
    }
    for ((group_plan, hf), group) in inputs
        .plan
        .groups
        .iter()
        .zip(&inputs.jobs.groups)
        .zip(inputs.groups)
    {
        let pass =
            group_plan
                .passes
                .get(pass_index)
                .ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "HF coefficient group has no requested pass",
                })?;
        let params = hf
            .params
            .as_ref()
            .and_then(|params| params.get(pass_index))
            .ok_or(VarDctDecodeError::EntropyWindowContract {
                detail: "whole-range HF pass has no parameter buffer",
            })?;
        pipelines.hf_coefficients.encode(
            device,
            encoder,
            HfCoefficientBuffers {
                codestream: inputs.source,
                entropy_bundle: &inputs.jobs.entropy_bundle,
                reconstruction: &group.reconstructed,
                params,
                status: &hf.status,
                artifact: &group.artifact,
                order_table: &inputs.jobs.order_table,
                coefficients: &group.coefficients,
                sink_params: &hf.sink_params,
            },
            u32::try_from(pass.parameter_range.len()).map_err(|_| {
                VarDctDecodeError::ArithmeticOverflow {
                    field: "HF pass dispatch count",
                }
            })?,
        );
    }
    Ok(())
}

/// Per-group parameter snapshots and immutable stream geometry. Host and GPU work for each
/// window is created only when submitted; the compressed source remains leased until then.
pub(super) struct HfCoefficientWindowCommands {
    source: GpuCodestream,
    groups: Box<[HfCoefficientGroupExecutionPlan]>,
    stream_bytes: usize,
    max_group_count: usize,
    batch_count: usize,
    pass_count: usize,
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
        .stream_plans()
        .map(crate::entropy_window::EntropyStreamPlan::max_group_count)
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
        pass_count: plan.pass_count(),
    })
}

impl HfCoefficientWindowCommands {
    pub(super) fn pass_count(&self) -> usize {
        self.pass_count
    }

    pub(super) fn submit(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipelines: &VarDctPipelines,
        buffers: &HfCoefficientJobBuffers,
        groups: &[VarDctGroupJobBuffers],
        mut before: Option<wgpu::CommandBuffer>,
    ) -> Result<usize, VarDctDecodeError> {
        let mut submitted = 0;
        for pass in 0..self.pass_count {
            submitted += self.submit_pass(
                device,
                queue,
                pipelines,
                buffers,
                groups,
                pass,
                before.take(),
            )?;
        }
        debug_assert_eq!(submitted, self.batch_count);
        Ok(submitted)
    }

    /// Completes this pass for every LF group before any later pass can run. The immutable plan
    /// and compressed source remain available for the next accumulation stage.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn submit_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipelines: &VarDctPipelines,
        buffers: &HfCoefficientJobBuffers,
        groups: &[VarDctGroupJobBuffers],
        pass_index: usize,
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
        let mut submitted = 0;
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
            let pass =
                plan.passes
                    .get(pass_index)
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "HF coefficient group has no requested pass",
                    })?;
            for batch in pass.streams.batches() {
                stream_upload.fill(0);
                params_upload.clear();
                for &segment in batch.segments() {
                    copy_stream_segment(
                        &self.source,
                        segment,
                        &mut stream_upload,
                        "HF stream segment exceeds the source or reusable upload",
                    )?;
                    params_upload.push(plan.params_for_segment(pass_index, segment).ok_or(
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
                submitted += 1;
            }
        }
        Ok(submitted)
    }
}
