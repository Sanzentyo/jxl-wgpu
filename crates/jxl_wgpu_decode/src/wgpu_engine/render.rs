//! Rendering is independent of the entropy assembly and of the pending frame's control state.

use std::num::NonZeroU64;
use std::sync::Mutex;

use jxl_wgpu::ResidentStorageBinding;

use crate::buffer_pool::DecodeBufferLease;
use crate::modular_finalize::{ModularFinalizeBindings, ModularFinalizeParams};
use crate::modular_render::ModularRenderBuffers;
use crate::modular_transform::GpuModularChannelLayout;
use crate::{Error, Result};

use super::execution::OutputPlan;
use super::lifetime::DecodeJobLifetime;
use super::types::ModularInversePipelines;

pub(super) struct ModularRenderTarget<'a> {
    pub output: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
    pub native_f64_dummy: Option<&'a wgpu::Buffer>,
    pub render: Option<&'a ModularRenderBuffers>,
    pub render_uniforms: &'a Mutex<Vec<wgpu::Buffer>>,
}

impl<'a> ModularRenderTarget<'a> {
    pub(super) fn for_job(job: &'a DecodeJobLifetime) -> Self {
        Self {
            output: job.output.as_wgpu_buffer(),
            status: job._status.buffer(),
            native_f64_dummy: job
                ._native_f64_dummy_words
                .as_ref()
                .map(DecodeBufferLease::buffer),
            render: job.render.as_ref(),
            render_uniforms: &job.render_uniforms,
        }
    }
}

pub(super) struct ModularRenderInput<'a> {
    pub arena: ResidentStorageBinding<'a>,
    pub planes: &'a [GpuModularChannelLayout],
    pub params: &'a [ModularFinalizeParams],
}

pub(super) fn encode_modular_output(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    output_plan: &OutputPlan,
    pipelines: &ModularInversePipelines,
    target: ModularRenderTarget<'_>,
    input: ModularRenderInput<'_>,
) -> Result<Vec<wgpu::Buffer>> {
    let arena = if let Some(plan) = &output_plan.render {
        let buffers = target
            .render
            .ok_or(Error::EngineContract("missing Modular render buffers"))?;
        let render = pipelines
            .render
            .as_ref()
            .ok_or(Error::EngineContract("missing Modular render pipeline"))?;
        let uniforms = render.encode(
            device,
            encoder,
            plan,
            buffers,
            input.arena,
            &output_plan.source_channels.select(input.planes)?,
        )?;
        target
            .render_uniforms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(uniforms);
        ResidentStorageBinding::entire(&buffers.output)
            .map_err(|_| Error::EngineContract("empty Modular render arena"))?
    } else {
        input.arena
    };
    let output_words = target.native_f64_dummy.unwrap_or(target.output);
    let binding = |buffer: &wgpu::Buffer| {
        NonZeroU64::new(buffer.size()).ok_or(Error::EngineContract("empty Modular output binding"))
    };
    let output_size = binding(target.output)?;
    let word_size = binding(output_words)?;
    let status_size = binding(target.status)?;
    input
        .params
        .iter()
        .map(|&params| {
            pipelines
                .finalize
                .encode(
                    device,
                    encoder,
                    ModularFinalizeBindings {
                        arena,
                        output_words: ResidentStorageBinding {
                            buffer: output_words,
                            offset: 0,
                            size: word_size,
                        },
                        status: ResidentStorageBinding {
                            buffer: target.status,
                            offset: 0,
                            size: status_size,
                        },
                        output_f64: target.native_f64_dummy.map(|_| ResidentStorageBinding {
                            buffer: target.output,
                            offset: 0,
                            size: output_size,
                        }),
                    },
                    params,
                )
                .map_err(Error::from)
        })
        .collect()
}
