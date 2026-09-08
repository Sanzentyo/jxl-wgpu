//! Raw matrix overlay around the shared resident Modular substream executor.
mod modular;

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::TransformKind;
use jxl_wgpu::{KernelVariant, WgpuBackend};
use wgpu::util::DeviceExt;

use super::types::STATUS_OK;
use crate::vardct_resource::{VarDctResourceLayout, hf_matrix_param_index};
use crate::vardct_side_image::RawHfDequantSideImagePlan;
use crate::{Error, Result};
pub(crate) use modular::{
    ModularSideImageJob, ModularSideImagePipeline, ModularSideImageStatus,
    ModularSideImageStreamPlan,
};

const OVERLAY_SHADER: &str = include_str!("../vardct_raw_matrix.wgsl");
const ERROR_RAW_MATRIX_VALUE: u32 = 15;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RawMatrixParams {
    denominator: f32,
    width: u32,
    height: u32,
    target_count: u32,
    plane_offsets: [u32; 4],
    plane_strides: [u32; 4],
    target_offsets: [u32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<RawMatrixParams>() == 64);
    assert!(std::mem::align_of::<RawMatrixParams>() == 16);
};

pub(crate) struct RawHfDequantSideImagePipeline {
    image: ModularSideImagePipeline,
    overlay: wgpu::ComputePipeline,
    variant: KernelVariant,
}
impl RawHfDequantSideImagePipeline {
    pub(crate) fn modular(&self) -> &ModularSideImagePipeline {
        &self.image
    }
    pub(crate) fn new(backend: &WgpuBackend, variant: KernelVariant) -> Self {
        let module = backend
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("jxl-wgpu raw HF dequant matrix overlay"),
                source: wgpu::ShaderSource::Wgsl(OVERLAY_SHADER.into()),
            });
        let (workgroup_x, _) = variant.workgroup_size();
        let overlay = backend
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu raw HF dequant matrix overlay"),
                layout: None,
                module: &module,
                entry_point: Some("overlay"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("wg_x", f64::from(workgroup_x))],
                    ..Default::default()
                },
                cache: None,
            });
        Self {
            image: ModularSideImagePipeline::new(backend, variant),
            overlay,
            variant,
        }
    }
    pub(crate) fn prepare(
        &self,
        backend: &WgpuBackend,
        codestream: &crate::GpuCodestream,
        resources: &wgpu::Buffer,
        resource_layout: VarDctResourceLayout,
        plan: &RawHfDequantSideImagePlan,
        stream: &ModularSideImageStreamPlan,
    ) -> Result<ModularSideImageJob> {
        let device = backend.device();
        let mut recording = self
            .image
            .record_source(backend, codestream, &plan.image, stream)?;
        let params = overlay_params(resource_layout, plan)?;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu raw HF dequant overlay parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu raw HF dequant overlay bindings"),
            layout: &self.overlay.get_bind_group_layout(0),
            entries: &[
                entry(0, recording.arena()),
                entry(1, resources),
                entry(2, recording.status()),
                entry(3, &uniform),
            ],
        });
        {
            let mut pass = recording.completion_encoder(device).begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("jxl-wgpu raw HF dequant matrix overlay"),
                    timestamp_writes: None,
                },
            );
            pass.set_pipeline(&self.overlay);
            pass.set_bind_group(0, &binding, &[]);
            let samples = params
                .width
                .checked_mul(params.height)
                .ok_or_else(|| Error::backend("raw HF dequant overlay extent overflow"))?;
            pass.dispatch_workgroups(samples.div_ceil(self.variant.workgroup_size().0), 1, 1);
        }
        recording.retain_uniform(uniform)?;
        Ok(recording.finish())
    }
    pub(crate) fn plan_source(
        &self,
        source: &crate::GpuCodestream,
        plan: &RawHfDequantSideImagePlan,
        packet_end: u32,
        stream_limit: u64,
    ) -> Result<ModularSideImageStreamPlan> {
        let mut stream = self
            .image
            .plan_source(source, &plan.image, packet_end, stream_limit)?;
        stream.memory_bytes = stream
            .memory_bytes
            .checked_add(std::mem::size_of::<RawMatrixParams>() as u64)
            .ok_or_else(|| Error::backend("raw HF dequant memory bytes overflow"))?;
        Ok(stream)
    }
}

fn overlay_params(
    resource_layout: VarDctResourceLayout,
    plan: &RawHfDequantSideImagePlan,
) -> Result<RawMatrixParams> {
    let mut target_offsets = [0_u32; 4];
    let mut target_count = 0_usize;
    for (index, transform) in TransformKind::ALL.into_iter().enumerate() {
        if hf_matrix_param_index(transform) == plan.matrix_index {
            let target = target_offsets
                .get_mut(target_count)
                .ok_or(Error::EngineContract(
                    "raw HF dequant matrix has too many resource targets",
                ))?;
            *target = resource_layout.matrix_offsets[index];
            target_count += 1;
        }
    }
    if target_count == 0 {
        return Err(Error::EngineContract(
            "raw HF dequant matrix has no resource target",
        ));
    }
    Ok(RawMatrixParams {
        denominator: plan.denominator,
        width: plan.image.final_planes[0].width,
        height: plan.image.final_planes[0].height,
        target_count: u32::try_from(target_count)
            .map_err(|_| Error::backend("raw HF dequant target count exceeds WGSL u32"))?,
        plane_offsets: [
            plan.image.final_planes[0].word_offset,
            plan.image.final_planes[1].word_offset,
            plan.image.final_planes[2].word_offset,
            0,
        ],
        plane_strides: [
            plan.image.final_planes[0].row_stride_words,
            plan.image.final_planes[1].row_stride_words,
            plan.image.final_planes[2].row_stride_words,
            0,
        ],
        target_offsets,
    })
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(crate) const fn raw_matrix_value_error(code: u32) -> bool {
    code == ERROR_RAW_MATRIX_VALUE
}

pub(crate) const fn raw_matrix_status_ok(code: u32) -> bool {
    code == STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_shader_and_uniform_validate_semantically() {
        let module = naga::front::wgsl::parse_str(OVERLAY_SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(std::mem::size_of::<RawMatrixParams>(), 64);
        assert_eq!(std::mem::align_of::<RawMatrixParams>(), 16);
    }

    #[test]
    fn representative_targets_share_one_canonical_raster() {
        let layout = VarDctResourceLayout::new(1, 1, 1).unwrap();
        let expected_counts = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 1, 2, 1, 2, 1, 2];
        for (matrix, expected) in expected_counts.into_iter().enumerate() {
            let actual = TransformKind::ALL
                .into_iter()
                .filter(|&transform| hf_matrix_param_index(transform) == matrix)
                .count();
            assert_eq!(actual, expected);
            let offsets = TransformKind::ALL
                .into_iter()
                .enumerate()
                .filter(|(_, transform)| hf_matrix_param_index(*transform) == matrix)
                .map(|(index, _)| layout.matrix_offsets[index])
                .collect::<Vec<_>>();
            assert_eq!(offsets.len(), expected);
        }
    }
}
