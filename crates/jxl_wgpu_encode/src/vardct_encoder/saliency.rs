//! GPU local-contrast heuristic. Only bounded per-group statistics leave the device.
use jxl_wgpu::KernelVariant;

use super::dispatch::shader_source;
use super::types::VarDctFrameLayout;
use crate::{BackendError, EncodeError};

pub(super) const READY: u32 = 0x53414c59;
pub(super) const WORKGROUP_BYTES: u32 = 256 * 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Record {
    pub(super) status: u32,
    pub(super) group: u32,
    pub(super) edges: u32,
    pub(super) contrast: u32,
}

const _: () = assert!(std::mem::size_of::<Record>() == 16);

pub(super) fn validate(records: &[Record], frame: VarDctFrameLayout) -> Result<(), BackendError> {
    let count = frame
        .ac_group_count()
        .map_err(|_| BackendError::InvalidArtifact("invalid saliency grid"))?;
    if records.len() != count as usize {
        return Err(BackendError::InvalidArtifact(
            "saliency record count differs from grid",
        ));
    }
    for (id, record) in records.iter().enumerate() {
        let id = id as u32;
        let (left, top) = (id % frame.ac_groups_x * 256, id / frame.ac_groups_x * 256);
        let (width, height) = ((frame.width - left).min(256), (frame.height - top).min(256));
        let edges =
            (width - u32::from(left == 0)) * height + (height - u32::from(top == 0)) * width;
        if record.status != READY
            || record.group != id
            || record.edges != edges
            || record.contrast > edges * 765
        {
            return Err(BackendError::InvalidArtifact(
                "invalid or incomplete saliency record",
            ));
        }
    }
    Ok(())
}

pub(super) struct Pipeline(wgpu::ComputePipeline);

impl Pipeline {
    pub(super) fn new(device: &wgpu::Device, variant: KernelVariant) -> Result<Self, EncodeError> {
        variant.validate_for("vardct_encode_saliency", &device.limits(), WORKGROUP_BYTES)?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VarDCT group local contrast"),
            source: wgpu::ShaderSource::Wgsl(shader_source(include_str!("saliency.wgsl")).into()),
        });
        Ok(Self(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("VarDCT group local contrast"),
                layout: None,
                module: &module,
                entry_point: Some("group_saliency"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("wg_x", f64::from(variant.workgroup_size().0))],
                    ..Default::default()
                },
                cache: None,
            },
        )))
    }

    pub(super) fn encode(
        &self,
        device: &wgpu::Device,
        commands: &mut wgpu::CommandEncoder,
        sources: [wgpu::BindGroupEntry<'_>; 4],
        parameters: &wgpu::Buffer,
        artifact: &wgpu::Buffer,
        frame: VarDctFrameLayout,
    ) {
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VarDCT saliency bindings"),
            layout: &self.0.get_bind_group_layout(0),
            entries: &[
                sources[0].clone(),
                sources[1].clone(),
                sources[2].clone(),
                sources[3].clone(),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parameters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: artifact.as_entire_binding(),
                },
            ],
        });
        let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("VarDCT saliency reduction"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(frame.ac_groups_x, frame.ac_groups_y, 1);
    }
}
