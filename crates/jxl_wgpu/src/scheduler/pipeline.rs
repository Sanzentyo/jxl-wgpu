// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::sync::Arc;

use bytemuck::Pod;
use wgpu::util::DeviceExt;

use crate::autotune::{KernelPolicy, KernelVariant};
use crate::buffer_pool::BufferPool;
use crate::pipeline_cache::{PipelineCache, PipelineKey};
use crate::{Error, Result};

pub(in crate::scheduler) struct PipelineFactory<'a> {
    pub(in crate::scheduler) device: &'a wgpu::Device,
    pub(in crate::scheduler) cache: &'a PipelineCache,
    pub(in crate::scheduler) buffers: &'a Arc<BufferPool>,
    pub(in crate::scheduler) kernel_policy: &'a KernelPolicy,
    pub(in crate::scheduler) variant: KernelVariant,
}

pub(in crate::scheduler) fn create_uniform<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    value: &T,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

pub(in crate::scheduler) fn create_pipeline(
    factory: &PipelineFactory<'_>,
    label: &str,
    descriptor: wgpu::ShaderModuleDescriptor<'static>,
) -> std::sync::Arc<wgpu::ComputePipeline> {
    create_pipeline_with_variant(factory, label, descriptor, "main", 0, factory.variant)
}

pub(in crate::scheduler) fn create_pipeline_with_variant(
    factory: &PipelineFactory<'_>,
    label: &str,
    descriptor: wgpu::ShaderModuleDescriptor<'static>,
    entry_point: &'static str,
    layout_hash: u64,
    variant: KernelVariant,
) -> std::sync::Arc<wgpu::ComputePipeline> {
    create_pipeline_entry(
        factory,
        label,
        descriptor,
        entry_point,
        layout_hash,
        variant,
    )
}

pub(in crate::scheduler) fn create_pipeline_entry(
    factory: &PipelineFactory<'_>,
    label: &str,
    descriptor: wgpu::ShaderModuleDescriptor<'static>,
    entry_point: &'static str,
    layout_hash: u64,
    variant: KernelVariant,
) -> std::sync::Arc<wgpu::ComputePipeline> {
    let key = PipelineKey::new(label, entry_point, variant, layout_hash);
    if let Some(pipeline) = factory.cache.get(&key) {
        return pipeline;
    }
    match factory.cache.get_or_insert_with(key, || {
        let module = factory.device.create_shader_module(descriptor);
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        let constants = [
            ("wg_x", f64::from(workgroup_x)),
            ("wg_y", f64::from(workgroup_y)),
        ];
        Ok::<_, std::convert::Infallible>(factory.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            },
        ))
    }) {
        Ok(pipeline) => pipeline,
        Err(never) => match never {},
    }
}

pub(in crate::scheduler) fn record_dispatch(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    resources: &[wgpu::BindingResource<'_>],
    width: u32,
    height: u32,
    variant: KernelVariant,
) {
    let layout = pipeline.get_bind_group_layout(0);
    let entries = resources
        .iter()
        .enumerate()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: resource.clone(),
        })
        .collect::<Vec<_>>();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jxl-wgpu dispatch bindings"),
        layout: &layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("jxl-wgpu dispatch"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let (workgroup_x, workgroup_y) = variant.workgroup_size();
    pass.dispatch_workgroups(width.div_ceil(workgroup_x), height.div_ceil(workgroup_y), 1);
}

pub(in crate::scheduler) fn record_linear_dispatch(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    resources: &[wgpu::BindingResource<'_>],
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let layout = pipeline.get_bind_group_layout(0);
    let entries = resources
        .iter()
        .enumerate()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: resource.clone(),
        })
        .collect::<Vec<_>>();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jxl-wgpu linear dispatch bindings"),
        layout: &layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("jxl-wgpu linear dispatch"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

pub(in crate::scheduler) fn linear_dispatch_shape(
    device: &wgpu::Device,
    word_count: u64,
    variant: KernelVariant,
) -> Result<(u32, u32, u32)> {
    let (workgroup_x, workgroup_y) = variant.workgroup_size();
    let limit = device.limits().max_compute_workgroups_per_dimension;
    let required_x = word_count.div_ceil(u64::from(workgroup_x));
    let workgroups_x =
        u32::try_from(required_x.min(u64::from(limit))).map_err(|_| Error::BufferSizeOverflow)?;
    let dispatch_width = workgroups_x
        .checked_mul(workgroup_x)
        .ok_or(Error::BufferSizeOverflow)?;
    let required_y = word_count.div_ceil(u64::from(dispatch_width));
    let workgroups_y = u32::try_from(required_y.div_ceil(u64::from(workgroup_y)))
        .map_err(|_| Error::BufferSizeOverflow)?;
    if workgroups_y > limit {
        return Err(Error::ResourceLimit(format!(
            "generic image output needs a {workgroups_x}x{workgroups_y} dispatch, exceeding the device limit {limit}"
        )));
    }
    Ok((workgroups_x, workgroups_y, dispatch_width))
}
