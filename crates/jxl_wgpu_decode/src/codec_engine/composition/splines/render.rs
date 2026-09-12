use jxl_wgpu::ResidentF32Plane;
use wgpu::util::DeviceExt;

use super::Cache;
use crate::{Error, Result};

const BATCH_REFERENCES: u32 = 256;
const UNIFORM_BYTES: u64 = 48;

impl Cache {
    pub(in super::super) fn scratch_bytes(&self) -> u64 {
        u64::from(self.max_population.div_ceil(BATCH_REFERENCES)) * UNIFORM_BYTES
    }

    pub(in super::super) fn record(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        planes: [ResidentF32Plane<'_>; 3],
    ) -> Result<Vec<wgpu::Buffer>> {
        let groups = [
            self.extent.width.div_ceil(8),
            self.extent.height.div_ceil(8),
        ];
        if groups
            .into_iter()
            .any(|n| n > device.limits().max_compute_workgroups_per_dimension)
        {
            return Err(Error::EngineContract(
                "spline raster dispatch exceeds device limits",
            ));
        }
        for plane in planes {
            if plane.width != self.extent.width
                || plane.height != self.extent.height
                || plane.stride < plane.width
                || (u64::from(plane.height - 1) * u64::from(plane.stride) + u64::from(plane.width))
                    * 4
                    > plane.storage.size.get()
            {
                return Err(Error::EngineContract("spline component plane geometry"));
            }
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JPEG XL ordered spline raster"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("JPEG XL ordered spline raster"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        (0..self.max_population.div_ceil(BATCH_REFERENCES))
            .map(|batch| {
                let params = [
                    self.extent.width,
                    self.extent.height,
                    self.tile_columns,
                    self.tile_count,
                    planes[0].stride,
                    planes[1].stride,
                    planes[2].stride,
                    0,
                    batch * BATCH_REFERENCES,
                    (batch + 1) * BATCH_REFERENCES,
                    32,
                    0,
                ];
                let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("JPEG XL spline raster batch"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let mut entries: Vec<_> = planes
                    .into_iter()
                    .enumerate()
                    .map(|(index, plane)| wgpu::BindGroupEntry {
                        binding: index as u32,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: plane.storage.buffer,
                            offset: plane.storage.offset,
                            size: Some(plane.storage.size),
                        }),
                    })
                    .collect();
                entries.extend(
                    [
                        self.records.as_wgpu_buffer(),
                        self.references.as_wgpu_buffer(),
                        self.tiles.as_wgpu_buffer(),
                        &uniform,
                    ]
                    .into_iter()
                    .enumerate()
                    .map(|(index, buffer)| wgpu::BindGroupEntry {
                        binding: 3 + index as u32,
                        resource: buffer.as_entire_binding(),
                    }),
                );
                let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("JPEG XL spline raster"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                });
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bindings, &[]);
                    pass.dispatch_workgroups(groups[0], groups[1], 1);
                }
                Ok(uniform)
            })
            .collect()
    }
}
