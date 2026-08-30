// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::BTreeSet;

use jxl_gpu_protocol::{AcceleratorCapabilities, RenderOpKind};

pub(crate) fn capabilities(
    device: &wgpu::Device,
    info: &wgpu::AdapterInfo,
    minimum_frame_pixels: Option<u64>,
) -> AcceleratorCapabilities {
    let features = device.features();
    let limits = device.limits();
    AcceleratorCapabilities {
        name: format!("{} ({:?})", info.name, info.backend),
        supported_ops: [
            RenderOpKind::Copy,
            RenderOpKind::ModularToF32,
            RenderOpKind::ChromaUpsample,
            RenderOpKind::Gaborish,
            RenderOpKind::Epf,
            RenderOpKind::Upsample,
            RenderOpKind::VarDct,
            RenderOpKind::YcbcrToRgb,
            RenderOpKind::PremultiplyAlpha,
            RenderOpKind::Convert,
            RenderOpKind::Save,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>(),
        minimum_frame_pixels,
        max_buffer_bytes: limits.max_buffer_size,
        max_workgroup_storage_bytes: limits.max_compute_workgroup_storage_size,
        max_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        supports_timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
        supports_f16: features.contains(wgpu::Features::SHADER_F16),
    }
}
