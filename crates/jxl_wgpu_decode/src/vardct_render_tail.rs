// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Construction and execution of post-inverse VarDCT pipelines on [`RenderPlan`].
//!
//! Replaces standalone per-filter WGSL dispatch with unified graph execution in [`WgpuFrameSession`].

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

use jxl_gpu_protocol::{
    Border2d, ChromaAxis, EpfParams, EpfPass, Extent2d, GaborishParams, OutputColorEncoding,
    OutputDesc, OutputId, OutputLayout, OutputOrientation, PlaneDesc, PlaneId, PlaneRole,
    PrecisionContract, RenderNode, RenderOp, RenderPlan, ResourceData, ResourceId, ResourceUpdate,
    SampleType, Scale2d, UpsamplingFactor, XybParams,
};

use jxl_wgpu::ResidentEpfParameters;

use crate::vardct_engine::VarDctDecodeError;
use crate::vardct_frontend::VarDctColorTransform;

/// Configuration for building the post-inverse VarDCT [`RenderPlan`].
pub struct VarDctRenderTailDesc {
    pub image_extent: Extent2d,
    pub padded_extent: Extent2d,
    pub channel_shifts: [crate::vardct_frontend::VarDctChannelShift; 3],
    pub color_transform: VarDctColorTransform,
    pub gaborish_weights: Option<jxl_wgpu::ResidentGaborishWeights>,
    pub epf_passes: Option<Vec<ResidentEpfParameters>>,
    pub frame_upsampling: Option<UpsamplingFactor>,
    pub upsampling_weights: Option<Vec<f32>>,
    pub xyb_params: Option<XybParams>,
}

/// The compiled render tail plan and initial resources.
pub struct CompiledVarDctRenderTail {
    pub plan: Arc<RenderPlan>,
    pub resources: BTreeMap<ResourceId, ResourceUpdate>,
    pub input_planes: [PlaneId; 3],
    pub sigma_plane: Option<PlaneId>,
}

impl VarDctRenderTailDesc {
    /// Builds a verified [`RenderPlan`] that performs all post-inverse stages:
    /// Chroma Upsampling -> Adaptive LF -> Gaborish -> EPF -> Frame Upsampling -> Color Conversion -> Save.
    pub fn compile(self) -> Result<CompiledVarDctRenderTail, VarDctDecodeError> {
        let mut plan = RenderPlan::default();
        let mut resources = BTreeMap::new();
        let mut next_plane_id = 0_u32;
        let mut alloc_plane_id = || {
            let id = PlaneId(next_plane_id);
            next_plane_id += 1;
            id
        };

        // 1. Declare input imported resident planes (X/Y/B or Y/Cb/Cr)
        let input_planes = [
            alloc_plane_id(),
            alloc_plane_id(),
            alloc_plane_id(),
        ];

        for (ch, &plane_id) in input_planes.iter().enumerate() {
            let shift = self.channel_shifts[ch];
            let width = self.padded_extent.width >> shift.horizontal;
            let height = self.padded_extent.height >> shift.vertical;
            plan.planes.push(PlaneDesc {
                id: plane_id,
                extent: Extent2d::new(width, height),
                stride: width,
                sample_type: SampleType::F32,
                role: PlaneRole::ImportedResident,
            });
        }

        // Optional sigma plane for EPF
        let sigma_plane = if self.epf_passes.is_some() {
            let sigma_id = alloc_plane_id();
            let blocks_x = self.image_extent.width.div_ceil(8);
            let blocks_y = self.image_extent.height.div_ceil(8);
            plan.planes.push(PlaneDesc {
                id: sigma_id,
                extent: Extent2d::new(blocks_x, blocks_y),
                stride: blocks_x,
                sample_type: SampleType::F32,
                role: PlaneRole::ImportedResident,
            });
            Some(sigma_id)
        } else {
            None
        };

        // 2. Component Chroma Upsampling (if any channel is subsampled)
        let mut current_planes = input_planes;
        for ch in 0..3 {
            let shift = self.channel_shifts[ch];
            if shift.is_subsampled() {
                let in_plane = current_planes[ch];
                let in_extent = plan.planes.iter().find(|p| p.id == in_plane).unwrap().extent;

                let mut cur_in = in_plane;
                let mut cur_extent = in_extent;

                if shift.horizontal != 0 {
                    let out_extent = Extent2d::new(cur_extent.width * 2, cur_extent.height);
                    let out_plane = alloc_plane_id();
                    plan.planes.push(PlaneDesc {
                        id: out_plane,
                        extent: out_extent,
                        stride: out_extent.width,
                        sample_type: SampleType::F32,
                        role: PlaneRole::Intermediate,
                    });
                    let node = RenderNode {
                        name: format!("chroma_upsample_h_ch{ch}").into(),
                        op: RenderOp::ChromaUpsample {
                            axis: ChromaAxis::Horizontal,
                        },
                        inputs: vec![cur_in],
                        outputs: vec![out_plane],
                        resources: Vec::new(),
                        scale: Scale2d::new(2, 1),
                        border: Border2d::symmetric(1, 0),
                        precision: PrecisionContract::default(),
                    };
                    plan.nodes.push(node);
                    cur_in = out_plane;
                    cur_extent = out_extent;
                }

                if shift.vertical != 0 {
                    let out_extent = Extent2d::new(cur_extent.width, cur_extent.height * 2);
                    let out_plane = alloc_plane_id();
                    plan.planes.push(PlaneDesc {
                        id: out_plane,
                        extent: out_extent,
                        stride: out_extent.width,
                        sample_type: SampleType::F32,
                        role: PlaneRole::Intermediate,
                    });
                    let node = RenderNode {
                        name: format!("chroma_upsample_v_ch{ch}").into(),
                        op: RenderOp::ChromaUpsample {
                            axis: ChromaAxis::Vertical,
                        },
                        inputs: vec![cur_in],
                        outputs: vec![out_plane],
                        resources: Vec::new(),
                        scale: Scale2d::new(1, 2),
                        border: Border2d::symmetric(0, 1),
                        precision: PrecisionContract::default(),
                    };
                    plan.nodes.push(node);
                    cur_in = out_plane;
                }

                current_planes[ch] = cur_in;
            }
        }

        // 3. Gaborish
        if let Some(weights) = self.gaborish_weights {
            let gaborish_outs = [alloc_plane_id(), alloc_plane_id(), alloc_plane_id()];
            let full_extent = Extent2d::new(self.padded_extent.width, self.padded_extent.height);
            for (ch, &out_plane) in gaborish_outs.iter().enumerate() {
                plan.planes.push(PlaneDesc {
                    id: out_plane,
                    extent: full_extent,
                    stride: full_extent.width,
                    sample_type: SampleType::F32,
                    role: PlaneRole::Intermediate,
                });
                let weight_arr = match ch {
                    0 => weights.x,
                    1 => weights.y,
                    _ => weights.b,
                };
                let node = RenderNode {
                    name: format!("gaborish_ch{ch}").into(),
                    op: RenderOp::Gaborish(GaborishParams {
                        channel: ch as u16,
                        weight0: 1.0,
                        weight1: weight_arr[0],
                        weight2: weight_arr[1],
                    }),
                    inputs: vec![current_planes[ch]],
                    outputs: vec![out_plane],
                    resources: Vec::new(),
                    scale: Scale2d::IDENTITY,
                    border: Border2d::symmetric(1, 1),
                    precision: PrecisionContract::default(),
                };
                plan.nodes.push(node);
            }
            current_planes = gaborish_outs;
        }

        // 4. EPF (Pass 0, 1, 2)
        if let Some(passes) = &self.epf_passes {
            let sigma = sigma_plane.expect("sigma plane initialized for EPF");
            let full_extent = Extent2d::new(self.padded_extent.width, self.padded_extent.height);
            for (pass_idx, pass_params) in passes.iter().enumerate() {
                let pass_outs = [alloc_plane_id(), alloc_plane_id(), alloc_plane_id()];
                for (_ch, &out_plane) in pass_outs.iter().enumerate() {
                    plan.planes.push(PlaneDesc {
                        id: out_plane,
                        extent: full_extent,
                        stride: full_extent.width,
                        sample_type: SampleType::F32,
                        role: PlaneRole::Intermediate,
                    });
                }
                for ch in 0..3 {
                    let border = match pass_params.pass {
                        EpfPass::Pass0 | EpfPass::Pass1 => Border2d::symmetric(1, 1),
                        EpfPass::Pass2 => Border2d::symmetric(2, 2),
                    };
                    let node = RenderNode {
                        name: format!("epf_pass{pass_idx}_ch{ch}").into(),
                        op: RenderOp::Epf(EpfParams {
                            pass: pass_params.pass,
                            sigma_scale: pass_params.sigma_scale,
                            border_sad_mul: pass_params.border_sad_mul,
                            channel_scale: pass_params.channel_scale,
                            sigma_resource: None,
                            sigma_plane: Some(sigma),
                        }),
                        inputs: current_planes.to_vec(),
                        outputs: vec![pass_outs[ch]],
                        resources: Vec::new(),
                        scale: Scale2d::IDENTITY,
                        border,
                        precision: PrecisionContract::default(),
                    };
                    plan.nodes.push(node);
                }
                current_planes = pass_outs;
            }
        }

        // 5. Frame Upsampling (if factor > 1)
        if let Some(factor) = self.frame_upsampling {
            let scale_u32 = factor.as_u32();
            let up_extent = Extent2d::new(
                self.image_extent.width * scale_u32,
                self.image_extent.height * scale_u32,
            );
            let weights_id = ResourceId(999);
            if let Some(w) = self.upsampling_weights {
                resources.insert(
                    weights_id,
                    ResourceUpdate {
                        id: weights_id,
                        revision: 1,
                        data: ResourceData::F32(w),
                    },
                );
            }

            let up_outs = [alloc_plane_id(), alloc_plane_id(), alloc_plane_id()];
            for (ch, &out_plane) in up_outs.iter().enumerate() {
                plan.planes.push(PlaneDesc {
                    id: out_plane,
                    extent: up_extent,
                    stride: up_extent.width,
                    sample_type: SampleType::F32,
                    role: PlaneRole::Intermediate,
                });
                plan.add_upsample_node(
                    format!("frame_upsample_ch{ch}"),
                    factor,
                    weights_id,
                    current_planes[ch],
                    out_plane,
                );
            }
            current_planes = up_outs;
        }

        // 6. Color conversion: XYB -> RGB or YCbCr -> RGB
        let final_extent = plan.planes.iter().find(|p| p.id == current_planes[0]).unwrap().extent;
        let rgb_outs = [alloc_plane_id(), alloc_plane_id(), alloc_plane_id()];
        for (_ch, &out_plane) in rgb_outs.iter().enumerate() {
            plan.planes.push(PlaneDesc {
                id: out_plane,
                extent: final_extent,
                stride: final_extent.width,
                sample_type: SampleType::F32,
                role: PlaneRole::Intermediate,
            });
        }

        match self.color_transform {
            VarDctColorTransform::Xyb => {
                let xyb = self.xyb_params.unwrap_or_default();
                plan.nodes.push(RenderNode {
                    name: "xyb_to_rgb".into(),
                    op: RenderOp::XybToRgb(xyb),
                    inputs: current_planes.to_vec(),
                    outputs: rgb_outs.to_vec(),
                    resources: Vec::new(),
                    scale: Scale2d::IDENTITY,
                    border: Border2d::ZERO,
                    precision: PrecisionContract::default(),
                });
            }
            VarDctColorTransform::Ycbcr => {
                plan.nodes.push(RenderNode {
                    name: "ycbcr_to_rgb".into(),
                    op: RenderOp::YcbcrToRgb,
                    inputs: current_planes.to_vec(),
                    outputs: rgb_outs.to_vec(),
                    resources: Vec::new(),
                    scale: Scale2d::IDENTITY,
                    border: Border2d::ZERO,
                    precision: PrecisionContract::default(),
                });
            }
        }

        // 7. Output Save
        let output_id = OutputId(0);
        plan.outputs.push(OutputDesc {
            id: output_id,
            extent: final_extent,
            sample_type: SampleType::F32,
            channels: 3,
            layout: OutputLayout::Planar,
            color_encoding: OutputColorEncoding::NonColor,
        });

        plan.nodes.push(RenderNode {
            name: "save_rgb".into(),
            op: RenderOp::Save(jxl_gpu_protocol::SaveParams {
                output: output_id,
                sample_type: SampleType::F32,
                channels: rgb_outs.to_vec(),
                layout: OutputLayout::Planar,
                orientation: OutputOrientation::Identity,
            }),
            inputs: rgb_outs.to_vec(),
            outputs: Vec::new(),
            resources: Vec::new(),
            scale: Scale2d::IDENTITY,
            border: Border2d::ZERO,
            precision: PrecisionContract::Exact,
        });

        plan.validate()?;

        Ok(CompiledVarDctRenderTail {
            plan: Arc::new(plan),
            resources,
            input_planes,
            sigma_plane,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vardct_frontend::VarDctChannelShift;
    use jxl_wgpu::{ResidentEpfParameters, ResidentGaborishWeights};

    #[test]
    fn render_tail_compiles_valid_render_plan_for_all_combinations() {
        let image_extent = Extent2d::new(16, 16);
        let padded_extent = Extent2d::new(16, 16);

        for color_transform in [VarDctColorTransform::Xyb, VarDctColorTransform::Ycbcr] {
            for gaborish in [None, Some(ResidentGaborishWeights::DEFAULT)] {
                for epf in [
                    None,
                    Some(vec![
                        ResidentEpfParameters {
                            pass: EpfPass::Pass0,
                            sigma_scale: 0.9,
                            border_sad_mul: 0.66,
                            channel_scale: [40.0, 5.0, 3.5],
                        },
                        ResidentEpfParameters {
                            pass: EpfPass::Pass1,
                            sigma_scale: 1.0,
                            border_sad_mul: 0.66,
                            channel_scale: [40.0, 5.0, 3.5],
                        },
                    ]),
                ] {
                    for frame_upsampling in [None, Some(UpsamplingFactor::X2)] {
                        let upsampling_weights = frame_upsampling.map(|f| {
                            let n = (f.as_u32() * f.as_u32() * 25) as usize;
                            vec![0.04; n]
                        });
                        let desc = VarDctRenderTailDesc {
                            image_extent,
                            padded_extent,
                            channel_shifts: [VarDctChannelShift::default(); 3],
                            color_transform,
                            gaborish_weights: gaborish,
                            epf_passes: epf.clone(),
                            frame_upsampling,
                            upsampling_weights,
                            xyb_params: Some(XybParams::default()),
                        };
                        let compiled = desc.compile().expect("compilation must succeed");
                        assert!(compiled.plan.validate().is_ok());
                    }
                }
            }
        }
    }

    #[test]
    fn render_tail_handles_chroma_subsampling() {
        let image_extent = Extent2d::new(16, 16);
        let padded_extent = Extent2d::new(16, 16);
        // 4:2:0 subsampling: Y is full resolution, X and B are half horizontal & vertical
        let half_shift = VarDctChannelShift {
            horizontal: 1,
            vertical: 1,
        };
        let channel_shifts = [half_shift, VarDctChannelShift::default(), half_shift];

        let desc = VarDctRenderTailDesc {
            image_extent,
            padded_extent,
            channel_shifts,
            color_transform: VarDctColorTransform::Ycbcr,
            gaborish_weights: Some(ResidentGaborishWeights::DEFAULT),
            epf_passes: None,
            frame_upsampling: None,
            upsampling_weights: None,
            xyb_params: None,
        };
        let compiled = desc.compile().expect("compilation with chroma subsampling must succeed");
        assert!(compiled.plan.validate().is_ok());

        // Check that ChromaUpsample nodes were generated for X (ch0) and B (ch2)
        let chroma_nodes: Vec<_> = compiled
            .plan
            .nodes
            .iter()
            .filter(|n| matches!(n.op, RenderOp::ChromaUpsample { .. }))
            .collect();
        // 2 channels * 2 axes = 4 nodes
        assert_eq!(chroma_nodes.len(), 4);
    }
}
