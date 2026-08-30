// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::sync::Arc;

use jxl_gpu_formats::convert_rgb_f32;
use jxl_gpu_protocol::{
    Border2d, Extent2d, FallbackGranularity, FrameSessionDesc, GroupId, GroupPayload, HostPlane,
    MemoryMode, OutputDesc, OutputId, OutputLayout, PlaneData, PlaneDesc, PlaneId, PlaneRole,
    PrecisionContract, PrecisionPolicy, RenderIntent, RenderNode, RenderOp, RenderPlan, SaveParams,
    Scale2d,
};
use jxl_wgpu::{
    ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, Error, ImageOutputRequest,
    Packed422Order, PixelFormat, RgbChannelOrder, WgpuAccelerator, WgpuAcceleratorConfig,
};

fn accelerator() -> Option<WgpuAccelerator> {
    match pollster::block_on(WgpuAccelerator::request_default(WgpuAcceleratorConfig {
        enable_timestamps: false,
        ..WgpuAcceleratorConfig::default()
    })) {
        Ok(accelerator) => Some(accelerator),
        Err(Error::NoAdapter) => {
            eprintln!("skipping GPU test: no wgpu adapter is available");
            None
        }
        Err(error) => panic!("failed to initialize GPU test device: {error}"),
    }
}

fn frame_desc(extent: Extent2d) -> FrameSessionDesc {
    FrameSessionDesc {
        frame_extent: extent,
        group_extent: extent,
        group_count: 1,
        precision: PrecisionPolicy::F32Only,
        memory_mode: MemoryMode::Resident,
        max_resident_bytes: 16 * 1024 * 1024,
        max_scratch_bytes: 16 * 1024 * 1024,
        fallback: FallbackGranularity::WholeFrame,
    }
}

fn plan(extent: Extent2d) -> Arc<RenderPlan> {
    let output = OutputId(0);
    let channels = vec![PlaneId(0), PlaneId(1), PlaneId(2)];
    Arc::new(RenderPlan {
        planes: channels
            .iter()
            .map(|&id| PlaneDesc {
                id,
                extent,
                stride: extent.width,
                sample_type: jxl_gpu_protocol::SampleType::F32,
                role: PlaneRole::Source,
            })
            .collect(),
        nodes: vec![RenderNode {
            name: "generic image save".into(),
            op: RenderOp::Save(SaveParams {
                output,
                sample_type: jxl_gpu_protocol::SampleType::F32,
                channels: channels.clone(),
                layout: OutputLayout::Interleaved,
                orientation: jxl_gpu_protocol::OutputOrientation::Identity,
            }),
            inputs: channels,
            outputs: Vec::new(),
            resources: Vec::new(),
            scale: Scale2d::IDENTITY,
            border: Border2d::default(),
            precision: PrecisionContract::default(),
        }],
        outputs: vec![OutputDesc {
            id: output,
            extent,
            sample_type: jxl_gpu_protocol::SampleType::F32,
            channels: 3,
            layout: OutputLayout::Interleaved,
        }],
    })
}

fn rgb_planes(extent: Extent2d) -> [Vec<f32>; 3] {
    let mut planes = std::array::from_fn(|_| Vec::with_capacity(extent.area().unwrap()));
    for y in 0..extent.height {
        for x in 0..extent.width {
            planes[0].push((0.071 + 0.137 * x as f32 + 0.053 * y as f32).fract());
            planes[1].push((0.193 + 0.079 * x as f32 + 0.149 * y as f32).fract());
            planes[2].push((0.317 + 0.113 * x as f32 + 0.097 * y as f32).fract());
        }
    }
    planes
}

fn enqueue(session: &mut jxl_wgpu::WgpuFrameSession, extent: Extent2d, channels: &[Vec<f32>; 3]) {
    session
        .enqueue(GroupPayload {
            group: GroupId(0),
            revision: 0,
            complete: true,
            planes: channels
                .iter()
                .enumerate()
                .map(|(index, data)| HostPlane {
                    id: PlaneId(index as u32),
                    extent,
                    stride: extent.width,
                    origin: (0, 0),
                    data: PlaneData::F32(data.clone()),
                })
                .collect(),
            vardct: None,
        })
        .expect("enqueue RGB planes");
}

fn color(range: ColorRange, location: ChromaLocation2d) -> ColorSpecification {
    ColorSpecification::Defined(ColorSpec::bt709(range, location))
}

#[test]
fn canonical_pitch_linear_formats_match_scalar_oracle() {
    let Some(accelerator) = accelerator() else {
        return;
    };
    let extent = Extent2d::new(5, 3);
    let channels = rgb_planes(extent);
    let limited_center = color(ColorRange::Limited, ChromaLocation2d::CENTER);
    let full_center = color(ColorRange::Full, ChromaLocation2d::CENTER);
    let formats = vec![
        PixelFormat::luma(8, limited_center),
        PixelFormat::luma(16, limited_center),
        PixelFormat::i444(8, 8, full_center).unwrap(),
        PixelFormat::i422(8, 8, limited_center).unwrap(),
        PixelFormat::i420(8, 8, limited_center).unwrap(),
        PixelFormat::nv12(limited_center),
        PixelFormat::nv21(limited_center),
        PixelFormat::nv24(full_center),
        PixelFormat::nv42(full_center),
        PixelFormat::p010(limited_center),
        PixelFormat::p012(limited_center),
        PixelFormat::p016(limited_center),
        PixelFormat::i420(12, 16, limited_center).unwrap(),
        PixelFormat::packed_yuv4228(Packed422Order::Yuyv, limited_center),
        PixelFormat::packed_yuv4228(Packed422Order::Uyvy, limited_center),
        PixelFormat::rgb8(RgbChannelOrder::Rgb, false, full_center),
        PixelFormat::rgb8(RgbChannelOrder::Bgr, false, full_center),
        PixelFormat::rgb8(RgbChannelOrder::Rgba, false, full_center),
        PixelFormat::rgb8(RgbChannelOrder::Bgra, false, full_center),
        PixelFormat::rgb8(RgbChannelOrder::Rgb, true, full_center),
        PixelFormat::rgb8(RgbChannelOrder::Bgra, true, full_center),
    ];

    for format in formats {
        let mut session = accelerator
            .create_session(&frame_desc(extent), plan(extent))
            .expect("create generic image session");
        enqueue(&mut session, extent, &channels);
        let token = session
            .submit_image(RenderIntent::Final, ImageOutputRequest::new(format.clone()))
            .expect("submit generic image readback");
        let frame = session.wait_image(token).expect("wait image readback");
        let expected = convert_rgb_f32([&channels[0], &channels[1], &channels[2]], extent, &format)
            .expect("scalar image oracle");
        assert_eq!(frame.outputs.len(), 1);
        assert_eq!(frame.outputs[0].layout, expected.layout);
        assert_eq!(frame.outputs[0].bytes, expected.bytes, "format {format:?}");
    }
}

#[test]
fn nv12_gpu_output_handles_degenerate_odd_edges_without_readback() {
    let Some(accelerator) = accelerator() else {
        return;
    };
    let format = PixelFormat::nv12(color(ColorRange::Limited, ChromaLocation2d::CENTER));
    for extent in [
        Extent2d::new(1, 1),
        Extent2d::new(1, 3),
        Extent2d::new(3, 1),
    ] {
        let channels = rgb_planes(extent);
        let mut session = accelerator
            .create_session(&frame_desc(extent), plan(extent))
            .expect("create zero-copy NV12 session");
        enqueue(&mut session, extent, &channels);
        let frame = session
            .submit_gpu_image(RenderIntent::Final, ImageOutputRequest::new(format.clone()))
            .expect("submit zero-copy NV12");
        let expected = jxl_wgpu::ImageLayout::packed(extent, format.clone()).unwrap();
        assert_eq!(frame.outputs.len(), 1);
        assert_eq!(frame.outputs[0].layout, expected);
        assert!(frame.outputs[0].buffer.size() >= expected.logical_size);
        assert_eq!(
            session
                .last_submission_stats()
                .map(|stats| stats.direct_readback),
            Some(false)
        );
        session.wait_gpu(frame.token).expect("wait zero-copy NV12");
    }
}
