// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::sync::Arc;

use jxl_gpu_formats::convert_rgb_f32;
use jxl_gpu_protocol::{
    Border2d, Extent2d, FrameSessionDesc, GroupId, GroupPayload, HostPlane, MemoryMode, OutputDesc,
    OutputId, OutputLayout, PlaneData, PlaneDesc, PlaneId, PlaneRole, PrecisionContract,
    PrecisionPolicy, RenderIntent, RenderNode, RenderOp, RenderPlan, SaveParams, Scale2d,
    TransferFunction as SourceTransferFunction,
};
use jxl_wgpu::{
    ChromaLocation2d, ColorRange, ColorSpace, ColorSpec, ColorSpecification, Error,
    ImageOutputRequest, OutputColorEncoding, Packed422Order, PixelFormat, RgbChannelOrder,
    RgbColorEncoding, RgbPrimaries, TransferFunction, WgpuBackend, WgpuBackendConfig,
    YcbcrEncoding,
};

fn backend() -> Option<WgpuBackend> {
    match pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
        enable_timestamps: false,
        ..WgpuBackendConfig::default()
    })) {
        Ok(backend) => Some(backend),
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
    }
}

fn plan(extent: Extent2d, source_encoding: RgbColorEncoding) -> Arc<RenderPlan> {
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
            color_encoding: OutputColorEncoding::Rgb(source_encoding),
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

fn rgb_color(space: ColorSpace, transfer: TransferFunction) -> ColorSpecification {
    ColorSpecification::Defined(ColorSpec {
        space,
        encoding: YcbcrEncoding::Undefined,
        transfer,
        range: ColorRange::Full,
        chroma_location: ChromaLocation2d::BOTH,
    })
}

fn submit_rgb8(
    backend: &WgpuBackend,
    source_encoding: RgbColorEncoding,
    request_encoding: RgbColorEncoding,
    target_color: ColorSpecification,
    samples: [f32; 3],
) -> Result<Vec<u8>, Error> {
    let extent = Extent2d::new(1, 1);
    let channels = [vec![samples[0]], vec![samples[1]], vec![samples[2]]];
    let mut session = backend.create_session(&frame_desc(extent), plan(extent, source_encoding))?;
    enqueue(&mut session, extent, &channels);
    let token = session.submit_image(
        RenderIntent::Final,
        ImageOutputRequest::new(
            request_encoding,
            PixelFormat::rgb8(RgbChannelOrder::Rgb, false, target_color),
        ),
    )?;
    Ok(session.wait_image(token)?.outputs.remove(0).bytes)
}

fn srgb_from_linear(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn bt709_to_linear(value: f32) -> f32 {
    if value < 0.081 {
        value / 4.5
    } else {
        ((value + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}

fn quantize8(value: f32) -> u8 {
    (255.0 * value).round().clamp(0.0, 255.0) as u8
}

#[test]
fn canonical_pitch_linear_formats_match_scalar_oracle() {
    let Some(backend) = backend() else {
        return;
    };
    let extent = Extent2d::new(5, 3);
    let channels = rgb_planes(extent);
    let limited_center = color(ColorRange::Limited, ChromaLocation2d::CENTER);
    let full_center = color(ColorRange::Full, ChromaLocation2d::CENTER);
    let bt601_limited = ColorSpecification::Defined(ColorSpec::bt601(
        ColorRange::Limited,
        ChromaLocation2d::CENTER,
    ));
    let formats = vec![
        PixelFormat::luma(8, limited_center),
        PixelFormat::luma(16, limited_center),
        PixelFormat::i444(8, 8, full_center).unwrap(),
        PixelFormat::i422(8, 8, limited_center).unwrap(),
        PixelFormat::i420(8, 8, limited_center).unwrap(),
        PixelFormat::nv12(limited_center),
        PixelFormat::nv12(bt601_limited),
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
        let mut session = backend
            .create_session(&frame_desc(extent), plan(extent, RgbColorEncoding::BT709))
            .expect("create generic image session");
        enqueue(&mut session, extent, &channels);
        let token = session
            .submit_image(
                RenderIntent::Final,
                ImageOutputRequest::new(RgbColorEncoding::BT709, format.clone()),
            )
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
    let Some(backend) = backend() else {
        return;
    };
    let format = PixelFormat::nv12(color(ColorRange::Limited, ChromaLocation2d::CENTER));
    for extent in [
        Extent2d::new(1, 1),
        Extent2d::new(1, 3),
        Extent2d::new(3, 1),
    ] {
        let channels = rgb_planes(extent);
        let mut session = backend
            .create_session(&frame_desc(extent), plan(extent, RgbColorEncoding::BT709))
            .expect("create zero-copy NV12 session");
        enqueue(&mut session, extent, &channels);
        let frame = session
            .submit_gpu_image(
                RenderIntent::Final,
                ImageOutputRequest::new(RgbColorEncoding::BT709, format.clone()),
            )
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

#[test]
fn gpu_output_converts_declared_transfer_functions() {
    let Some(backend) = backend() else {
        return;
    };
    let samples = [0.18, 0.5, 0.82];
    let bt709_as_linear = submit_rgb8(
        &backend,
        RgbColorEncoding::BT709,
        RgbColorEncoding::BT709,
        rgb_color(ColorSpace::Bt709, TransferFunction::Linear),
        samples,
    )
    .expect("BT.709 to linear RGB8");
    let expected_linear = samples.map(|value| quantize8(bt709_to_linear(value)));
    assert!(
        bt709_as_linear
            .iter()
            .zip(expected_linear)
            .all(|(&actual, expected)| actual.abs_diff(expected) <= 1),
        "GPU linear bytes {bt709_as_linear:?}, expected {expected_linear:?}"
    );

    let srgb_as_linear = submit_rgb8(
        &backend,
        RgbColorEncoding::SRGB_BT709,
        RgbColorEncoding::SRGB_BT709,
        rgb_color(ColorSpace::Bt709, TransferFunction::Linear),
        samples,
    )
    .expect("sRGB to linear RGB8");
    let expected_srgb_as_linear = samples.map(|value| quantize8(srgb_to_linear(value)));
    assert!(
        srgb_as_linear
            .iter()
            .zip(expected_srgb_as_linear)
            .all(|(&actual, expected)| actual.abs_diff(expected) <= 1),
        "GPU sRGB-decoded bytes {srgb_as_linear:?}, expected {expected_srgb_as_linear:?}"
    );

    let identity_linear = submit_rgb8(
        &backend,
        RgbColorEncoding::LINEAR_BT709,
        RgbColorEncoding::LINEAR_BT709,
        rgb_color(ColorSpace::Bt709, TransferFunction::Linear),
        samples,
    )
    .expect("linear to linear RGB8");
    assert_eq!(identity_linear, samples.map(quantize8));

    let srgb = submit_rgb8(
        &backend,
        RgbColorEncoding::LINEAR_BT709,
        RgbColorEncoding::LINEAR_BT709,
        rgb_color(ColorSpace::Bt709, TransferFunction::Srgb),
        samples,
    )
    .expect("linear to sRGB RGB8");
    let expected_srgb = samples.map(|value| quantize8(srgb_from_linear(value)));
    assert!(
        srgb.iter()
            .zip(expected_srgb)
            .all(|(&actual, expected)| actual.abs_diff(expected) <= 1),
        "GPU sRGB bytes {srgb:?}, expected {expected_srgb:?}"
    );
    assert_ne!(
        identity_linear, srgb,
        "target transfer must change output bytes"
    );
}

#[test]
fn generic_output_rejects_mismatched_or_unsupported_color_contracts() {
    let Some(backend) = backend() else {
        return;
    };
    let supported_target = rgb_color(ColorSpace::Bt709, TransferFunction::Bt709);
    let mismatch = submit_rgb8(
        &backend,
        RgbColorEncoding::BT709,
        RgbColorEncoding::SRGB_BT709,
        supported_target,
        [0.25, 0.5, 0.75],
    )
    .expect_err("source encoding mismatch must fail");
    assert!(matches!(mismatch, Error::InvalidPayload(_)));

    for (name, source, target) in [
        (
            "wide source",
            RgbColorEncoding {
                primaries: RgbPrimaries::Bt2020,
                transfer: SourceTransferFunction::Linear,
            },
            supported_target,
        ),
        (
            "HDR source",
            RgbColorEncoding {
                primaries: RgbPrimaries::Bt709,
                transfer: SourceTransferFunction::Pq,
            },
            supported_target,
        ),
        (
            "wide target",
            RgbColorEncoding::BT709,
            rgb_color(ColorSpace::Bt2020, TransferFunction::Bt2020),
        ),
        (
            "HDR target",
            RgbColorEncoding::BT709,
            rgb_color(ColorSpace::Bt709, TransferFunction::Pq),
        ),
        (
            "undefined target",
            RgbColorEncoding::BT709,
            ColorSpecification::Undefined,
        ),
    ] {
        let error =
            submit_rgb8(&backend, source, source, target, [0.25, 0.5, 0.75]).expect_err(name);
        assert!(matches!(error, Error::Unsupported(_)), "{name}: {error}");
    }
}
