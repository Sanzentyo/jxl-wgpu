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
    ChromaLocation2d, ColorRange, ColorSpace, ColorSpec, ColorSpecification, Error, ImageLayout,
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
    Ok(submit_format(
        backend,
        source_encoding,
        request_encoding,
        PixelFormat::rgb8(RgbChannelOrder::Rgb, false, target_color),
        samples,
    )?
    .1)
}

fn submit_format(
    backend: &WgpuBackend,
    source_encoding: RgbColorEncoding,
    request_encoding: RgbColorEncoding,
    format: PixelFormat,
    samples: [f32; 3],
) -> Result<(ImageLayout, Vec<u8>), Error> {
    let extent = Extent2d::new(1, 1);
    let channels = [vec![samples[0]], vec![samples[1]], vec![samples[2]]];
    let mut session = backend.create_session(&frame_desc(extent), plan(extent, source_encoding))?;
    enqueue(&mut session, extent, &channels);
    let token = session.submit_image(
        RenderIntent::Final,
        ImageOutputRequest::new(request_encoding, format),
    )?;
    let output = session.wait_image(token)?.outputs.remove(0);
    Ok((output.layout, output.bytes))
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

fn signed_map(value: f32, map: impl FnOnce(f32) -> f32) -> f32 {
    map(value.abs()).copysign(value)
}

fn pq_to_linear(value: f32) -> f32 {
    signed_map(value, |magnitude| {
        let m1 = 2610.0 / 16384.0;
        let m2 = (2523.0 / 4096.0) * 128.0;
        let c1 = 3424.0 / 4096.0;
        let c2 = (2413.0 / 4096.0) * 32.0;
        let c3 = (2392.0 / 4096.0) * 32.0;
        let powered = magnitude.powf(1.0 / m2);
        ((powered - c1).max(0.0) / (c2 - c3 * powered).max(1e-10)).powf(1.0 / m1)
    })
}

fn pq_from_linear(value: f32) -> f32 {
    signed_map(value, |magnitude| {
        let m1 = 2610.0 / 16384.0;
        let m2 = (2523.0 / 4096.0) * 128.0;
        let c1 = 3424.0 / 4096.0;
        let c2 = (2413.0 / 4096.0) * 32.0;
        let c3 = (2392.0 / 4096.0) * 32.0;
        let powered = magnitude.powf(m1);
        ((c1 + c2 * powered) / (1.0 + c3 * powered)).powf(m2)
    })
}

fn hlg_to_linear(value: f32) -> f32 {
    signed_map(value, |magnitude| {
        let a = 0.178_832_77;
        let b = 1.0 - 4.0 * a;
        let c = 0.559_910_7;
        if magnitude <= 0.5 {
            magnitude * magnitude / 3.0
        } else {
            ((magnitude - c) / a).exp().mul_add(1.0, b) / 12.0
        }
    })
}

fn hlg_from_linear(value: f32) -> f32 {
    signed_map(value, |magnitude| {
        let a = 0.178_832_77;
        let b = 1.0 - 4.0 * a;
        let c = 0.559_910_7;
        if magnitude <= 1.0 / 12.0 {
            (3.0 * magnitude).sqrt()
        } else {
            a * (12.0 * magnitude - b).ln() + c
        }
    })
}

fn bt2020_from_linear(value: f32) -> f32 {
    signed_map(value, |magnitude| {
        let alpha = 1.099_296_8;
        let beta = 0.018_053_97;
        if magnitude < beta {
            4.5 * magnitude
        } else {
            alpha * magnitude.powf(0.45) - (alpha - 1.0)
        }
    })
}

type Matrix3 = [[f32; 3]; 3];

fn multiply_matrix(lhs: Matrix3, rhs: Matrix3) -> Matrix3 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|inner| lhs[row][inner] * rhs[inner][column])
                .sum()
        })
    })
}

fn apply_matrix(matrix: Matrix3, value: [f32; 3]) -> [f32; 3] {
    matrix.map(|row| row[0] * value[0] + row[1] * value[1] + row[2] * value[2])
}

fn scalar_primaries_transform(
    source: RgbPrimaries,
    target: ColorSpace,
    value: [f32; 3],
) -> [f32; 3] {
    let source_to_xyz = match source {
        RgbPrimaries::Bt709 => [
            [0.412_456_4, 0.357_576_1, 0.180_437_5],
            [0.212_672_9, 0.715_152_2, 0.072_175],
            [0.019_333_9, 0.119_192, 0.950_304_1],
        ],
        RgbPrimaries::Bt2020 => [
            [0.636_958, 0.144_616_9, 0.168_881],
            [0.262_700_2, 0.677_998_1, 0.059_301_7],
            [0.0, 0.028_072_7, 1.060_985_1],
        ],
        RgbPrimaries::DisplayP3 => [
            [0.486_570_95, 0.265_667_7, 0.198_217_29],
            [0.228_974_57, 0.691_738_55, 0.079_286_91],
            [0.0, 0.045_113_38, 1.043_944_4],
        ],
        RgbPrimaries::Undefined => panic!("undefined primaries are not an oracle input"),
    };
    let xyz_to_target = match target {
        ColorSpace::Bt709 => [
            [3.240_454_2, -1.537_138_5, -0.498_531_4],
            [-0.969_266, 1.876_010_8, 0.041_556],
            [0.055_643_4, -0.204_025_9, 1.057_225_2],
        ],
        ColorSpace::Bt2020 => [
            [1.716_651_2, -0.355_670_8, -0.253_366_3],
            [-0.666_684_4, 1.616_481_2, 0.015_768_5],
            [0.017_639_9, -0.042_770_6, 0.942_103_1],
        ],
        ColorSpace::DisplayP3 => [
            [2.493_497, -0.931_383_6, -0.402_710_8],
            [-0.829_489, 1.762_664, 0.023_624_7],
            [0.035_845_8, -0.076_172_4, 0.956_884_5],
        ],
        unsupported => panic!("unsupported oracle primaries {unsupported:?}"),
    };
    apply_matrix(multiply_matrix(xyz_to_target, source_to_xyz), value)
}

fn scalar_color_transform(
    samples: [f32; 3],
    source: RgbColorEncoding,
    target_space: ColorSpace,
    target_transfer: TransferFunction,
) -> [u8; 3] {
    let source_linear = samples.map(|value| match source.transfer {
        SourceTransferFunction::Linear => value,
        SourceTransferFunction::Srgb => srgb_to_linear(value),
        SourceTransferFunction::Bt709 => bt709_to_linear(value),
        SourceTransferFunction::Pq => pq_to_linear(value),
        SourceTransferFunction::Hlg => hlg_to_linear(value),
        SourceTransferFunction::Gamma => panic!("gamma exponent is not available"),
    });
    let target_linear = scalar_primaries_transform(source.primaries, target_space, source_linear);
    target_linear.map(|value| {
        quantize8(match target_transfer {
            TransferFunction::Linear => value,
            TransferFunction::Srgb | TransferFunction::Sycc => srgb_from_linear(value),
            TransferFunction::Bt709 => signed_map(value, |magnitude| {
                if magnitude < 0.018 {
                    4.5 * magnitude
                } else {
                    1.099 * magnitude.powf(0.45) - 0.099
                }
            }),
            TransferFunction::Pq => pq_from_linear(value),
            TransferFunction::Hlg => hlg_from_linear(value),
            TransferFunction::Bt2020 => bt2020_from_linear(value),
            TransferFunction::Undefined | TransferFunction::Smpte240M => {
                panic!("unsupported oracle transfer")
            }
        })
    })
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
fn gpu_output_converts_wide_gamut_and_hdr_contracts() {
    let Some(backend) = backend() else {
        return;
    };
    let cases = [
        (
            "BT.2020 linear to BT.709 linear",
            RgbColorEncoding {
                primaries: RgbPrimaries::Bt2020,
                transfer: SourceTransferFunction::Linear,
            },
            ColorSpace::Bt709,
            TransferFunction::Linear,
            [0.21, 0.43, 0.67],
        ),
        (
            "Display-P3 linear to BT.709 sRGB",
            RgbColorEncoding {
                primaries: RgbPrimaries::DisplayP3,
                transfer: SourceTransferFunction::Linear,
            },
            ColorSpace::Bt709,
            TransferFunction::Srgb,
            [0.24, 0.46, 0.61],
        ),
        (
            "BT.2020 PQ to BT.2020 linear",
            RgbColorEncoding {
                primaries: RgbPrimaries::Bt2020,
                transfer: SourceTransferFunction::Pq,
            },
            ColorSpace::Bt2020,
            TransferFunction::Linear,
            [0.45, 0.58, 0.72],
        ),
        (
            "BT.709 linear to BT.709 PQ",
            RgbColorEncoding::LINEAR_BT709,
            ColorSpace::Bt709,
            TransferFunction::Pq,
            [0.03, 0.18, 0.74],
        ),
        (
            "Display-P3 HLG to Display-P3 linear",
            RgbColorEncoding {
                primaries: RgbPrimaries::DisplayP3,
                transfer: SourceTransferFunction::Hlg,
            },
            ColorSpace::DisplayP3,
            TransferFunction::Linear,
            [0.28, 0.52, 0.81],
        ),
        (
            "BT.709 linear to BT.709 HLG",
            RgbColorEncoding::LINEAR_BT709,
            ColorSpace::Bt709,
            TransferFunction::Hlg,
            [0.01, 0.12, 0.65],
        ),
        (
            "BT.2020 linear to BT.2020 OETF",
            RgbColorEncoding {
                primaries: RgbPrimaries::Bt2020,
                transfer: SourceTransferFunction::Linear,
            },
            ColorSpace::Bt2020,
            TransferFunction::Bt2020,
            [0.02, 0.31, 0.77],
        ),
    ];

    for (name, source, target_space, target_transfer, samples) in cases {
        let actual = submit_rgb8(
            &backend,
            source,
            source,
            rgb_color(target_space, target_transfer),
            samples,
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let expected = scalar_color_transform(samples, source, target_space, target_transfer);
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(&actual, expected)| actual.abs_diff(expected) <= 2),
            "{name}: GPU bytes {actual:?}, scalar bytes {expected:?}"
        );
    }
}

#[test]
fn gpu_output_implements_both_bt2020_ycbcr_matrices() {
    let Some(backend) = backend() else {
        return;
    };
    let source = RgbColorEncoding {
        primaries: RgbPrimaries::Bt2020,
        transfer: SourceTransferFunction::Linear,
    };
    let samples = [0.18, 0.43, 0.72];
    let encoded = samples.map(bt2020_from_linear);
    let kr = 0.2627;
    let kb = 0.0593;
    let kg = 1.0 - kr - kb;

    for (encoding, expected) in [
        (YcbcrEncoding::Bt2020, {
            let y = kr * encoded[0] + kg * encoded[1] + kb * encoded[2];
            [
                y,
                (encoded[2] - y) / (2.0 * (1.0 - kb)) + 0.5,
                (encoded[0] - y) / (2.0 * (1.0 - kr)) + 0.5,
            ]
        }),
        (YcbcrEncoding::Bt2020ConstantLuminance, {
            let y = bt2020_from_linear(kr * samples[0] + kg * samples[1] + kb * samples[2]);
            let cb_divisor = if encoded[2] > y { 1.5816 } else { 1.9404 };
            let cr_divisor = if encoded[0] > y { 0.9936 } else { 1.7184 };
            [
                y,
                (encoded[2] - y) / cb_divisor + 0.5,
                (encoded[0] - y) / cr_divisor + 0.5,
            ]
        }),
    ] {
        let target = ColorSpecification::Defined(ColorSpec {
            space: ColorSpace::Bt2020,
            encoding,
            transfer: TransferFunction::Bt2020,
            range: ColorRange::Full,
            chroma_location: ChromaLocation2d::BOTH,
        });
        let format = PixelFormat::i444(8, 8, target).expect("valid BT.2020 I444");
        let (layout, bytes) = submit_format(&backend, source, source, format, samples)
            .unwrap_or_else(|error| panic!("{encoding:?}: {error}"));
        let actual: [u8; 3] = std::array::from_fn(|plane| {
            bytes[usize::try_from(layout.planes[plane].offset).expect("plane offset fits usize")]
        });
        let expected = expected.map(quantize8);
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(&actual, expected)| actual.abs_diff(expected) <= 2),
            "{encoding:?}: GPU bytes {actual:?}, scalar bytes {expected:?}"
        );
    }
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
            "undefined source primaries",
            RgbColorEncoding {
                primaries: RgbPrimaries::Undefined,
                transfer: SourceTransferFunction::Linear,
            },
            supported_target,
        ),
        (
            "source gamma without exponent",
            RgbColorEncoding {
                primaries: RgbPrimaries::Bt709,
                transfer: SourceTransferFunction::Gamma,
            },
            supported_target,
        ),
        (
            "sensor target",
            RgbColorEncoding::BT709,
            rgb_color(ColorSpace::Sensor, TransferFunction::Linear),
        ),
        (
            "undefined target transfer",
            RgbColorEncoding::BT709,
            rgb_color(ColorSpace::Bt709, TransferFunction::Undefined),
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
