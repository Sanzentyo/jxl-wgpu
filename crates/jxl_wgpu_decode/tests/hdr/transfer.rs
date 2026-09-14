use super::oracle;
use jxl_gpu_formats::{
    ColorSpace, ColorSpecification, ImageLayout, PixelFormat, RgbChannelOrder, TransferFunction,
};
use jxl_gpu_protocol::{Extent2d, OutputOrientation, RgbColorEncoding, WhitePointAdaptation};
use jxl_wgpu::ResidentStorageBinding;
use jxl_wgpu_decode::color_output::{
    ColorOutputConfig, ColorOutputInputs, ColorOutputPacker, ColorOutputPlane, ColorOutputTransform,
};
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

fn binding(buffer: &wgpu::Buffer) -> ResidentStorageBinding<'_> {
    ResidentStorageBinding {
        buffer,
        offset: 0,
        size: NonZeroU64::new(buffer.size()).unwrap(),
    }
}

#[test]
fn gpu_display_transfers_follow_f64_luminance_and_ootf_equations() {
    let backend = super::backend();
    let device = backend.device();
    let samples = [
        [0.0, 0.0, 0.0],
        [1e-10, 2e-10, 3e-10],
        [0.001, 0.002, 0.003],
        [0.08, 0.18, 0.01],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-0.01, 0.5, 0.2],
        [-0.1, -0.2, -0.3],
        [0.0, -0.001, 0.00001],
        [-1.0, -1.0, -1.0],
    ];
    let inputs: [wgpu::Buffer; 3] = std::array::from_fn(|c| {
        let values: Vec<f32> = samples.iter().map(|rgb| rgb[c]).collect();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HDR independent transfer input"),
            contents: bytemuck::cast_slice(&values),
            usage: wgpu::BufferUsages::STORAGE,
        })
    });
    let extent = Extent2d::new(samples.len() as u32, 1);
    let size = samples.len() as u64 * 16;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("HDR transfer output"),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("HDR transfer readback"),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    });
    let packer = ColorOutputPacker::new(device).unwrap();
    let mut checks = 0;
    for nits in [
        100.0, 255.0, 280.0, 300.0, 320.0, 500.0, 1000.0, 4000.0, 10000.0,
    ] {
        for (source_space, target_space) in [
            (ColorSpace::Bt2020, ColorSpace::Bt2020),
            (ColorSpace::Bt709, ColorSpace::DisplayP3),
        ] {
            for (source, target) in [
                (TransferFunction::Linear, TransferFunction::Pq),
                (TransferFunction::Pq, TransferFunction::Linear),
                (TransferFunction::Linear, TransferFunction::Hlg),
                (TransferFunction::Hlg, TransferFunction::Linear),
                (TransferFunction::Pq, TransferFunction::Hlg),
                (TransferFunction::Hlg, TransferFunction::Pq),
                (TransferFunction::Pq, TransferFunction::Pq),
                (TransferFunction::Hlg, TransferFunction::Hlg),
            ] {
                let ColorSpecification::Defined(mut color) =
                    jxl_wgpu_decode::vardct_rgb8_format().color_spec
                else {
                    unreachable!()
                };
                color.space = target_space;
                color.transfer = target;
                let layout = ImageLayout::packed(
                    extent,
                    PixelFormat::rgb_f32(
                        RgbChannelOrder::Rgba,
                        false,
                        ColorSpecification::Defined(color),
                    ),
                )
                .unwrap();
                let config = ColorOutputConfig {
                    extent,
                    orientation: OutputOrientation::Identity,
                    intensity_target: nits as f32,
                    transform: ColorOutputTransform::Rgb(RgbColorEncoding {
                        space: source_space.rgb_space().unwrap(),
                        transfer: match source {
                            TransferFunction::Linear => jxl_gpu_protocol::TransferFunction::Linear,
                            TransferFunction::Pq => jxl_gpu_protocol::TransferFunction::Pq,
                            TransferFunction::Hlg => jxl_gpu_protocol::TransferFunction::Hlg,
                            _ => unreachable!(),
                        },
                    }),
                    white_point_adaptation: WhitePointAdaptation::Bradford,
                    linear_black_threshold: None,
                    alpha_conversion: jxl_wgpu::AlphaConversion::Preserve,
                };
                let mut encoder = device.create_command_encoder(&Default::default());
                let scratch = packer
                    .encode(
                        device,
                        &mut encoder,
                        ColorOutputInputs {
                            planes: std::array::from_fn(|c| ColorOutputPlane {
                                storage: binding(&inputs[c]),
                                width: extent.width,
                                height: 1,
                                stride: extent.width,
                            }),
                            alpha: None,
                            output: binding(&output),
                            layout: &layout,
                            config: &config,
                        },
                    )
                    .unwrap();
                encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, size);
                backend.queue().submit([encoder.finish()]);
                let (sender, receiver) = std::sync::mpsc::channel();
                staging
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        sender.send(result).unwrap();
                    });
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                receiver.recv().unwrap().unwrap();
                let mapped = staging.slice(..).get_mapped_range().unwrap();
                for (i, pixel) in mapped.as_chunks::<16>().0.iter().enumerate() {
                    let actual: [f64; 4] = std::array::from_fn(|c| {
                        f64::from(f32::from_le_bytes(pixel[c * 4..][..4].try_into().unwrap()))
                    });
                    let identity = source == target && source_space == target_space;
                    let expected = if identity {
                        samples[i].map(f64::from)
                    } else {
                        oracle::convert(
                            samples[i].map(f64::from),
                            source,
                            target,
                            source_space,
                            target_space,
                            nits,
                        )
                    };
                    for c in 0..3 {
                        let error = (actual[c] - expected[c]).abs() / (1.0 + expected[c].abs());
                        assert!(
                            actual[c].is_finite() && error <= 5e-5,
                            "{source:?}/{target:?} {source_space:?}/{target_space:?} {nits} nit pixel {i}/{c}: {}, f64 {}, error {error}",
                            actual[c],
                            expected[c]
                        );
                        if identity {
                            assert_eq!((actual[c] as f32).to_bits(), samples[i][c].to_bits());
                        }
                    }
                    assert_eq!(actual[3], 1.0);
                    checks += 1;
                }
                drop(mapped);
                staging.unmap();
                drop(scratch);
            }
        }
    }
    assert_eq!(checks, 1728);
}
