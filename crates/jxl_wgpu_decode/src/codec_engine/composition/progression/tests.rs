use super::*;
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction};
use jxl_gpu_protocol::{ChangedRegions, OutputId, SubmissionToken};
use jxl_wgpu::{GpuImageFrame, GpuImageOutput, ImageReadbackPipeline};
use wgpu::util::DeviceExt;

fn image() -> ImageHeaderInventory {
    let hex = include_str!("../../../../test-data/testsrc_vardct_progressive_dc_ac.jxl.hex");
    let compact = hex.split_whitespace().collect::<String>();
    let encoded = compact
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect::<Vec<_>>();
    jxl_gpu_bitstream::parse(&encoded, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .image_header
}

// Direct double-precision evaluation of the compact triangle, independently of GPU phase tables,
// accumulation order, buffer addressing, and intermediate allocations.
fn up8(
    input: &[f64],
    source: Extent2d,
    output: Extent2d,
    weights: &[jxl_gpu_bitstream::FiniteF32; 210],
) -> Vec<f64> {
    let mirror = |coordinate: i64, size: u32| {
        let period = i64::from(size) * 2;
        let x = coordinate.rem_euclid(period);
        x.min(period - x - 1) as usize
    };
    (0..output.height)
        .flat_map(|y| {
            (0..output.width).map(move |x| {
                let mut sum = 0.0;
                let mut low = f64::INFINITY;
                let mut high = f64::NEG_INFINITY;
                for row in 0..5 {
                    for col in 0..5 {
                        let ix = mirror(i64::from(x / 8) + col - 2, source.width);
                        let iy = mirror(i64::from(y / 8) + row - 2, source.height);
                        let value = input[iy * source.width as usize + ix];
                        let phase = |p: u32, tap: i64| {
                            i64::from(p.min(7 - p)) * 5 + if p < 4 { tap } else { 4 - tap }
                        };
                        let a = phase(x % 8, col);
                        let b = phase(y % 8, row);
                        let (i, j) = (a.min(b), a.max(b));
                        let coefficient = weights[(i * (41 - i) / 2 + j - i) as usize].to_f32();
                        sum += value * f64::from(coefficient);
                        low = low.min(value);
                        high = high.max(value);
                    }
                }
                sum.clamp(low, high)
            })
        })
        .collect()
}

#[test]
fn recursive_lf_rendering_clips_odd_grids_and_accounts_levels_one_through_four() {
    let backend = match pollster::block_on(WgpuBackend::request_default(Default::default())) {
        Ok(backend) => backend,
        Err(jxl_wgpu::Error::NoAdapter) => return,
        Err(error) => panic!("{error:?}"),
    };
    let memory = backend.transient_memory_budget();
    for custom in [false, true] {
        for (level, width, height) in [
            (1, 17, 9),
            (1, 1, 33),
            (2, 257, 17),
            (3, 1025, 9),
            (4, 8193, 1),
        ] {
            let mut image = image();
            image.width = width;
            image.height = height;
            if custom {
                image
                    .upsampling_weights
                    .up8
                    .fill(jxl_gpu_bitstream::FiniteF32::from_f32(1.0 / 32.0).unwrap());
            }
            let mut format = PixelFormat::rgb_f32(
                RgbChannelOrder::Rgb,
                false,
                crate::vardct_rgb8_format().color_spec,
            );
            let ColorSpecification::Defined(ref mut color) = format.color_spec else {
                unreachable!()
            };
            color.transfer = TransferFunction::Linear;
            let renderer = LfPreview::new(
                backend.clone(),
                &image,
                &GpuOutputRequest::color(format).unwrap(),
            )
            .unwrap();
            let source = Extent2d::new(
                width.div_ceil(1 << (3 * level)),
                height.div_ceil(1 << (3 * level)),
            );
            let stride = source.width + 3;
            let mut cpu = std::array::from_fn::<_, 3, _>(|channel| {
                (0..source.width * source.height)
                    .map(|i| {
                        let value = match channel {
                            0 => (i % 5) as f32 * 0.005 - 0.01,
                            1 => 0.3 + (i % 7) as f32 * 0.03,
                            _ => 0.2 + (i % 3) as f32 * 0.03,
                        };
                        f64::from(value)
                    })
                    .collect::<Vec<_>>()
            });
            let leases = std::array::from_fn(|channel| {
                let mut padded = vec![f32::NAN; (stride * source.height) as usize];
                for y in 0..source.height {
                    for x in 0..source.width {
                        padded[(y * stride + x) as usize] =
                            cpu[channel][(y * source.width + x) as usize] as f32;
                    }
                }
                let buffer =
                    backend
                        .device()
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("LF scalar oracle input with poisoned padding"),
                            contents: bytemuck::cast_slice(&padded),
                            usage: wgpu::BufferUsages::STORAGE,
                        });
                let permit = memory.try_reserve(buffer.size()).unwrap();
                GpuBufferLease::from_tracked(buffer, permit)
            });
            let planes =
                ProgressiveDcXybPlanes::from_leases(leases, source.width, source.height, stride)
                    .unwrap();
            let bytes = |w: u32, h: u32| u64::from(w) * u64::from(h) * 4;
            let total = renderer.output_plan.memory.total_bytes
                + renderer.kernel.weight_bytes()
                + u64::from(level) * 3 * ResidentUpsamplePipeline::UNIFORM_BYTES
                + (0..level)
                    .map(|i| bytes(width.div_ceil(1 << (3 * i)), height.div_ceil(1 << (3 * i))) * 3)
                    .sum::<u64>();
            let blocker = memory
                .try_reserve(memory.snapshot().available_bytes - total + 1)
                .unwrap();
            let before = memory.snapshot().reserved_bytes;
            assert!(
                matches!(renderer.submit(&planes, level), Err(Error::MemoryBackpressure(jxl_wgpu::MemoryBudgetError::Exhausted { requested_bytes, .. })) if requested_bytes == total)
            );
            assert_eq!(memory.snapshot().reserved_bytes, before);
            drop(blocker);
            for invalid in [0, 5] {
                assert!(matches!(
                    renderer.submit(&planes, invalid),
                    Err(Error::EngineContract(_))
                ));
            }
            let output = renderer.submit(&planes, level).unwrap().wait().unwrap();
            let mut size = source;
            for next in (0..level).rev() {
                let extent = Extent2d::new(
                    width.div_ceil(1 << (3 * next)),
                    height.div_ceil(1 << (3 * next)),
                );
                cpu = cpu.map(|input| up8(&input, size, extent, &image.upsampling_weights.up8));
                size = extent;
            }
            let inverse = InverseOpsin::from_image(&image).unwrap();
            let expected = (0..cpu[0].len())
                .flat_map(|i| {
                    let mixed = [cpu[1][i] + cpu[0][i], cpu[1][i] - cpu[0][i], cpu[2][i]];
                    let lms = std::array::from_fn::<_, 3, _>(|c| {
                        ((mixed[c] - f64::from(inverse.opsin_bias[c]).cbrt()).powi(3)
                            + f64::from(inverse.opsin_bias[c]))
                            * 255.0
                            / f64::from(inverse.intensity_target)
                    });
                    inverse.inverse_opsin_matrix.map(|row| {
                        row.into_iter()
                            .zip(lms)
                            .map(|(a, b)| f64::from(a) * b)
                            .sum::<f64>()
                    })
                })
                .collect::<Vec<_>>();
            let frame = GpuImageFrame {
                token: SubmissionToken(1),
                outputs: vec![GpuImageOutput {
                    id: OutputId(0),
                    layout: renderer.layout.clone(),
                    buffer: output,
                }],
                changed: ChangedRegions::default(),
            };
            let actual = ImageReadbackPipeline::new(&backend)
                .submit(&frame)
                .unwrap()
                .wait()
                .unwrap()
                .frame
                .outputs[0]
                .bytes
                .clone();
            assert_eq!(actual.len(), expected.len() * 4);
            let error = actual
                .chunks_exact(4)
                .zip(&expected)
                .map(|(a, b)| {
                    let actual = f32::from_le_bytes(a.try_into().unwrap());
                    assert!(actual.is_finite());
                    (f64::from(actual) - b).abs()
                })
                .fold(0_f64, f64::max);
            assert!(error < 1e-5, "level {level} custom {custom}: {error}");
            drop(frame);
            drop(planes);
            drop(renderer);
            assert_eq!(memory.snapshot().reserved_bytes, 0);
        }
    }
}
