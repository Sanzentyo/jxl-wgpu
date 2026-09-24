//! Native/Rust animation composition, shared frame admission and GPU ownership.

use super::*;
use crate::{
    AnimationHeader, BlendMode, FrameBlend, FrameCrop, FrameIndex, FrameOptions, FrameTiming,
    ReferenceSlot, VarDctAnimationDescriptor, VarDctAnimationSession,
};
use jxl_gpu_formats::{PixelFormat, RgbChannelOrder};
use jxl_test_support::gpu::planes::open_fragmented;
use jxl_test_support::oracles::extra_channels::{floats, rust_frame_planes};
use jxl_test_support::oracles::progressive::native_updates;
use jxl_wgpu_decode::WgpuDecodeEngine;

mod layered_still;
mod reference_only;

fn timebase(numerator: u32, denominator: u32, loops: u32, timecodes: bool) -> AnimationHeader {
    AnimationHeader::Animation {
        ticks_per_second_numerator: numerator.try_into().unwrap(),
        ticks_per_second_denominator: denominator.try_into().unwrap(),
        num_loops: loops,
        have_timecodes: timecodes,
    }
}

fn backend() -> WgpuBackend {
    pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("actual GPU required for VarDCT animation evidence")
}

fn configuration() -> VarDctConfig {
    VarDctConfig {
        quantization: VarDctQuantization::new(
            35_252,
            16,
            crate::VarDctHfMultiplier::new(12).unwrap(),
        )
        .unwrap(),
        ..Default::default()
    }
}

fn descriptor(
    width: usize,
    height: usize,
    animation: AnimationHeader,
) -> VarDctAnimationDescriptor {
    VarDctAnimationDescriptor::new(width as u32, height as u32, animation).unwrap()
}

#[derive(Clone)]
struct Layer {
    width: usize,
    height: usize,
    options: FrameOptions,
}

fn options(
    duration: u32,
    timecode: Option<u32>,
    mode: BlendMode,
    source: u8,
    save: u8,
) -> FrameOptions {
    FrameOptions {
        timing: FrameTiming {
            duration_ticks: duration,
            timecode,
        },
        color_blend: FrameBlend {
            mode,
            source_reference: ReferenceSlot::new(source).unwrap(),
            clamp: mode == BlendMode::Multiply,
        },
        save_as_reference: ReferenceSlot::new(save).unwrap(),
        ..Default::default()
    }
}

fn pixels(width: usize, height: usize, frame: usize) -> Vec<[u8; 3]> {
    (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                std::array::from_fn(|c| (32 + (x * 3 + y * 5 + c * 41 + frame * 23) % 144) as u8)
            })
        })
        .collect()
}

fn compare(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert!(
            (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
            "{label}, sample {index}: {a} vs {b}"
        );
        let code = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
        assert!(code(a).abs_diff(code(b)) <= 1, "RGB8 sample {index}");
    }
}

// Independent signal-domain composition of separately Rust-decoded stills. This does not
// consume encoder frame plans, decoder inventories, or production blending helpers.
fn compose(width: usize, height: usize, layers: &[Layer], samples: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let blank: Vec<_> = (0..width * height)
        .flat_map(|_| [0.0, 0.0, 0.0, 1.0])
        .collect();
    let mut references = [blank.clone(), blank.clone(), blank.clone(), blank];
    let mut presentations = Vec::new();
    for (index, (layer, samples)) in layers.iter().zip(samples).enumerate() {
        let frame = &layer.options;
        let mut canvas = references[usize::from(frame.color_blend.source_reference.get())].clone();
        let (left, top) = frame
            .crop
            .map_or((0, 0), |crop| (i64::from(crop.x()), i64::from(crop.y())));
        for y in 0..layer.height {
            for x in 0..layer.width {
                let (cx, cy) = (left + x as i64, top + y as i64);
                if cx < 0 || cy < 0 || cx >= width as i64 || cy >= height as i64 {
                    continue;
                }
                for c in 0..3 {
                    let dest = (cy as usize * width + cx as usize) * 4 + c;
                    let foreground = samples[(y * layer.width + x) * 4 + c];
                    canvas[dest] = match frame.color_blend.mode {
                        BlendMode::Replace => foreground,
                        BlendMode::Add => canvas[dest] + foreground,
                        BlendMode::Multiply => {
                            canvas[dest]
                                * if frame.color_blend.clamp {
                                    foreground.clamp(0.0, 1.0)
                                } else {
                                    foreground
                                }
                        }
                        _ => unreachable!("RGB-only animation"),
                    };
                }
            }
        }
        let last = index + 1 == layers.len();
        if !last && (frame.timing.duration_ticks == 0 || frame.save_as_reference.get() != 0) {
            references[usize::from(frame.save_as_reference.get())] = canvas.clone();
        }
        if last || frame.timing.duration_ticks != 0 {
            presentations.push(canvas);
        }
    }
    presentations
}

fn encode_layers(
    context: &WgpuContext,
    mut session: VarDctAnimationSession,
    layers: &[Layer],
    mut still: impl FnMut(BufferImageSource) -> Vec<u8>,
    container: bool,
) -> (Vec<u8>, Vec<Vec<f32>>) {
    let mut submissions = Vec::new();
    let mut samples = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        let input = pixels(layer.width, layer.height, index);
        let source = padded_rgb_source_sized(context, layer.width, layer.height, &input);
        let baseline = still(source.clone());
        let native = native_updates(&baseline, false).expect("required native animation oracle");
        let decoded = floats(&native.last().unwrap().pixels);
        // Retain a fixed source-quality bound at the declared quantizers, independently of
        // the composition checks against separately decoded frames.
        let rgb8: Vec<_> = decoded
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| {
                p[..3]
                    .iter()
                    .map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
            })
            .collect();
        let rust = decode_rgb8_sized(&baseline, layer.width, layer.height);
        assert!(max_abs_error(&rust, &rgb8) <= 1);
        let quality = psnr(&input, &rgb8);
        assert!(
            quality > 30.0,
            "{}x{} frame {index}: PSNR={quality}, source={:?}, native={:?}, Rust={:?}",
            layer.width,
            layer.height,
            input[0],
            &decoded[..4],
            &decode_rgb8_sized(&baseline, layer.width, layer.height)[..3]
        );
        let mut rust = rust_frame_planes(&baseline);
        assert_eq!(rust.len(), 1);
        let rust = rust.remove(0).0;
        compare("Rust/native physical source", &rust, &decoded);
        samples.push(rust);
        let submitted = if index + 1 == layers.len() {
            session.submit_last_frame(source, layer.options.clone())
        } else {
            session.submit_frame(source, layer.options.clone())
        }
        .unwrap();
        submissions.push(submitted);
        assert_eq!(
            session.next_frame_index(),
            FrameIndex::new(index as u32 + 1)
        );
    }
    // Independent futures may complete and be inserted in reverse order.
    for (index, submission) in submissions.into_iter().rev().enumerate() {
        let artifacts = if index % 2 == 0 {
            pollster::block_on(submission)
        } else {
            submission.wait()
        }
        .unwrap();
        session.insert(artifacts).unwrap();
    }
    let encoded = if container {
        session.finish_indexed_container(Default::default(), Default::default())
    } else {
        session.finish_raw()
    }
    .unwrap();
    assert_eq!(context.memory_budget().snapshot().reserved_bytes, 0);
    (encoded, samples)
}

fn check_sequence(
    backend: &WgpuBackend,
    encoded: &[u8],
    descriptor: &VarDctAnimationDescriptor,
    layers: &[Layer],
    samples: &[Vec<f32>],
    passes: usize,
) {
    check_sequence_with_oracle(
        backend,
        encoded,
        descriptor,
        layers,
        samples,
        passes,
        CompositionOracle::JxlOxide,
    );
}

enum CompositionOracle {
    JxlOxide,
    /// Native whole-stream output plus explicit composition of independently Rust-decoded stills.
    IndependentStills,
}

fn check_sequence_with_oracle(
    backend: &WgpuBackend,
    encoded: &[u8],
    descriptor: &VarDctAnimationDescriptor,
    layers: &[Layer],
    samples: &[Vec<f32>],
    passes: usize,
    oracle: CompositionOracle,
) {
    let width = descriptor.canvas_width() as usize;
    let height = descriptor.canvas_height() as usize;
    let inventory = jxl_gpu_bitstream::parse(encoded, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(
        (inventory.image_header.width, inventory.image_header.height),
        (width as u32, height as u32)
    );
    let (ticks_per_second_numerator, ticks_per_second_denominator) = match descriptor.animation() {
        AnimationHeader::Still => {
            assert!(inventory.image_header.animation.is_none());
            (1, 1)
        }
        AnimationHeader::Animation {
            ticks_per_second_numerator,
            ticks_per_second_denominator,
            num_loops,
            have_timecodes,
        } => {
            let animation = inventory.image_header.animation.unwrap();
            assert_eq!(
                animation.ticks_per_second_numerator,
                ticks_per_second_numerator.get()
            );
            assert_eq!(
                animation.ticks_per_second_denominator,
                ticks_per_second_denominator.get()
            );
            assert_eq!(animation.num_loops, num_loops);
            assert_eq!(animation.have_timecodes, have_timecodes);
            (
                ticks_per_second_numerator.get(),
                ticks_per_second_denominator.get(),
            )
        }
    };
    assert!(inventory.image_header.xyb_encoded);
    assert_eq!(inventory.frames.len(), layers.len());
    for (index, (frame, layer)) in inventory.frames.iter().zip(layers).enumerate() {
        let options = &layer.options;
        assert_eq!(
            (frame.width, frame.height),
            (layer.width as u32, layer.height as u32)
        );
        assert_eq!(
            (frame.x0, frame.y0),
            options.crop.map_or((0, 0), |c| (c.x(), c.y()))
        );
        assert_eq!(frame.duration_ticks, options.timing.duration_ticks);
        assert_eq!(frame.timecode, options.timing.timecode);
        assert_eq!(frame.is_last, index + 1 == layers.len());
        assert_eq!(
            frame.save_as_reference,
            u32::from(options.save_as_reference.get())
        );
        assert_eq!(
            frame.num_passes as usize,
            if options.kind == crate::FrameKind::ReferenceOnly {
                1
            } else {
                passes
            }
        );
        assert_eq!(
            frame.frame_type,
            if options.kind == crate::FrameKind::ReferenceOnly {
                jxl_gpu_bitstream::FrameType::ReferenceOnly
            } else {
                jxl_gpu_bitstream::FrameType::Regular
            }
        );
        let mode = match options.color_blend.mode {
            BlendMode::Replace => jxl_gpu_bitstream::FrameBlendMode::Replace,
            BlendMode::Add => jxl_gpu_bitstream::FrameBlendMode::Add,
            BlendMode::Multiply => jxl_gpu_bitstream::FrameBlendMode::Multiply,
            _ => unreachable!(),
        };
        assert_eq!(frame.color_blend.mode, mode);
        let full_canvas = options.crop.is_none_or(|crop| {
            crop.x() <= 0
                && crop.y() <= 0
                && i64::from(crop.x()) + i64::from(crop.width()) >= width as i64
                && i64::from(crop.y()) + i64::from(crop.height()) >= height as i64
        });
        if mode != jxl_gpu_bitstream::FrameBlendMode::Replace || !full_canvas {
            assert_eq!(
                frame.color_blend.source,
                u32::from(options.color_blend.source_reference.get())
            );
        }
    }
    let expected = compose(width, height, layers, samples);
    let timings: Vec<_> = layers
        .iter()
        .enumerate()
        .filter(|(i, l)| *i + 1 == layers.len() || l.options.timing.duration_ticks != 0)
        .map(|(_, l)| l.options.timing)
        .collect();
    let native = native_updates(encoded, false).expect("required native animation oracle");
    let native: Vec<_> = native.iter().filter(|update| update.complete).collect();

    let other: Option<Vec<Vec<f32>>> = if matches!(oracle, CompositionOracle::JxlOxide) {
        let oxide = jxl_oxide::JxlImage::read_with_defaults(encoded).unwrap();
        Some(
            (0..oxide.num_loaded_keyframes())
                .map(|index| {
                    let render = oxide.render_frame(index).unwrap().image_all_channels();
                    render
                        .buf()
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .flat_map(|p| [p[0], p[1], p[2], 1.0])
                        .collect()
                })
                .collect(),
        )
    } else {
        // jxl-oxide panics on empty off-canvas foregrounds; Rust jxl rejects oversized
        // ReferenceOnly internal buffers. Keep these inputs and compare native whole-stream
        // output and GPU output to independent composition of Rust-decoded physical stills.
        None
    };
    assert_eq!(native.len(), expected.len());

    if let Some(other) = &other {
        assert_eq!(other.len(), expected.len());
    }
    for (index, (native, expected)) in native.iter().zip(&expected).enumerate() {
        assert_eq!(
            (native.duration, native.timecode),
            (
                timings[index].duration_ticks,
                timings[index].timecode.unwrap_or(0)
            )
        );
        assert_eq!(native.is_last, index + 1 == timings.len());
        compare(
            &format!("native presentation {index}"),
            &floats(&native.pixels),
            expected,
        );
        if let Some(other) = &other {
            compare(
                &format!("jxl-oxide presentation {index}"),
                &other[index],
                expected,
            );
        }
    }
    let readback = ImageReadbackPipeline::new(backend);
    let format = PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        vardct_rgb8_format().color_spec,
    );
    let mut whole = Vec::new();
    for window in [u64::MAX, 256] {
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(window).unwrap()),
        );
        let request = GpuOutputRequest::color(format.clone()).unwrap();
        let mut session = if window == u64::MAX {
            decoder.open(encoded, request).unwrap()
        } else {
            open_fragmented(&decoder, encoded, request)
        };
        let mut actual = Vec::new();
        let mut held = Vec::new();
        let mut ticks = 0;
        while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
            let index = actual.len();
            assert_eq!(frame.metadata.index, index);
            assert_eq!(frame.metadata.duration.ticks, timings[index].duration_ticks);
            assert_eq!(frame.metadata.timecode, timings[index].timecode);
            assert_eq!(frame.metadata.presentation_ticks, ticks);
            ticks += u64::from(timings[index].duration_ticks);
            let pixels = readback
                .submit(frame.output())
                .unwrap()
                .wait()
                .unwrap()
                .frame
                .outputs[0]
                .bytes
                .clone();
            compare(
                &format!("GPU presentation {index}"),
                &floats(&pixels),
                &expected[index],
            );
            compare(
                &format!("GPU/native presentation {index}"),
                &floats(&pixels),
                &floats(&native[index].pixels),
            );
            actual.push(pixels);
            // Release the logical frame slot while retaining its separately budgeted outputs.
            held.push(jxl_wgpu::GpuImageFrame {
                token: frame.output().token,
                outputs: frame
                    .output()
                    .outputs
                    .iter()
                    .map(|output| jxl_wgpu::GpuImageOutput {
                        id: output.id,
                        layout: output.layout.clone(),
                        buffer: output.buffer.clone(),
                    })
                    .collect(),
                changed: frame.output().changed.clone(),
            });
        }
        assert_eq!(actual.len(), expected.len());
        drop(session);
        for (frame, pixels) in held.iter().zip(&actual) {
            assert_eq!(
                &readback
                    .submit(frame)
                    .unwrap()
                    .wait()
                    .unwrap()
                    .frame
                    .outputs[0]
                    .bytes,
                pixels
            );
        }
        drop(held);
        let parsed = jxl_gpu_bitstream::parse(encoded, Default::default()).unwrap();
        if let Some(index) =
            jxl_gpu_bitstream::FrameIndex::from_container(&parsed, Default::default()).unwrap()
        {
            assert_eq!(index.tick_numerator(), ticks_per_second_denominator);
            assert_eq!(index.tick_denominator().get(), ticks_per_second_numerator);
            assert_eq!(
                index
                    .entries()
                    .iter()
                    .map(|entry| entry.frames)
                    .sum::<u64>(),
                timings.len() as u64
            );
            assert_eq!(
                index
                    .entries()
                    .iter()
                    .map(|entry| entry.duration_ticks)
                    .sum::<u64>(),
                ticks
            );
            for target in (0..actual.len()).rev() {
                let mut seek = decoder
                    .open_seek(
                        encoded,
                        GpuOutputRequest::color(format.clone()).unwrap(),
                        target,
                        Default::default(),
                        Default::default(),
                    )
                    .unwrap();
                let frame = pollster::block_on(seek.next_frame_async())
                    .unwrap()
                    .unwrap();
                assert_eq!(frame.metadata.index, target);
                assert_eq!(
                    frame.metadata.duration.ticks,
                    timings[target].duration_ticks
                );
                assert_eq!(frame.metadata.timecode, timings[target].timecode);
                assert!(seek.next_frame().unwrap().is_none());
                drop(seek);
                // The retained target remains readable after all seek state is released.
                let pixels = &readback
                    .submit(frame.output())
                    .unwrap()
                    .wait()
                    .unwrap()
                    .frame
                    .outputs[0]
                    .bytes;
                assert_eq!(pixels, &actual[target]);
                compare(
                    "indexed GPU/native target",
                    &floats(pixels),
                    &floats(&native[target].pixels),
                );
            }
        }
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
        if window == u64::MAX {
            whole = actual;
        } else {
            assert_eq!(actual, whole);
        }
    }
}

#[test]
fn indexed_animation_restores_an_old_reference_across_a_new_independent_frame() {
    let backend = backend();
    let context = WgpuContext::from_backend(&backend);
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), configuration()).unwrap();
    let desc = descriptor(9, 7, timebase(60_000, 1001, 2, true));
    let layers: Vec<_> = [
        (0, BlendMode::Replace, 0, 0),
        (3, BlendMode::Add, 0, 1),
        (5, BlendMode::Replace, 0, 0),
        (7, BlendMode::Add, 1, 0),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (duration, mode, source, save))| Layer {
        width: 9,
        height: 7,
        options: options(duration, Some(100 + i as u32), mode, source, save),
    })
    .collect();
    let (encoded, samples) = encode_layers(
        &context,
        encoder.begin_animation(desc.clone()).unwrap(),
        &layers,
        |source| encoder.encode(source).unwrap(),
        true,
    );
    let parsed = jxl_gpu_bitstream::parse(&encoded, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let index = jxl_gpu_bitstream::FrameIndex::from_container(&parsed, Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(
        index.entries(),
        [
            jxl_gpu_bitstream::FrameIndexEntry {
                codestream_offset: inventory.frames[0].header_bits.offset / 8,
                duration_ticks: 3,
                frames: 1
            },
            jxl_gpu_bitstream::FrameIndexEntry {
                codestream_offset: inventory.frames[2].header_bits.offset / 8,
                duration_ticks: 12,
                frames: 2
            },
        ]
    );
    let bound =
        jxl_wgpu_decode::BoundFrameIndex::new(Arc::new(inventory), Some(index), Default::default())
            .unwrap();
    assert_eq!(
        bound
            .seek(1, Default::default())
            .unwrap()
            .restart_presentation(),
        1
    );
    assert_eq!(
        bound
            .seek(2, Default::default())
            .unwrap()
            .restart_presentation(),
        0
    );
    check_sequence(&backend, &encoded, &desc, &layers, &samples, 1);
}

#[test]
fn tiled_animation_composes_signed_crops_hidden_frames_and_all_reference_slots() {
    let backend = backend();
    let context = WgpuContext::from_backend(&backend);
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), configuration()).unwrap();
    assert!(encoder.capabilities().animation);
    for timecodes in [false, true] {
        let desc = descriptor(17, 9, timebase(60_000, 1001, 2, timecodes));
        let mut layers: Vec<_> = [
            (1, BlendMode::Replace, 0, 1),
            (0, BlendMode::Replace, 0, 2),
            (257, BlendMode::Add, 2, 3),
            (0, BlendMode::Multiply, 3, 0),
            (65_536, BlendMode::Replace, 0, 1),
            (u32::MAX, BlendMode::Multiply, 1, 0),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (duration, mode, source, save))| Layer {
            width: 17,
            height: 9,
            options: options(
                duration,
                timecodes.then_some(0x1234_5678 + i as u32),
                mode,
                source,
                save,
            ),
        })
        .collect();
        for (index, x, y, width, height) in [(2, -2, 3, 9, 7), (4, 4, -1, 5, 6)] {
            layers[index].width = width;
            layers[index].height = height;
            layers[index].options.crop =
                Some(FrameCrop::new(x, y, width as u32, height as u32).unwrap());
        }
        layers[5].options.color_blend.clamp = false;
        let (encoded, samples) = encode_layers(
            &context,
            encoder.begin_animation(desc.clone()).unwrap(),
            &layers,
            |s| encoder.encode(s).unwrap(),
            timecodes,
        );
        check_sequence(&backend, &encoded, &desc, &layers, &samples, 1);
    }
}

#[test]
fn progressive_animation_crosses_ac_and_lf_groups_with_variable_source_extents() {
    let backend = backend();
    let context = WgpuContext::from_backend(&backend);
    let config = VarDctConfig {
        progressive: progressive::combined(),
        group_order: super::super::VarDctGroupOrder::center_first(),
        ..configuration()
    };
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
    for (width, height) in [(1, 1), (257, 17), (2057, 1)] {
        let desc = descriptor(width, height, timebase(1024, 256, 65_535, false));
        let layers = [
            Layer {
                width,
                height,
                options: options(1, None, BlendMode::Replace, 0, 1),
            },
            Layer {
                width: 1,
                height: 1,
                options: FrameOptions {
                    crop: Some(FrameCrop::new(width as i32 - 1, 0, 1, 1).unwrap()),
                    ..options(0, None, BlendMode::Add, 1, 0)
                },
            },
        ];
        let (encoded, samples) = encode_layers(
            &context,
            encoder.begin_animation(desc.clone()).unwrap(),
            &layers,
            |s| encoder.encode(s).unwrap(),
            false,
        );
        check_sequence(&backend, &encoded, &desc, &layers, &samples, 5);
    }
}

#[test]
fn single_and_mixed_transform_animations_share_the_frame_contract() {
    let backend = backend();
    let context = WgpuContext::from_backend(&backend);
    let config = VarDctConfig {
        progressive: progressive::combined(),
        ..configuration()
    };
    for (encoder, width, height) in [
        (
            VarDctEncoder::new_with_config(context.clone(), VarDctStrategy::Dct8, config.clone())
                .unwrap(),
            8,
            8,
        ),
        (
            VarDctEncoder::new_with_strategy_map(
                context.clone(),
                mixed::packed_map(25, 17, false),
                config,
            )
            .unwrap(),
            25,
            17,
        ),
    ] {
        let desc = descriptor(width - 1, height - 1, timebase(1000, 1024, u32::MAX, true));
        let layers: Vec<_> = (0..3)
            .map(|i| Layer {
                width,
                height,
                options: FrameOptions {
                    crop: Some(
                        FrameCrop::new(-(i as i32), -1, width as u32, height as u32).unwrap(),
                    ),
                    ..options(
                        i + 1,
                        Some(i),
                        if i == 1 {
                            BlendMode::Add
                        } else {
                            BlendMode::Replace
                        },
                        1,
                        u8::from(i != 2),
                    )
                },
            })
            .collect();
        let (encoded, samples) = encode_layers(
            &context,
            encoder.begin_animation(desc.clone()).unwrap(),
            &layers,
            |s| encoder.encode(s).unwrap(),
            true,
        );
        check_sequence(&backend, &encoded, &desc, &layers, &samples, 5);
    }
}

#[test]
fn animation_descriptor_checks_wire_dimensions_and_timebase_bounds() {
    let animation = timebase(100, 1, 0, false);
    for (width, height) in [(0, 1), (1, 0), (1 << 30, 1), (1, u32::MAX)] {
        assert!(matches!(
            VarDctAnimationDescriptor::new(width, height, animation),
            Err(EncodeError::InvalidConfiguration(_))
        ));
    }
    for animation in [
        timebase(u32::MAX, 1, 0, false),
        timebase(100, 1025, 0, false),
    ] {
        assert!(matches!(
            VarDctAnimationDescriptor::new(8, 8, animation),
            Err(EncodeError::InvalidConfiguration(_))
        ));
    }
    // Exercise all U32 alternatives in the stream timebase at their legal boundaries.
    for (num, den, loops) in [
        (100, 1, 0),
        (1000, 1001, 7),
        (1, 2, 8),
        (1024, 256, 65_535),
        (1 << 30, 1024, u32::MAX),
    ] {
        let animation = timebase(num, den, loops, true);
        let desc = VarDctAnimationDescriptor::new((1 << 30) - 1, 1, animation).unwrap();
        assert_eq!(desc.animation(), animation);
    }
}

#[test]
fn rejected_animation_controls_do_not_admit_memory_or_advance_the_session() {
    let context = test_context().expect("actual GPU required for animation admission");
    let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
    let desc = descriptor(8, 8, timebase(100, 1, 0, false));
    let source = padded_rgb_source_sized(&context, 8, 8, &pixels(8, 8, 0));
    let mut session = encoder.begin_animation(desc.clone()).unwrap();
    let good = options(1, None, BlendMode::Replace, 0, 0);
    let mut invalid = vec![
        FrameOptions {
            crop: Some(FrameCrop::new(0, 0, 7, 8).unwrap()),
            ..good.clone()
        },
        FrameOptions {
            crop: Some(FrameCrop::new(i32::MIN, 0, 8, 8).unwrap()),
            ..good.clone()
        },
        FrameOptions {
            crop: Some(FrameCrop::new(i32::MAX, 0, 8, 8).unwrap()),
            ..good.clone()
        },
        FrameOptions {
            extra_channel_blends: vec![FrameBlend::default()],
            ..good.clone()
        },
        FrameOptions {
            save_before_color_transform: true,
            ..good.clone()
        },
        options(1, Some(0), BlendMode::Replace, 0, 0),
        options(1, None, BlendMode::Replace, 0, 1), // final frames have no reference destination
        options(1, None, BlendMode::Blend, 0, 0),
        options(1, None, BlendMode::MultiplyAdd, 0, 0),
    ];
    let mut clamp = options(1, None, BlendMode::Add, 0, 0);
    clamp.color_blend.clamp = true;
    invalid.push(clamp);
    for options in invalid {
        assert!(matches!(
            session.submit_last_frame(source.clone(), options),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert_eq!(session.next_frame_index(), FrameIndex::new(0));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
    let mut with_codes = encoder
        .begin_animation(descriptor(8, 8, timebase(100, 1, 0, true)))
        .unwrap();
    assert!(matches!(
        with_codes.submit_frame(source.clone(), good.clone()),
        Err(EncodeError::InvalidConfiguration(_))
    ));
    assert_eq!(with_codes.next_frame_index(), FrameIndex::new(0));
    assert!(matches!(
        with_codes.finish_raw(),
        Err(EncodeError::MissingFinalFrame)
    ));
    let result = session
        .submit_last_frame(source.clone(), good.clone())
        .unwrap();
    assert!(matches!(
        session.submit_frame(source, good),
        Err(EncodeError::SessionClosed)
    ));
    session.insert(result.wait().unwrap()).unwrap();
    assert_eq!(decode_rgb8(&session.finish_raw().unwrap()).len(), 8 * 8 * 3);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn cancelled_animation_jobs_release_exact_budgets_and_failed_admission_can_retry() {
    let context = test_context().expect("actual GPU required for animation cancellation");
    let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
    let source = padded_rgb_source_sized(&context, 17, 9, &pixels(17, 9, 0));
    let bytes = encoder.memory_plan(&source).unwrap().owned_bytes_per_job;
    let desc = descriptor(17, 9, timebase(100, 1, 0, false));
    for limit in [bytes - 1, bytes] {
        let context = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(limit).unwrap(),
        )
        .unwrap();
        let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
        let mut session = encoder.begin_animation(desc.clone()).unwrap();
        let first =
            session.submit_frame(source.clone(), options(1, None, BlendMode::Replace, 0, 1));
        if limit < bytes {
            assert!(matches!(first, Err(EncodeError::MemoryBackpressure(_))));
            assert_eq!(session.next_frame_index(), FrameIndex::new(0));
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
            continue;
        }
        let first = first.unwrap();
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, bytes);
        let last = options(1, None, BlendMode::Add, 1, 0);
        assert!(matches!(
            session.submit_last_frame(source.clone(), last.clone()),
            Err(EncodeError::MemoryBackpressure(_))
        ));
        assert_eq!(session.next_frame_index(), FrameIndex::new(1));
        // Completion releases only the GPU work reservation; the completed packet remains valid.
        let first = pollster::block_on(first).unwrap();
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        let final_job = session.submit_last_frame(source.clone(), last).unwrap();
        session.insert(first).unwrap();
        session.insert(final_job.wait().unwrap()).unwrap();
        let encoded = session.finish_raw().unwrap();
        assert_eq!(rust_frame_planes(&encoded).len(), 2);
        let mut abandoned_session = encoder.begin_animation(desc.clone()).unwrap();
        let abandoned = abandoned_session
            .submit_last_frame(source.clone(), options(1, None, BlendMode::Replace, 0, 0))
            .unwrap();
        drop(abandoned_session);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, bytes);
        drop(abandoned);
        context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while encoder.in_flight_memory_stats().reserved_bytes != 0
            && std::time::Instant::now() < deadline
        {
            context.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        let mut recovered = encoder.begin_animation(desc.clone()).unwrap();
        let job = recovered
            .submit_last_frame(source.clone(), options(1, None, BlendMode::Replace, 0, 0))
            .unwrap();
        recovered.insert(job.wait().unwrap()).unwrap();
        assert_eq!(rust_frame_planes(&recovered.finish_raw().unwrap()).len(), 1);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}
