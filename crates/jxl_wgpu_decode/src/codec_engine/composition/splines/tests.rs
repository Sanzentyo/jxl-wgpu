use std::sync::Arc;

use jxl_gpu_bitstream::{BitRange, BitWriter};
use jxl_test_support::{fixtures::frame_features, gpu::planes};
use jxl_wgpu::WgpuBackend;

use super::*;

fn pack(value: i32) -> u32 {
    (value as u32).wrapping_shl(1) ^ ((value >> 31) as u32)
}

fn make_plan(values: &[u32], window: u64, skew: u8) -> (Plan, Arc<GpuCodestream>) {
    let entropy = frame_features::dictionary(values);
    bit_plan(&entropy, window, skew)
}

fn bit_plan(entropy: &BitWriter, window: u64, skew: u8) -> (Plan, Arc<GpuCodestream>) {
    let mut bits = BitWriter::new();
    bits.write_bits(0, skew).unwrap();
    frame_features::copy_bits(&mut bits, entropy.as_bytes(), 0, entropy.bit_len() as u64);
    let end = bits.bit_len() as u64;
    let bytes: Arc<[u8]> = bits.into_bytes().into();
    let source =
        Arc::new(GpuCodestream::from_shared(bytes.clone(), 0..bytes.len(), false).unwrap());
    let mut frame = jxl_gpu_bitstream::parse(
        include_bytes!("../../../../../../fixtures/gpu_gray8_lossless.jxl"),
        Default::default(),
    )
    .unwrap()
    .codestream_inventory(Default::default())
    .unwrap()
    .frames
    .remove(0);
    frame.width = 64;
    frame.height = 32;
    frame.sections.truncate(1);
    frame.sections[0].kind = FrameSectionKind::Single;
    frame.sections[0].bits = BitRange {
        offset: u64::from(skew),
        length: end - u64::from(skew),
    };
    (Plan::new(&source, &frame, None, window).unwrap(), source)
}

pub(super) fn backend() -> WgpuBackend {
    pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap()
}

pub(super) fn drain(backend: &WgpuBackend, expected: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while backend.transient_memory_budget().snapshot().reserved_bytes != expected
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        expected
    );
}

fn words(backend: &WgpuBackend, output: &entropy_program::DecodedProgram) -> Vec<u32> {
    read_words(backend, &output.commands, output.count)
}

fn read_words(backend: &WgpuBackend, buffer: &jxl_wgpu::GpuBufferLease, count: u32) -> Vec<u32> {
    planes::read(
        backend,
        &jxl_wgpu::GpuImageOutput {
            id: jxl_gpu_protocol::OutputId(0),
            layout: jxl_gpu_formats::ImageLayout::packed(
                jxl_gpu_protocol::Extent2d::new(count, 1),
                jxl_gpu_formats::PixelFormat::non_color(
                    jxl_gpu_formats::SampleKind::Unsigned,
                    32,
                    &[jxl_gpu_formats::Channel::X],
                ),
            )
            .unwrap(),
            buffer: buffer.clone(),
        },
    )
}

#[test]
fn spline_entropy_shader_validates() {
    let source = Params::SHADER
        .replace(
            "/*__JXL_MODULAR_ENTROPY_ABI__*/",
            include_str!("../../../modular_entropy_abi.wgsl"),
        )
        .replace(
            "/*__JXL_MODULAR_ENTROPY__*/",
            include_str!("../../../modular_entropy.wgsl"),
        );
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    for source in [include_str!("geometry.wgsl"), include_str!("render.wgsl")] {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}

pub(super) fn line_program(backend: &WgpuBackend, points: u32) -> entropy_program::DecodedProgram {
    repeated_line_program(backend, points, 1)
}

fn repeated_line_program(
    backend: &WgpuBackend,
    points: u32,
    copies: u32,
) -> entropy_program::DecodedProgram {
    let mut values = vec![copies - 1, 1, 2];
    values.extend(std::iter::repeat_n(0, (copies as usize - 1) * 2));
    values.push(0);
    let mut coefficients = [0; 128];
    coefficients[0] = pack(100);
    coefficients[32] = pack(8);
    coefficients[64] = pack(-3);
    coefficients[96] = pack(3);
    for _ in 0..copies {
        values.push(points - 1);
        if points > 1 {
            values.extend([pack(40), 0]);
        }
        values.extend(coefficients);
    }
    let (plan, source) = make_plan(&values, 40, 0);
    plan.submit(backend.clone(), source)
        .unwrap()
        .wait()
        .unwrap()
        .0
}

#[test]
fn spline_prefix_unary_and_lz77_run_across_fields_keep_the_exact_cursor() {
    let backend = backend();
    for (lz77, unary) in [(false, false), (false, true), (true, false)] {
        let mut bits = BitWriter::new();
        bits.write_bits(u64::from(lz77), 1).unwrap();
        if lz77 {
            bits.write_bits(0, 2).unwrap(); // Minimum symbol 224.
            bits.write_bits(0, 2).unwrap(); // Minimum length 3.
            bits.write_bits(8, 4).unwrap(); // Direct length values.
        }
        bits.write_bits(1, 1).unwrap();
        bits.write_bits(0, 2).unwrap(); // One cluster for every spline and distance context.
        bits.write_bits(1, 1).unwrap(); // Prefix coding.
        bits.write_bits(15, 4).unwrap();
        if unary {
            bits.write_bits(0, 1).unwrap();
        } else {
            bits.write_bits(1, 1).unwrap();
            bits.write_bits(if lz77 { 8 } else { 0 }, 4).unwrap();
            if lz77 {
                bits.write_bits(97, 8).unwrap();
            } // Alphabet 354.
            bits.write_bits(1, 2).unwrap(); // Simple histogram.
            bits.write_bits(1, 2).unwrap(); // Two symbols.
            let width = if lz77 { 9 } else { 1 };
            bits.write_bits(0, width).unwrap();
            bits.write_bits(if lz77 { 353 } else { 1 }, width).unwrap();
            if lz77 {
                bits.write_bits(0, 1).unwrap(); // First literal: spline count minus one.
                bits.write_bits(1, 1).unwrap(); // Copy the following 132 fields.
                bits.write_bits(0, 1).unwrap(); // Distance one, including overlapping history.
            } else {
                for _ in 0..133 {
                    bits.write_bits(0, 1).unwrap();
                }
            }
        }
        for skew in [0, 3, 7] {
            let (plan, source) = bit_plan(&bits, 40, skew);
            let end = plan.token_start + u64::from(plan.windows.get(0).unwrap().stream_token_end);
            assert_eq!(plan.history_words != 0, lz77);
            let (output, _) = plan
                .submit(backend.clone(), source)
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(output.end, end);
            let mut expected = vec![0; 140];
            expected[..6].copy_from_slice(&[1, 0, 140, 0, 140, 1]);
            assert_eq!(words(&backend, &output), expected);
            drop(output);
            drain(&backend, 0);
        }
    }
}

#[test]
fn geometry_tiles_and_ordered_raster_match_a_constant_straight_spline() {
    use jxl_gpu_protocol::Extent2d;
    use jxl_wgpu::{GpuBufferLease, ResidentF32Plane, ResidentStorageBinding};
    use wgpu::util::DeviceExt;

    let backend = backend();
    let extent = Extent2d::new(64, 32);
    for copies in [1, 16] {
        let plan = GeometryPlan::new(
            repeated_line_program(&backend, 2, copies),
            extent,
            [0.0, 1.0],
            &backend.device().limits(),
        )
        .unwrap();
        let mut pending: PendingGeometry = plan.submit(backend.clone()).unwrap();
        let cache = pollster::block_on(std::future::poll_fn(|cx| pending.poll(cx)))
            .unwrap()
            .unwrap();
        assert!(pending.submissions >= 2);
        if copies > 1 {
            assert!(pending.submissions > 4);
            assert!(cache.max_population > 256);
            assert!(cache.scratch_bytes() > 48);
        }
        drop(pending);
        let records = read_words(&backend, &cache.records, (cache.records.size() / 4) as u32);
        assert_eq!(records.len(), 41 * 12 * copies as usize);
        for (index, record) in records.as_chunks::<12>().0.iter().enumerate() {
            let values: Vec<_> = record[..8].iter().copied().map(f32::from_bits).collect();
            let expected = [
                (index % 41) as f32 + 1.0,
                2.0,
                1.0 / 0.9999,
                0.9999 / 4.0,
                0.42,
                0.6,
                0.39,
                0.0,
            ];
            for (actual, expected) in values.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 0.0001,
                    "splat {index}: {values:?} != {expected}"
                );
            }
        }
        let tiles = read_words(&backend, &cache.tiles, (cache.tiles.size() / 4) as u32);
        let references = read_words(
            &backend,
            &cache.references,
            (cache.references.size() / 4) as u32,
        );
        let tile_count = cache.tile_count as usize;
        assert_eq!(tiles[2 * tile_count] as usize, references.len());
        for tile in 0..tile_count {
            let start = tiles[tile_count + tile] as usize;
            let end = tiles[tile_count + tile + 1] as usize;
            assert_eq!(end - start, tiles[tile] as usize);
            assert!(
                references[start..end]
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            );
        }
        let mut permit = backend
            .transient_memory_budget()
            .try_reserve(3 * 64 * 32 * 4 + cache.scratch_bytes())
            .unwrap();
        let outputs: [GpuBufferLease; 3] = std::array::from_fn(|_| {
            GpuBufferLease::from_tracked(
                backend
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("spline raster test plane"),
                        contents: &vec![0; 64 * 32 * 4],
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    }),
                permit.split_off(64 * 32 * 4).unwrap(),
            )
        });
        let plane_bindings = outputs.each_ref().map(|buffer| ResidentF32Plane {
            storage: ResidentStorageBinding {
                buffer: buffer.as_wgpu_buffer(),
                offset: 0,
                size: std::num::NonZeroU64::new(buffer.size()).unwrap(),
            },
            width: 64,
            height: 32,
            stride: 64,
        });
        let mut encoder = backend.device().create_command_encoder(&Default::default());
        let uniforms = cache
            .record(backend.device(), &mut encoder, plane_bindings)
            .unwrap();
        let work = super::super::submission::submit_recorded(
            &backend,
            encoder,
            outputs,
            vec![
                cache.records.clone(),
                cache.references.clone(),
                cache.tiles.clone(),
            ],
            (uniforms, cache.clone()),
            permit,
            backend.submission_poller().try_reserve().unwrap(),
        )
        .unwrap();
        let outputs = work.wait().unwrap();
        for (channel, output) in outputs.iter().enumerate() {
            let actual = read_words(&backend, output, 64 * 32);
            for y in 0..32 {
                for x in 0..64 {
                    let expected: f64 = records
                        .as_chunks::<12>()
                        .0
                        .iter()
                        .filter(|r| x >= r[8] && y >= r[9] && x < r[10] && y < r[11])
                        .map(|r| {
                            let dx = f64::from(x) - f64::from(f32::from_bits(r[0]));
                            let dy = f64::from(y) - f64::from(f32::from_bits(r[1]));
                            let distance = dx.hypot(dy);
                            let inverse = f64::from(f32::from_bits(r[2]));
                            let erf = |v: f64| {
                                let a = v.abs();
                                let d = (((0.0777394369 * a + 0.000205260015) * a + 0.232120216)
                                    * a
                                    + 0.277820801)
                                    * a
                                    + 1.0;
                                v.signum() * (1.0 - 1.0 / d.powi(4))
                            };
                            let factor = erf((distance * 0.5
                                + std::f64::consts::FRAC_1_SQRT_2 / 2.0)
                                * inverse)
                                - erf((distance * 0.5 - std::f64::consts::FRAC_1_SQRT_2 / 2.0)
                                    * inverse);
                            f64::from(f32::from_bits(r[4 + channel]))
                                * f64::from(f32::from_bits(r[3]))
                                * factor
                                * factor
                        })
                        .sum();
                    let value = f32::from_bits(actual[(y * 64 + x) as usize]);
                    assert!(
                        (f64::from(value) - expected).abs() < 0.000002 * f64::from(copies),
                        "channel {channel} at {x},{y}: {value} != {expected}"
                    );
                }
            }
        }
        drop((outputs, cache));
        drain(&backend, 0);
    }
}

#[test]
fn spline_geometry_cancellation_and_empty_cache_release_all_storage() {
    let backend = backend();
    for finish in [false, true] {
        let plan = GeometryPlan::new(
            line_program(&backend, 2),
            jxl_gpu_protocol::Extent2d::new(64, 32),
            [0.0, 1.0],
            &backend.device().limits(),
        )
        .unwrap();
        let mut pending = plan.submit(backend.clone()).unwrap();
        if finish {
            while pending.submissions == 1 {
                assert!(pending.complete_submission().unwrap().is_none());
            }
        }
        drop(pending);
        drain(&backend, 0);
    }
    let plan = GeometryPlan::new(
        line_program(&backend, 1),
        jxl_gpu_protocol::Extent2d::new(64, 32),
        [0.0, 1.0],
        &backend.device().limits(),
    )
    .unwrap();
    assert!(
        plan.submit(backend.clone())
            .unwrap()
            .wait()
            .unwrap()
            .0
            .is_none()
    );
    drain(&backend, 0);
}

#[test]
fn quantized_splines_preserve_coefficients_points_and_the_exact_bounded_cursor() {
    let backend = backend();
    let first: Vec<_> = (0..128)
        .map(|i| if i == 127 { i32::MIN } else { i - 64 })
        .collect();
    let second: Vec<_> = (0..128).map(|i| 3 * i - 100).collect();
    let mut values = vec![
        1,
        9,
        7,
        pack(-6),
        pack(4),
        pack(-8),
        2,
        pack(5),
        pack(-1),
        pack(-2),
        pack(4),
    ];
    values.extend(first.iter().copied().map(pack));
    values.push(0);
    values.extend(second.iter().copied().map(pack));
    let mut expected = vec![0; 280];
    expected[..4].copy_from_slice(&[2, (-8i32) as u32, 276, 0]);
    expected[4..8].copy_from_slice(&[276, 3, 9, 7]);
    expected[140..144].copy_from_slice(&[280, 1, 3, 11]);
    for (offset, coefficients) in [(8, first), (144, second)] {
        for (out, value) in expected[offset..offset + 128].iter_mut().zip(coefficients) {
            *out = value as u32;
        }
    }
    expected[276..].copy_from_slice(&[14, 6, 17, 9]);
    for window in [1 << 20, 40] {
        for skew in [0, 3, 7] {
            let (plan, source) = make_plan(&values, window, skew);
            let end = plan.token_start + u64::from(plan.windows.get(0).unwrap().stream_token_end);
            let (output, submissions) = plan
                .submit(backend.clone(), source)
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(output.end, end);
            assert_eq!(output.count, 280);
            assert_eq!(output.stride, 1);
            assert_eq!(words(&backend, &output), expected);
            assert!(submissions >= if window == 40 { 4 } else { 2 });
            drop(output);
            drain(&backend, 0);
        }
    }
}

#[test]
fn spline_count_replay_limits_and_cancellation_are_accounted() {
    let backend = backend();
    let mut values = vec![39];
    values.extend(std::iter::repeat_n(0, 80));
    values.push(0);
    values.extend(std::iter::repeat_n(0, 40 * 129));
    let (plan, source) = make_plan(&values, 40, 0);
    let output_bytes = u64::from(4 + 40 * HEADER_WORDS) * 4;
    for phase in 0..3 {
        let mut pending: Pending = plan
            .clone()
            .submit(backend.clone(), source.clone())
            .unwrap();
        if phase != 0 {
            while pending.submissions < 3 {
                assert!(pending.complete_submission().unwrap().is_none());
            }
        }
        if phase == 2 {
            drop(pending.wait().unwrap());
        } else {
            drop(pending);
        }
        drain(&backend, 0);
    }
    let (output, _) = plan
        .submit(backend.clone(), source)
        .unwrap()
        .wait()
        .unwrap();
    drain(&backend, output_bytes);
    assert_eq!(output.count, 4 + 40 * HEADER_WORDS);
    drop(output);
    drain(&backend, 0);

    for (prefix, resource) in [
        (vec![1024], Some(SplineResource::ControlPoints)),
        (
            vec![0, u32::MAX, 0],
            Some(SplineResource::CoordinateMagnitude),
        ),
        (
            vec![0, 0, 0, 0, 1, pack(1 << 30), 0],
            Some(SplineResource::DeltaMagnitude),
        ),
        (vec![0, 0, 0, 0, 1, 0, 0], None),
        (
            vec![0, (1 << 23) - 2, 0, 0, 1, pack(3), 0],
            Some(SplineResource::CoordinateMagnitude),
        ),
    ] {
        let mut values = prefix;
        values.extend(std::iter::repeat_n(0, 128));
        let (plan, source) = make_plan(&values, 40, 0);
        let error = plan
            .submit(backend.clone(), source)
            .unwrap()
            .wait()
            .unwrap_err();
        if let Some(expected) = resource {
            assert!(
                matches!(error, Error::SplineResourceLimit { resource, .. } if resource == expected),
                "{error:?}"
            );
        } else {
            assert!(
                matches!(error, Error::SplineGeometry { code: 15 }),
                "{error:?}"
            );
        }
        drain(&backend, 0);
    }
}
