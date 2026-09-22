//! Spectral/refinement entropy, native pass images and GPU ownership.

use super::super::dispatch::VarDctBackend;
mod delivery;
mod images;
mod limits;
mod probes;

use super::*;
use crate::{
    AnimationHeader, Determinism, EncodeProfile, FrameEncodeRequest, FrameIndex, FrameOptions,
    GpuEncodeBackend, GpuFrameSource, ProgressiveDownsampling, ProgressivePass, ProgressivePlan,
    VarDctGroupOrder,
};

fn plan(passes: &[(u8, u8)]) -> ProgressivePlan {
    ProgressivePlan::new(
        passes
            .iter()
            .map(|&(square, shift)| ProgressivePass {
                coefficient_square: std::num::NonZeroU8::new(square).unwrap(),
                shift,
            })
            .collect(),
    )
    .unwrap()
}

pub(super) fn combined() -> ProgressivePlan {
    plan(&[(2, 2), (4, 3), (4, 0), (8, 1), (8, 0)])
}

pub(super) fn maximum() -> ProgressivePlan {
    plan(&[
        (2, 3),
        (2, 2),
        (2, 1),
        (2, 0),
        (3, 3),
        (3, 2),
        (3, 1),
        (3, 0),
        (4, 0),
        (6, 0),
        (8, 0),
    ])
}

#[test]
fn progressive_configuration_bounds_and_tiled_passes_interoperate() {
    let context = test_context().expect("actual GPU required for progressive encoding");
    for invalid in [
        vec![],
        vec![(9, 0)],
        vec![(8, 1)],
        vec![(4, 4), (8, 0)],
        vec![(4, 1), (4, 2), (8, 0)],
        vec![(4, 0), (2, 0), (8, 0)],
        vec![(8, 3); 12],
    ] {
        assert!(
            ProgressivePlan::new(
                invalid
                    .into_iter()
                    .map(|(square, shift)| ProgressivePass {
                        coefficient_square: std::num::NonZeroU8::new(square).unwrap(),
                        shift,
                    })
                    .collect()
            )
            .is_err()
        );
    }
    let source = padded_rgb_source_sized(&context, 13, 21, &reference::pattern(13, 21));
    let baseline = TiledVarDctEncoder::new(context.clone())
        .unwrap()
        .encode(source.clone())
        .unwrap();
    let reference = decode_rgb8_sized(&baseline, 13, 21);
    for progressive in [
        plan(&[(1, 3), (8, 0)]),
        plan(&[(2, 0), (4, 0), (8, 0)]),
        plan(&[(8, 3), (8, 1), (8, 0)]),
        combined(),
        maximum(),
    ] {
        let encoder = TiledVarDctEncoder::new_with_config(
            context.clone(),
            VarDctConfig {
                progressive,
                ..Default::default()
            },
        )
        .unwrap();
        let bytes = encoder.encode(source.clone()).unwrap();
        assert_eq!(decode_rgb8_sized(&bytes, 13, 21), reference);
        quantization::assert_decoders_agree(&bytes, 13, 21);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}

fn fragments(
    context: &WgpuContext,
    source: &BufferImageSource,
    strategy: VarDctStrategy,
    config: &VarDctConfig,
) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let backend = VarDctBackend::new_with_config(context, strategy, config.clone()).unwrap();
    let extent = source.layout.extent;
    let request = FrameEncodeRequest {
        frame_index: FrameIndex::new(0),
        is_last: true,
        profile: EncodeProfile::VarDct {
            quantization: config.quantization,
        },
        progressive: config.progressive.clone(),
        minimum_determinism: Determinism::SameDevice,
        animation: AnimationHeader::Still,
        canvas_width: extent.width,
        canvas_height: extent.height,
        options: FrameOptions::default(),
    };
    let (words, lengths, artifact) = backend
        .submit(context, GpuFrameSource::Buffer(source.clone()), &request)
        .unwrap()
        .wait_with_ac_fragments_for_test()
        .unwrap();
    assert_eq!(
        artifact.packets.layout.passes() as usize,
        config.progressive.passes().len()
    );
    let mut bytes = image_header(extent.width, extent.height)
        .unwrap()
        .bytes()
        .to_vec();
    bytes.extend_from_slice(assemble_frame(artifact.packets).unwrap().bytes());
    (words, lengths, bytes)
}

fn coefficients(
    words: &[u32],
    bit_len: u32,
    order: &[usize],
    permutations: Option<[&[u32]; 3]>,
) -> Vec<[i64; 3]> {
    let entropy = HfEntropyPlan::single_cluster_prefix().unwrap();
    let mut output = BitWriter::new();
    super::super::entropy::write_prefix_config(&mut output, &entropy.code, 1).unwrap();
    super::super::bitstream::append_gpu_fragment(&mut output, words, 0, bit_len).unwrap();
    let length = output.bit_len();
    let bytes = output.into_bytes();
    let mut bits = jxl_bitstream::Bitstream::new(&bytes);
    let mut decoder = jxl_coding::Decoder::parse(&mut bits, 1).unwrap();
    decoder.begin(&mut bits).unwrap();
    let mut result = vec![[0; 3]; order.len()];
    for channel in [1, 0, 2] {
        let mut count = decoder.read_varint(&mut bits, 0).unwrap();
        assert!(count as usize <= order.len() - order.len() / 64);
        for rank in order.len() / 64..order.len() {
            let packed = if count == 0 {
                0
            } else {
                decoder.read_varint(&mut bits, 0).unwrap()
            };
            count -= u32::from(packed != 0);
            let index = order[permutations.map_or(rank, |orders| orders[channel][rank] as usize)];
            result[index][channel] = i64::from(packed >> 1) ^ -i64::from(packed & 1);
        }
        assert_eq!(count, 0);
    }
    decoder.finalize().unwrap();
    assert_eq!(bits.num_read_bits(), length);
    result
}

fn native_rgb8(directory: &Path, bytes: &[u8], width: usize, height: usize) -> Vec<u8> {
    let input = directory.join("progressive.jxl");
    let output = directory.join("progressive.ppm");
    fs::write(&input, bytes).unwrap();
    let result = Command::new("djxl")
        .arg(&input)
        .arg(&output)
        .args(["--num_threads=0", "--quiet"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    read_ppm_rgb8(&output, width, height)
}

#[test]
fn all_strategies_split_spectral_and_quantized_coefficients_exactly() {
    let (device, queue, info) = test_device().expect("actual GPU required for progressive AC");
    let backend = WgpuBackend::from_device(
        device.as_ref().clone(),
        queue.as_ref().clone(),
        info,
        WgpuBackendConfig::default(),
    )
    .unwrap();
    let context = WgpuContext::from_backend(&backend);
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let readback = ImageReadbackPipeline::new(&backend);
    let directory = oracle_directory();
    fs::create_dir_all(&directory).unwrap();
    let mut oracles = native::native_oracles();
    for (strategy, oracle) in VarDctStrategy::ALL.into_iter().zip(&mut oracles) {
        let extent = strategy.pixel_extent();
        let (w, h) = (extent.width as usize, extent.height as usize);
        let pixels = reference::pattern(w, h);
        let source = padded_rgb_source_sized(&context, w, h, &pixels);
        let config = VarDctConfig {
            coefficient_orders: orders::selected([strategy]),
            dequant_matrices: raw_matrices::selected([strategy]),
            lf_metadata: custom_lf_metadata(),
            ..Default::default()
        };
        oracle.dequant = raw_matrices::oracle_scales(strategy);
        let (baseline_words, baseline_lengths, baseline) =
            fragments(&context, &source, strategy, &config);
        native::check_ac(
            &baseline_words,
            baseline_lengths[0],
            &native::forward(&pixels, w, h, oracle),
            oracle,
            config.clone(),
        );
        let baseline_coefficients = coefficients(
            &baseline_words,
            baseline_lengths[0],
            &oracle.order,
            config.coefficient_orders.permutations(strategy),
        );
        let baseline_pixels = decode_rgb8_sized(&baseline, w, h);
        for progressive in [
            plan(&[(2, 0), (4, 0), (8, 0)]),
            plan(&[(8, 3), (8, 1), (8, 0)]),
            combined(),
        ] {
            let config = VarDctConfig {
                progressive,
                ..config.clone()
            };
            let (words, lengths, bytes) = fragments(&context, &source, strategy, &config);
            assert_eq!(lengths.len(), config.progressive.passes().len());
            let stride = words.len() / lengths.len();
            let mut cumulative = vec![[0i64; 3]; w * h];
            for (pass, &length) in lengths.iter().enumerate() {
                let values = coefficients(
                    &words[pass * stride..(pass + 1) * stride],
                    length,
                    &oracle.order,
                    config.coefficient_orders.permutations(strategy),
                );
                let shift = config.progressive.passes()[pass].shift;
                for (index, values) in values.iter().enumerate() {
                    let (x, y) = (index % w.max(h), index / w.max(h));
                    let best = config.progressive.passes()[..=pass]
                        .iter()
                        .filter(|spec| {
                            x < w.max(h) / 8 * usize::from(spec.coefficient_square.get())
                                && y < w.min(h) / 8 * usize::from(spec.coefficient_square.get())
                        })
                        .map(|spec| spec.shift)
                        .min();
                    for channel in 0..3 {
                        cumulative[index][channel] += values[channel] * (1i64 << shift);
                        let expected = best.map_or(0, |bits| {
                            let step = 1i64 << bits;
                            baseline_coefficients[index][channel] / step * step
                        });
                        assert_eq!(
                            cumulative[index][channel], expected,
                            "{strategy:?}, pass {pass}, coefficient {index}, channel {channel}"
                        );
                    }
                }
            }
            assert_eq!(cumulative, baseline_coefficients);
            let rust = decode_rgb8_sized(&bytes, w, h);
            assert_eq!(
                rust, baseline_pixels,
                "{strategy:?}: final progressive pixels"
            );
            assert!(max_abs_error(&native_rgb8(&directory, &bytes, w, h), &rust) <= 1);
            let mut session = decoder
                .open(
                    &bytes,
                    GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
                )
                .unwrap();
            let frame = session.next_frame().unwrap().unwrap();
            let gpu = readback
                .submit(frame.output())
                .unwrap()
                .wait()
                .unwrap()
                .frame
                .outputs[0]
                .bytes
                .clone();
            assert!(
                max_abs_error(&gpu, &rust) <= 1,
                "{strategy:?}: GPU final pixels"
            );
            drop(frame);
            assert!(session.next_frame().unwrap().is_none());
        }
    }
    fs::remove_dir_all(directory).unwrap();
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
