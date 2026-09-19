use super::*;
use jxl_gpu_formats::{ColorSample, ColorStorage};
use jxl_gpu_protocol::icc::{IccProfile, IccRenderingIntent};
use jxl_wgpu_decode::{AlphaOutputPolicy, gain_map::GainMapRendering};

mod cases;
mod packing;

const TARGETS: [(&str, usize); 13] = [
    ("gamma_v4", 3),
    ("gray", 1),
    ("intents/rgb_2", 3),
    ("intents/rgb_8", 3),
    ("mpe/identity", 3),
    ("lut/lut8_xyz_3", 3),
    ("lut/lut16_lab_3", 3),
    ("lut/ab_lab_1", 1),
    ("lut/lut16_v2_xyz_3", 3),
    ("lut/lut16_channels2", 2),
    ("lut/ab_lab_4", 4),
    ("lut/ab_channels5", 5),
    ("lut/ab_channels15", 15),
];
const INTENTS: [IccRenderingIntent; 4] = [
    IccRenderingIntent::Perceptual,
    IccRenderingIntent::Relative,
    IccRenderingIntent::Saturation,
    IccRenderingIntent::Absolute,
];

fn profile(name: &str) -> IccProfile {
    IccProfile::parse(
        std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../jxl_wgpu/test-data/icc")
                .join(format!("{name}.icc")),
        )
        .unwrap()
        .into(),
        Default::default(),
    )
    .unwrap()
}

#[test]
fn gain_renditions_match_native_and_independent_icc_profiles_in_requested_storage() {
    let Some(native) = native::Oracle::new() else {
        return;
    };
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    eprintln!("gain ICC GPU: {:?}", backend.adapter_info());
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let cases = cases::all();
    assert_eq!(cases.len(), 160);
    let mut outputs = 0;
    let mut components = 0;
    let mut native_calls = 0;
    let mut retained = Vec::new();
    let mut targets = std::collections::BTreeSet::new();
    for (index, case) in cases.into_iter().enumerate() {
        // A declared pairing, not a Cartesian product of all source/profile combinations.
        let (target, channels) = TARGETS[index % TARGETS.len()];
        let profile = profile(target);
        assert_eq!(
            profile.header().device_space.device_channels(),
            Some(channels as u8)
        );
        targets.insert(target);
        eprintln!("gain ICC {} -> {target}", case.name);
        for intent in INTENTS {
            let expected = native.icc(target, intent as u32, &case.pcs);
            assert_eq!(
                expected.len(),
                case.extent.into_iter().product::<usize>() * channels
            );
            native_calls += 1;
            for (configuration, (sample, storage)) in [
                (ColorSample::F32, ColorStorage::Interleaved),
                (ColorSample::F32, ColorStorage::Planar),
                (ColorSample::U8, ColorStorage::Interleaved),
                (ColorSample::U8, ColorStorage::Planar),
            ]
            .into_iter()
            .enumerate()
            {
                let packing = packing::Packing {
                    sample,
                    storage,
                    associated: (index + intent as usize + configuration).is_multiple_of(2),
                    apply_orientation: configuration.is_multiple_of(2),
                };
                let request = packing
                    .request(profile.clone(), channels)
                    .with_icc_rendering_intent(intent);
                let frame = pollster::block_on(decoder.decode_gain_map(
                    &case.bytes,
                    request,
                    case.rendering,
                    Default::default(),
                ))
                .unwrap_or_else(|e| panic!("{} -> {target} {intent:?}: {e}", case.name));
                let output = &frame.output().outputs[0];
                let bytes = planes::read_bytes(&backend, output);
                components +=
                    packing.check(&case, channels, &expected, &output.layout, &bytes, intent);
                outputs += 1;
                if index.is_multiple_of(40)
                    && configuration == 0
                    && intent == IccRenderingIntent::Relative
                {
                    retained.push((frame, bytes));
                }
            }
        }
    }
    assert_eq!(targets.len(), 13);
    assert_eq!(native_calls, 640);
    assert_eq!(outputs, 2560);
    assert!(components > 2_000_000);
    assert_eq!(retained.len(), 4);
    for (frame, expected) in &retained {
        assert_eq!(
            planes::read_bytes(&backend, &frame.output().outputs[0]),
            *expected
        );
    }
    drop(retained);
    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    eprintln!(
        "gain ICC: {outputs} GPU outputs, {components} components, {native_calls} live native ICC comparisons; retained outputs stable and all bytes released"
    );
}

#[test]
fn icc_baseline_selection_preserves_ordinary_output_and_skips_unused_map() {
    use jxl_wgpu_decode::gain_map::GainMapRendition;
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let base = std::fs::read(directory().join("case_0.jxl")).unwrap();
    let mut map = reference::Map::load(0);
    map.code = vec![0xff, 0x0a];
    map.metadata.base_hdr_headroom.numerator = 0;
    map.metadata.alternate_hdr_headroom.numerator = 2;
    let bytes = map.container(&base);
    for (target, channels) in [TARGETS[1], TARGETS[4], TARGETS[12]] {
        for storage in [ColorStorage::Interleaved, ColorStorage::Planar] {
            let request = packing::Packing {
                sample: ColorSample::F32,
                storage,
                associated: true,
                apply_orientation: true,
            }
            .request(profile(target), channels);
            let mut session = decoder.open(&base, request.clone()).unwrap();
            let original = session.next_frame().unwrap().unwrap();
            let frame = pollster::block_on(decoder.decode_gain_map(
                &bytes,
                request,
                GainMapRendering {
                    rendition: GainMapRendition::DisplayHeadroom(0.0),
                    ..Default::default()
                },
                Default::default(),
            ))
            .unwrap();
            assert_eq!(
                original.output().outputs[0].layout,
                frame.output().outputs[0].layout
            );
            assert_eq!(
                planes::read_bytes(&backend, &original.output().outputs[0]),
                planes::read_bytes(&backend, &frame.output().outputs[0])
            );
        }
    }
    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn unsupported_icc_adaptation_is_rejected_before_image_admission() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let source = std::fs::read(directory().join("case_0.jxl")).unwrap();
    let memory = backend.transient_memory_budget();
    let blocker = memory
        .try_reserve(memory.snapshot().available_bytes)
        .unwrap();
    let request = packing::Packing {
        sample: ColorSample::F32,
        storage: ColorStorage::Planar,
        associated: false,
        apply_orientation: true,
    }
    .request(profile("gamma_v4"), 3)
    .with_white_point_adaptation(jxl_gpu_protocol::WhitePointAdaptation::None);
    assert!(matches!(
        pollster::block_on(decoder.decode_alternate(&source, request, Default::default())),
        Err(jxl_wgpu_decode::Error::UnsupportedOutputFormat(message))
            if message == "ICC color conversion currently requires Bradford adaptation"
    ));
    assert_eq!(memory.snapshot().reserved_bytes, blocker.bytes());
    drop(blocker);
    assert_eq!(memory.snapshot().reserved_bytes, 0);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}
