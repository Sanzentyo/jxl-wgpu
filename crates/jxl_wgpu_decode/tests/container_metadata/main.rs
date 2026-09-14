#![cfg(not(target_arch = "wasm32"))]

use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use jxl_gpu_bitstream::metadata::{
    BrotliOptions, EXIF, JUMBF, Metadata, MetadataBox, MetadataCollector, MetadataCompression,
    MetadataLimits, MetadataSelection, XMP,
};
use jxl_gpu_bitstream::{ContainerStreamScanner, FragmentedContainerWriter};
use jxl_gpu_formats::{PixelFormat, RgbChannelOrder};
use jxl_test_support::{fixtures, gpu::planes};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AnimationMetadata, FrameMetadata, FrameProgression, GpuDecoder, GpuOutputRequest,
    ImageSelection, OrientationPolicy, WgpuDecodeEngine,
};

type Presentation = (
    AnimationMetadata,
    Vec<(FrameMetadata, Option<FrameProgression>, Vec<u32>)>,
);

fn payloads(compressed: bool) -> Metadata {
    let limits = MetadataLimits::default();
    let compression = if compressed {
        MetadataCompression::Brotli(BrotliOptions::default())
    } else {
        MetadataCompression::None
    };
    let exif = [
        0, 0, 0, 0, b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 1, 3, 0, 1, 0, 0, 0, 8, 0, 0, 0, 0,
        0, 0, 0,
    ];
    let xmp = b"<x:xmpmeta xmlns:x='adobe:ns:meta/' xmlns:tiff='http://ns.adobe.com/tiff/1.0/' tiff:Orientation='8' tiff:ImageWidth='9000' tiff:ImageLength='7000'/>";
    let mut result = Metadata::default();
    for (kind, payload) in [
        (EXIF, &exif[..]),
        (XMP, &xmp[..]),
        (JUMBF, b"opaque JUMBF image claims"),
        (*b"priv", b"\0\xffunknown data"),
    ] {
        result
            .push(
                MetadataBox::new(kind, payload, compression, limits).unwrap(),
                limits,
            )
            .unwrap();
    }
    result
}

fn source(name: &str) -> Vec<u8> {
    jxl_test_support::offline::unhex(
        &std::fs::read_to_string(
            jxl_test_support::decoder_directory().join(format!("test-data/{name}.jxl.hex")),
        )
        .unwrap(),
    )
}

fn render(
    backend: &WgpuBackend,
    data: &[u8],
    selection: ImageSelection,
    orientation: OrientationPolicy,
    bounded: bool,
) -> (Presentation, Metadata) {
    let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    let engine = if bounded {
        engine.with_stream_window_limit(NonZeroU64::new(256).unwrap())
    } else {
        engine
    };
    let decoder = GpuDecoder::new(engine);
    let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_image_selection(selection)
    .with_orientation_policy(orientation)
    .with_progressive_output(true)
    .with_max_frame_slots(NonZeroUsize::new(64).unwrap());
    let (mut session, metadata) = if bounded {
        let mut stream = decoder.stream(request).unwrap();
        let mut collector =
            MetadataCollector::new(MetadataSelection::All, MetadataLimits::default());
        let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
        for chunk in data.chunks(43) {
            for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
                collector.push_transport_event(&event).unwrap();
                stream.push_transport_event(&event).unwrap();
            }
        }
        for event in transport.finish_input().unwrap() {
            collector.push_transport_event(&event).unwrap();
            stream.push_transport_event(&event).unwrap();
        }
        (stream.finish().unwrap(), collector.finish().unwrap())
    } else {
        let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
        let metadata = parsed
            .metadata(&MetadataSelection::All, MetadataLimits::default())
            .unwrap();
        (decoder.open(data, request).unwrap(), metadata)
    };
    let image_metadata = session.metadata().clone();
    let mut held = Vec::new();
    while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
        let words = planes::read(backend, &update.output().outputs[0]);
        held.push((update, words));
    }
    drop(session);
    let presentations = held
        .into_iter()
        .map(|(update, words)| {
            assert_eq!(planes::read(backend, &update.output().outputs[0]), words);
            (update.metadata.clone(), update.progression(), words)
        })
        .collect();
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0
    );
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
    ((image_metadata, presentations), metadata)
}

#[test]
fn container_metadata_preserves_gpu_pixels_timing_progression_and_codestream_precedence() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    eprintln!("container metadata adapter: {:?}", backend.adapter_info());
    let mut cases = [
        "noise/modular_rgb_group256",
        "noise/vardct_rgb_257x17",
        "noise/mixed_frames",
    ]
    .map(|name| (name.to_owned(), source(name), ImageSelection::Main))
    .to_vec();
    for selection in [ImageSelection::Main, ImageSelection::Preview] {
        cases.push((
            format!("preview {selection:?}"),
            source("preview/animation_modular"),
            selection,
        ));
    }
    cases.extend(
        fixtures::embedded_icc::cases()
            .map(|case| (case.name(), case.bytes(), ImageSelection::Main)),
    );
    for case in fixtures::hdr::cases()
        .into_iter()
        .filter(|case| !case.sequence)
        .step_by(12)
    {
        cases.push((case.name.clone(), case.bytes(), ImageSelection::Main));
    }
    let mut outputs = 0;
    let mut words = 0;
    for (name, source, selection) in cases {
        eprintln!("container metadata source: {name}");
        let source = jxl_gpu_bitstream::parse(&source, Default::default()).unwrap();
        let oriented = if name.starts_with("preview ") {
            fixtures::preview::orient(source.codestream(), 6)
        } else {
            source.codestream().to_vec()
        };
        for orientation in [OrientationPolicy::Apply, OrientationPolicy::Keep] {
            let baseline = render(&backend, &oriented, selection, orientation, false).0;
            for variant in 0..3 {
                let mut metadata = payloads(variant != 0);
                if variant == 2 {
                    metadata
                        .replace(EXIF, None, MetadataLimits::default())
                        .unwrap();
                    metadata
                        .replace(
                            XMP,
                            Some(
                                MetadataBox::new(
                                    XMP,
                                    b"replacement bytes",
                                    MetadataCompression::None,
                                    MetadataLimits::default(),
                                )
                                .unwrap(),
                            ),
                            MetadataLimits::default(),
                        )
                        .unwrap();
                }
                let wrapped = if variant == 1 {
                    let mut writer = FragmentedContainerWriter::new();
                    for item in metadata.boxes() {
                        writer.push_box(item.as_container_box()).unwrap();
                    }
                    writer.push_fragment(&oriented[..1], false).unwrap();
                    writer.push_fragment(&oriented[1..], true).unwrap();
                    writer.finish().unwrap()
                } else {
                    metadata.write_container(&oriented).unwrap()
                };
                for bounded in [false, true] {
                    let (actual, retained) =
                        render(&backend, &wrapped, selection, orientation, bounded);
                    assert_eq!(retained, metadata, "{name} metadata");
                    assert_eq!(
                        actual, baseline,
                        "{name} {orientation:?} variant {variant} bounded {bounded}"
                    );
                    outputs += actual.1.len();
                    words += actual
                        .1
                        .iter()
                        .map(|(_, _, values)| values.len())
                        .sum::<usize>();
                }
            }
        }
    }
    eprintln!(
        "container metadata GPU: {outputs} immutable presentations / {words} exact F32 words passed"
    );
}
