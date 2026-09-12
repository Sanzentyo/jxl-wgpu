#![cfg(not(target_arch = "wasm32"))]
//! Engine-boundary tests: no pixel codec or GPU device is needed to audit retained source ranges.

use std::num::{NonZeroU64, NonZeroUsize};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{
    CodestreamInventory, ContainerBox, ContainerStreamEvent, ContainerStreamScanner,
    FragmentedContainerWriter, StreamSlice,
};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu_decode::{
    AnimationMetadata, DecodeProfile, Error, GpuCodestream, GpuDecoder, GpuOutputRequest,
    GpuPendingFrame, GpuSubmissionEngine, GpuSubmissionSession, ImageSelection,
    ImageSelectionError, ImageSourceInventory, IncrementalInputBudget, IncrementalInputBudgetError,
    ModularChannels, ModularGrouping, ModularPredictionProfile, ModularPredictor,
    PreparedGpuSession, Result, SelectedImageInventory, SubmittedGpuFrame,
};

#[derive(Default)]
struct Engine {
    opened: Mutex<Vec<Weak<Source>>>,
    reject_next: AtomicBool,
}

struct Source {
    bytes: GpuCodestream,
    inventory: SelectedImageInventory,
}

struct Session(Arc<Source>);
struct Pending;

impl GpuPendingFrame for Pending {
    type Frame = ();
    fn wait(self) -> Result<SubmittedGpuFrame<()>> {
        unreachable!()
    }
    fn poll_complete(
        self: Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<()>>> {
        unreachable!()
    }
}

impl GpuSubmissionSession for Session {
    type Frame = ();
    type Pending = Pending;
    fn submit_next(&mut self) -> Result<Option<Pending>> {
        // This mock audits only opening/ownership, never manufactures decoded pixels.
        assert!(self.0.bytes.logical_bytes() > 0);
        unreachable!()
    }
}

impl GpuSubmissionEngine for Engine {
    type Session = Session;
    fn open(
        &self,
        bytes: GpuCodestream,
        request: &GpuOutputRequest,
        inventory: SelectedImageInventory,
    ) -> Result<PreparedGpuSession<Session>> {
        assert_eq!(request.image_selection(), inventory.selection());
        if self.reject_next.swap(false, Ordering::AcqRel) {
            return Err(Error::backend("injected engine admission failure"));
        }
        let header = &inventory.reconstruction_inventory().image_header;
        let metadata = AnimationMetadata::still(Extent2d::new(header.width, header.height));
        let source = Arc::new(Source { bytes, inventory });
        self.opened.lock().unwrap().push(Arc::downgrade(&source));
        Ok(PreparedGpuSession::new(
            DecodeProfile::Modular {
                sample_bit_depth: jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample: 8 },
                channels: ModularChannels::Gray.into(),
                prediction: ModularPredictionProfile::Fixed {
                    predictor: ModularPredictor::Zero,
                },
                grouping: ModularGrouping::SingleGroup,
                passes: 1,
            },
            metadata,
            Session(source),
        ))
    }
}

fn fixture(mode: &str) -> Vec<u8> {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("test-data/preview/{mode}.jxl.hex")),
    )
    .unwrap();
    let digits = text.split_whitespace().collect::<String>();
    digits
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn inventory(data: &[u8]) -> CodestreamInventory {
    jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
}

fn preview_end(inventory: &CodestreamInventory) -> usize {
    inventory.frames[0]
        .sections
        .last()
        .unwrap()
        .bytes
        .end()
        .unwrap() as usize
}

fn request() -> GpuOutputRequest {
    GpuOutputRequest::color(jxl_wgpu_decode::vardct_rgb8_format()).unwrap()
}

fn push(
    stream: &mut jxl_wgpu_decode::GpuDecodeStream<Engine>,
    data: &[u8],
    offset: usize,
) -> Result<()> {
    stream.push_transport_event(&ContainerStreamEvent::CodestreamChunk {
        logical_offset: offset as u64,
        bytes: StreamSlice::from_shared(Arc::from(data)),
    })
}

fn last_source(decoder: &GpuDecoder<Engine>) -> Arc<Source> {
    decoder
        .engine()
        .opened
        .lock()
        .unwrap()
        .last()
        .unwrap()
        .upgrade()
        .unwrap()
}

fn check_preview(source: &Source, original: &CodestreamInventory, bytes: &[u8]) {
    let end = preview_end(original);
    let ImageSourceInventory::PreviewPrefix {
        image_header,
        frame,
        codestream_bytes,
    } = source.inventory.source_inventory()
    else {
        panic!("preview claimed complete source")
    };
    assert_eq!(image_header.as_ref(), &original.image_header);
    assert_eq!(frame.as_ref(), &original.frames[0]);
    assert_eq!(*codestream_bytes, end as u64);
    assert_eq!(source.bytes.is_container(), None);
    assert_eq!(source.bytes.logical_bytes(), end as u64);
    let mut actual = vec![0; end];
    source.bytes.copy_range(0..end as u64, &mut actual).unwrap();
    assert_eq!(actual, bytes[..end]);
    assert!(
        source
            .bytes
            .for_each_range_chunk(end as u64..end as u64 + 1, |_| Ok(()))
            .is_err()
    );
    let lowered = source.inventory.reconstruction_inventory();
    assert_eq!(lowered.frames.len(), 1);
    assert_eq!(lowered.frames[0].frame_index, 0);
    assert_eq!(lowered.frames[0].noise_seed, original.frames[0].noise_seed);
    assert_eq!(lowered.frames[0].sections, original.frames[0].sections);
    assert_eq!(lowered.codestream_bytes, end as u64);
}

#[test]
fn every_transport_split_preserves_preview_prefix_main_and_shared_charges() {
    for mode in ["modular", "vardct"] {
        let raw = fixture(mode);
        let original = inventory(&raw);
        let end = preview_end(&original);
        let auxiliary = ContainerBox {
            box_type: *b"Exif",
            payload: b"retained auxiliary payload",
        };
        let mut fragments = FragmentedContainerWriter::new();
        fragments.push_box(auxiliary).unwrap();
        fragments.push_fragment(&raw[..1], false).unwrap();
        fragments.push_fragment(&raw[1..end], false).unwrap();
        fragments.push_fragment(&raw[end..], true).unwrap();
        let containers = [
            raw.clone(),
            jxl_gpu_bitstream::write_container_with_boxes(&raw, &[auxiliary]).unwrap(),
            fragments.finish().unwrap(),
        ];
        for (kind, input) in containers.iter().enumerate() {
            for split in 0..=input.len() {
                let decoder = GpuDecoder::new(Engine::default());
                let mut stream = decoder.stream(request()).unwrap();
                let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
                let mut preview = None;
                let mut prefix_charge = 0;
                let mut auxiliary_bytes = Vec::new();
                for chunk in [&input[..split], &input[split..]] {
                    for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
                        if let ContainerStreamEvent::CodestreamChunk {
                            logical_offset,
                            bytes,
                        } = &event
                            && *logical_offset < end as u64
                        {
                            prefix_charge += bytes.len() as u64;
                        }
                        if let ContainerStreamEvent::AuxiliaryBoxChunk { bytes, .. } = &event {
                            auxiliary_bytes.extend_from_slice(bytes.bytes());
                        }
                        stream.push_transport_event(&event).unwrap();
                        if preview.is_none() {
                            let before = stream.stats().input_budget;
                            preview = stream.take_preview(request()).unwrap();
                            assert_eq!(stream.stats().input_budget, before);
                        }
                    }
                }
                let preview = preview.expect("complete preview before End");
                assert!(!stream.is_ready());
                check_preview(&last_source(&decoder), &original, &raw);
                if kind != 0 {
                    assert_eq!(auxiliary_bytes, auxiliary.payload);
                }
                for event in transport.finish_input().unwrap() {
                    stream.push_transport_event(&event).unwrap();
                }
                let main = stream.finish().unwrap();
                {
                    let source = last_source(&decoder);
                    assert_eq!(source.bytes.is_container(), Some(kind != 0));
                    assert_eq!(
                        source.inventory.source_inventory().complete_inventory(),
                        Some(&original)
                    );
                    assert_eq!(source.inventory.selection(), ImageSelection::Main);
                    assert_eq!(source.bytes.logical_bytes(), raw.len() as u64);
                    assert_eq!(
                        decoder.incremental_input_budget().snapshot().reserved_bytes,
                        raw.len() as u64
                    );
                }
                drop(main);
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    prefix_charge,
                    "{mode} transport {kind} split {split}"
                );
                drop(preview);
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_spans,
                    0
                );
            }
        }
    }
}

#[test]
fn delayed_take_excludes_already_received_later_ranges_and_never_grows_with_future_ranges() {
    for mode in ["modular", "vardct"] {
        let raw = fixture(mode);
        let original = inventory(&raw);
        let end = preview_end(&original);
        for extra in [0, 1, 17] {
            let decoder = GpuDecoder::new(Engine::default());
            let mut stream = decoder.stream(request()).unwrap();
            push(&mut stream, &raw[..end + extra], 0).unwrap();
            push(&mut stream, &raw[end + extra..raw.len() - 1], end + extra).unwrap();
            let preview = stream.take_preview(request()).unwrap().unwrap();
            let source = last_source(&decoder);
            check_preview(&source, &original, &raw);
            assert_eq!(source.bytes.retained_input_bytes(), (end + extra) as u64);
            push(&mut stream, &raw[raw.len() - 1..], raw.len() - 1).unwrap();
            drop(stream);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                (end + extra) as u64
            );
            drop(source);
            drop(preview);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}

#[test]
fn preview_readiness_engine_retry_missing_preview_and_incomplete_finish_are_distinct() {
    let raw = fixture("modular");
    let original = inventory(&raw);
    let end = preview_end(&original);
    let decoder = GpuDecoder::new(Engine::default());
    let mut stream = decoder.stream(request()).unwrap();
    for offset in 0..end {
        assert!(stream.take_preview(request()).unwrap().is_none());
        push(&mut stream, &raw[offset..offset + 1], offset).unwrap();
    }
    let before = stream.stats();
    decoder.engine().reject_next.store(true, Ordering::Release);
    assert!(stream.take_preview(request()).is_err());
    assert_eq!(stream.stats(), before);
    let preview = stream.take_preview(request()).unwrap().unwrap();
    assert!(matches!(
        stream.take_preview(request()),
        Err(Error::ImageSelection(
            ImageSelectionError::PreviewAlreadyTaken
        ))
    ));
    assert!(matches!(
        stream.finish(),
        Err(Error::IncrementalInputIncomplete)
    ));
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        end as u64
    );
    drop(preview);
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0
    );

    let data = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/basic.jxl.hex"),
    )
    .unwrap();
    let digits = data.split_whitespace().collect::<String>();
    let data: Vec<_> = digits
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    let raw = jxl_gpu_bitstream::parse(&data, Default::default()).unwrap();
    let mut stream = decoder.stream(request()).unwrap();
    push(&mut stream, raw.codestream(), 0).unwrap();
    assert!(matches!(
        stream.take_preview(request()),
        Err(Error::ImageSelection(ImageSelectionError::MissingPreview))
    ));
    drop(stream);
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0
    );
}

#[test]
fn later_stream_failure_cannot_invalidate_a_published_preview() {
    for mode in ["modular", "vardct"] {
        let raw = fixture(mode);
        let original = inventory(&raw);
        let end = preview_end(&original);
        let decoder = GpuDecoder::new(Engine::default());
        let mut stream = decoder.stream(request()).unwrap();
        push(&mut stream, &raw[..end], 0).unwrap();
        let preview = stream.take_preview(request()).unwrap().unwrap();
        assert!(push(&mut stream, &[0], end + 1).is_err());
        assert_eq!(stream.stats().retained_codestream_bytes, 0);
        assert_eq!(stream.stats().retained_spans, 0);
        assert!(!stream.is_preview_ready());
        assert!(matches!(
            stream.take_preview(request()),
            Err(Error::IncrementalInputPoisoned)
        ));
        assert!(matches!(
            stream.finish(),
            Err(Error::IncrementalInputPoisoned)
        ));
        check_preview(&last_source(&decoder), &original, &raw);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            end as u64
        );
        drop(preview);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
    }
}

#[test]
fn span_and_byte_admission_are_atomic_and_retry_after_other_owner_releases() {
    for span_limit in [false, true] {
        let raw = fixture("modular");
        let budget = IncrementalInputBudget::with_limits(
            NonZeroU64::new(if span_limit { 4096 } else { 2 }).unwrap(),
            NonZeroUsize::new(if span_limit { 1 } else { 10 }).unwrap(),
        );
        let decoder =
            GpuDecoder::new(Engine::default()).with_incremental_input_budget(budget.clone());
        let mut held = decoder.stream(request()).unwrap();
        push(&mut held, &raw[..2], 0).unwrap();
        let mut next = decoder.stream(request()).unwrap();
        let before = next.stats();
        let error = push(&mut next, &raw[..1], 0).unwrap_err();
        if span_limit {
            assert!(matches!(
                error,
                Error::IncrementalInputBudget(IncrementalInputBudgetError::SpanLimit { .. })
            ));
        } else {
            assert!(matches!(
                error,
                Error::IncrementalInputBudget(IncrementalInputBudgetError::Exhausted { .. })
            ));
        }
        assert_eq!(next.stats(), before);
        drop(held);
        push(&mut next, &raw[..1], 0).unwrap();
        assert_eq!(budget.snapshot().reserved_bytes, 1);
        assert_eq!(budget.snapshot().reserved_spans, 1);
        drop(next);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
        assert_eq!(budget.snapshot().reserved_spans, 0);
    }
}
