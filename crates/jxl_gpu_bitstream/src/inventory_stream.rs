//! Incremental image/frame inventory and section delivery over validated transport events.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use thiserror::Error;

use crate::inventory::{
    ImageContext, InventoryProgress, ParsedFramePrefix, parse_frame_prefix, parse_image_header,
};
use crate::{
    BitReader, ContainerStreamEvent, Error as BitReaderError, FrameInventory, FrameSection,
    ImageHeaderInventory, InventoryError, InventoryLimits, StreamSlice,
};

const INITIAL_PREFIX_PROBE_BYTES: usize = 2;

/// Independent limits for incremental codestream inventory.
///
/// Prefix limits cover logical bytes copied into the current contiguous metadata probe. Section
/// payloads are not included: after a TOC is known they are emitted as shared [`StreamSlice`]
/// ranges. Allocator capacity and collection metadata are not included in the logical counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodestreamStreamLimits {
    pub inventory: InventoryLimits,
    pub max_codestream_bytes: u64,
    pub max_image_prefix_bytes: u64,
    pub max_frame_prefix_bytes: u64,
}

impl Default for CodestreamStreamLimits {
    fn default() -> Self {
        Self {
            inventory: InventoryLimits::default(),
            max_codestream_bytes: 1 << 30,
            max_image_prefix_bytes: 1 << 29,
            max_frame_prefix_bytes: 1 << 24,
        }
    }
}

/// Stable phase reported by incremental inventory errors and statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodestreamStreamPhase {
    ImageHeader,
    FrameHeader,
    FrameSections,
    AwaitEnd,
}

/// Logical byte and event accounting for one incremental inventory scanner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CodestreamStreamStats {
    pub codestream_bytes_received: u64,
    pub prefix_bytes_copied: u64,
    pub buffered_prefix_bytes: u64,
    pub peak_buffered_prefix_bytes: u64,
    pub section_bytes_emitted: u64,
    pub frames_started: u32,
    pub frames_completed: u32,
}

/// Incremental metadata and section event.
#[derive(Clone, Debug)]
pub enum CodestreamInventoryEvent {
    /// Complete image-header metadata. The scanner retains only its compact parsing context.
    ImageHeader(Arc<ImageHeaderInventory>),
    /// Complete frame header and TOC, emitted before any section bytes for this frame.
    FrameStart(Arc<FrameInventory>),
    /// One byte range within a physical frame section.
    SectionChunk {
        frame_index: u32,
        section: FrameSection,
        section_offset: u64,
        bytes: StreamSlice,
    },
    /// Every declared section byte for this frame has been emitted.
    FrameEnd { frame_index: u32 },
    /// The final main frame and transport end were both validated.
    End {
        codestream_bytes: u64,
        frame_count: u32,
    },
}

/// Failure while incrementally inventorying a logical JPEG XL codestream.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CodestreamStreamError {
    #[error(transparent)]
    Inventory(#[from] InventoryError),
    #[error(
        "incremental codestream chunk starts at {actual}, but the next logical offset is {expected}"
    )]
    UnexpectedOffset { expected: u64, actual: u64 },
    #[error("incremental codestream reached {bytes} bytes, limit is {limit}")]
    CodestreamSizeLimit { bytes: u64, limit: u64 },
    #[error(
        "incremental {phase:?} prefix requires {bytes} buffered bytes, logical limit is {limit}"
    )]
    PrefixSizeLimit {
        phase: CodestreamStreamPhase,
        bytes: u64,
        limit: u64,
    },
    #[error("transport ended at codestream byte {actual}, scanner expected {expected}")]
    EndOffset { expected: u64, actual: u64 },
    #[error("codestream has data after its final main-frame section at byte {byte_offset}")]
    TrailingData { byte_offset: u64 },
    #[error("allocation failed while buffering incremental {0}")]
    AllocationFailed(&'static str),
    #[error("incremental codestream inventory size arithmetic overflow")]
    SizeOverflow,
    #[error("incremental codestream inventory contract failed: {0}")]
    Contract(&'static str),
    #[error("incremental codestream inventory was already finished")]
    AlreadyFinished,
    #[error("incremental codestream inventory is poisoned by an earlier error")]
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    ImagePrefix,
    FramePrefix,
    Sections,
    AwaitEnd,
    Finished,
    Failed,
}

impl Phase {
    const fn public(self) -> Option<CodestreamStreamPhase> {
        match self {
            Self::ImagePrefix => Some(CodestreamStreamPhase::ImageHeader),
            Self::FramePrefix => Some(CodestreamStreamPhase::FrameHeader),
            Self::Sections => Some(CodestreamStreamPhase::FrameSections),
            Self::AwaitEnd => Some(CodestreamStreamPhase::AwaitEnd),
            Self::Finished | Self::Failed => None,
        }
    }
}

#[derive(Debug)]
struct ActiveFrame {
    frame: Arc<FrameInventory>,
    section_index: usize,
    section_offset: u64,
    section_end_byte: u64,
}

#[derive(Debug)]
struct PendingSlice {
    logical_offset: u64,
    bytes: StreamSlice,
}

/// Bounded incremental image/frame inventory and shared section router.
///
/// This scanner consumes logically ordered codestream ranges, not container bytes. Feed it the
/// `CodestreamChunk` and `End` values produced by [`crate::ContainerStreamScanner`], or call
/// [`Self::push_chunk`] and [`Self::finish_input`] directly. Image and frame metadata prefixes are
/// copied into one bounded contiguous probe because the published metadata parsers require a
/// slice. Once a frame TOC is known, section payload ranges are emitted without whole-codestream
/// accumulation. Events are authoritative only after [`CodestreamInventoryEvent::End`].
pub struct CodestreamStreamScanner {
    limits: CodestreamStreamLimits,
    phase: Phase,
    prefix: Vec<u8>,
    prefix_base: u64,
    next_probe_bytes: usize,
    image_context: Option<ImageContext>,
    is_preview: bool,
    progress: InventoryProgress,
    active_frame: Option<ActiveFrame>,
    next_offset: u64,
    frame_count: u32,
    stats: CodestreamStreamStats,
}

impl fmt::Debug for CodestreamStreamScanner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodestreamStreamScanner")
            .field("limits", &self.limits)
            .field("phase", &self.phase)
            .field("prefix_base", &self.prefix_base)
            .field("next_probe_bytes", &self.next_probe_bytes)
            .field("is_preview", &self.is_preview)
            .field("next_offset", &self.next_offset)
            .field("frame_count", &self.frame_count)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl CodestreamStreamScanner {
    #[must_use]
    pub const fn new(limits: CodestreamStreamLimits) -> Self {
        Self {
            limits,
            phase: Phase::ImagePrefix,
            prefix: Vec::new(),
            prefix_base: 0,
            next_probe_bytes: INITIAL_PREFIX_PROBE_BYTES,
            image_context: None,
            is_preview: false,
            progress: InventoryProgress {
                total_toc_entries: 0,
                total_section_bytes: 0,
            },
            active_frame: None,
            next_offset: 0,
            frame_count: 0,
            stats: CodestreamStreamStats {
                codestream_bytes_received: 0,
                prefix_bytes_copied: 0,
                buffered_prefix_bytes: 0,
                peak_buffered_prefix_bytes: 0,
                section_bytes_emitted: 0,
                frames_started: 0,
                frames_completed: 0,
            },
        }
    }

    #[must_use]
    pub const fn limits(&self) -> CodestreamStreamLimits {
        self.limits
    }

    #[must_use]
    pub const fn stats(&self) -> CodestreamStreamStats {
        self.stats
    }

    #[must_use]
    pub const fn phase(&self) -> Option<CodestreamStreamPhase> {
        self.phase.public()
    }

    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(self.phase, Phase::Finished)
    }

    /// Observes a transport event without taking ownership of auxiliary-box events.
    ///
    /// Auxiliary events have no effect and remain available to the caller for metadata policy.
    pub fn push_transport_event(
        &mut self,
        event: &ContainerStreamEvent,
    ) -> Result<Vec<CodestreamInventoryEvent>, CodestreamStreamError> {
        match event {
            ContainerStreamEvent::CodestreamChunk {
                logical_offset,
                bytes,
            } => self.push_chunk(*logical_offset, bytes.clone()),
            ContainerStreamEvent::End {
                codestream_bytes, ..
            } => self.finish_input(*codestream_bytes),
            ContainerStreamEvent::AuxiliaryBoxStart(_)
            | ContainerStreamEvent::AuxiliaryBoxChunk { .. }
            | ContainerStreamEvent::AuxiliaryBoxEnd { .. } => {
                self.ensure_active()?;
                Ok(Vec::new())
            }
        }
    }

    /// Consumes one logically contiguous codestream range.
    pub fn push_chunk(
        &mut self,
        logical_offset: u64,
        bytes: StreamSlice,
    ) -> Result<Vec<CodestreamInventoryEvent>, CodestreamStreamError> {
        self.ensure_active()?;
        if logical_offset != self.next_offset {
            return self.fail(CodestreamStreamError::UnexpectedOffset {
                expected: self.next_offset,
                actual: logical_offset,
            });
        }
        let length = u64::try_from(bytes.len()).map_err(|_| CodestreamStreamError::SizeOverflow)?;
        let end = logical_offset
            .checked_add(length)
            .ok_or(CodestreamStreamError::SizeOverflow)?;
        if end > self.limits.max_codestream_bytes {
            return self.fail(CodestreamStreamError::CodestreamSizeLimit {
                bytes: end,
                limit: self.limits.max_codestream_bytes,
            });
        }
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        let stats_before = self.stats;
        let mut events = Vec::new();
        let result = self.process_slices(
            VecDeque::from([PendingSlice {
                logical_offset,
                bytes,
            }]),
            &mut events,
        );
        match result {
            Ok(()) => {
                self.next_offset = end;
                self.stats.codestream_bytes_received = end;
                Ok(events)
            }
            Err(error) => {
                self.stats = stats_before;
                self.fail(error)
            }
        }
    }

    /// Declares the authoritative end of the logical codestream.
    pub fn finish_input(
        &mut self,
        codestream_bytes: u64,
    ) -> Result<Vec<CodestreamInventoryEvent>, CodestreamStreamError> {
        self.ensure_active()?;
        if codestream_bytes != self.next_offset {
            return self.fail(CodestreamStreamError::EndOffset {
                expected: self.next_offset,
                actual: codestream_bytes,
            });
        }
        let stats_before = self.stats;
        let mut events = Vec::new();
        let result = self.finish_inner(codestream_bytes, &mut events);
        match result {
            Ok(()) => Ok(events),
            Err(error) => {
                self.stats = stats_before;
                self.fail(error)
            }
        }
    }

    fn ensure_active(&self) -> Result<(), CodestreamStreamError> {
        match self.phase {
            Phase::Finished => Err(CodestreamStreamError::AlreadyFinished),
            Phase::Failed => Err(CodestreamStreamError::Failed),
            _ => Ok(()),
        }
    }

    fn fail<T>(&mut self, error: CodestreamStreamError) -> Result<T, CodestreamStreamError> {
        self.phase = Phase::Failed;
        self.prefix.clear();
        self.active_frame = None;
        self.stats.buffered_prefix_bytes = 0;
        Err(error)
    }

    fn process_slices(
        &mut self,
        mut pending: VecDeque<PendingSlice>,
        events: &mut Vec<CodestreamInventoryEvent>,
    ) -> Result<(), CodestreamStreamError> {
        while let Some(item) = pending.pop_front() {
            let mut cursor = 0usize;
            while cursor < item.bytes.len() {
                match self.phase {
                    Phase::ImagePrefix | Phase::FramePrefix => {
                        if let Some(tail) = self.consume_prefix(&item, &mut cursor, events)? {
                            if cursor < item.bytes.len() {
                                pending.push_front(PendingSlice {
                                    logical_offset: item
                                        .logical_offset
                                        .checked_add(
                                            u64::try_from(cursor)
                                                .map_err(|_| CodestreamStreamError::SizeOverflow)?,
                                        )
                                        .ok_or(CodestreamStreamError::SizeOverflow)?,
                                    bytes: item.bytes.slice(cursor..item.bytes.len()).ok_or(
                                        CodestreamStreamError::Contract(
                                            "input remainder is outside its StreamSlice",
                                        ),
                                    )?,
                                });
                            }
                            pending.push_front(tail);
                            break;
                        }
                    }
                    Phase::Sections => {
                        self.consume_sections(&item, &mut cursor, events)?;
                    }
                    Phase::AwaitEnd => {
                        let byte_offset = item
                            .logical_offset
                            .checked_add(
                                u64::try_from(cursor)
                                    .map_err(|_| CodestreamStreamError::SizeOverflow)?,
                            )
                            .ok_or(CodestreamStreamError::SizeOverflow)?;
                        return Err(CodestreamStreamError::TrailingData { byte_offset });
                    }
                    Phase::Finished => return Err(CodestreamStreamError::AlreadyFinished),
                    Phase::Failed => return Err(CodestreamStreamError::Failed),
                }
            }
        }
        Ok(())
    }

    fn consume_prefix(
        &mut self,
        item: &PendingSlice,
        cursor: &mut usize,
        events: &mut Vec<CodestreamInventoryEvent>,
    ) -> Result<Option<PendingSlice>, CodestreamStreamError> {
        let phase = self.phase.public().ok_or(CodestreamStreamError::Contract(
            "non-prefix phase copied prefix bytes",
        ))?;
        if self.prefix.is_empty() {
            self.prefix_base = item
                .logical_offset
                .checked_add(
                    u64::try_from(*cursor).map_err(|_| CodestreamStreamError::SizeOverflow)?,
                )
                .ok_or(CodestreamStreamError::SizeOverflow)?;
        }
        let expected = self
            .prefix_base
            .checked_add(
                u64::try_from(self.prefix.len())
                    .map_err(|_| CodestreamStreamError::SizeOverflow)?,
            )
            .ok_or(CodestreamStreamError::SizeOverflow)?;
        let actual = item
            .logical_offset
            .checked_add(u64::try_from(*cursor).map_err(|_| CodestreamStreamError::SizeOverflow)?)
            .ok_or(CodestreamStreamError::SizeOverflow)?;
        if expected != actual {
            return Err(CodestreamStreamError::Contract(
                "metadata prefix input is not logically contiguous",
            ));
        }

        if self.prefix.len() < self.next_probe_bytes {
            let remaining = item.bytes.len().saturating_sub(*cursor);
            let needed = self.next_probe_bytes.saturating_sub(self.prefix.len());
            let copied = remaining.min(needed);
            self.append_prefix(&item.bytes.bytes()[*cursor..*cursor + copied], phase)?;
            *cursor = cursor
                .checked_add(copied)
                .ok_or(CodestreamStreamError::SizeOverflow)?;
            if self.prefix.len() < self.next_probe_bytes {
                return Ok(None);
            }
        }

        match self.try_parse_prefix(events) {
            Ok(tail) => Ok(tail),
            Err(CodestreamStreamError::Inventory(InventoryError::UnexpectedEndOfBits {
                ..
            })) => {
                self.advance_probe(phase)?;
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn append_prefix(
        &mut self,
        bytes: &[u8],
        phase: CodestreamStreamPhase,
    ) -> Result<(), CodestreamStreamError> {
        if bytes.is_empty() {
            return Ok(());
        }
        let projected = u64::try_from(self.prefix.len())
            .map_err(|_| CodestreamStreamError::SizeOverflow)?
            .checked_add(
                u64::try_from(bytes.len()).map_err(|_| CodestreamStreamError::SizeOverflow)?,
            )
            .ok_or(CodestreamStreamError::SizeOverflow)?;
        let limit = self.prefix_limit(phase);
        if projected > limit {
            return Err(CodestreamStreamError::PrefixSizeLimit {
                phase,
                bytes: projected,
                limit,
            });
        }
        self.prefix
            .try_reserve(bytes.len())
            .map_err(|_| CodestreamStreamError::AllocationFailed("metadata prefix"))?;
        self.prefix.extend_from_slice(bytes);
        self.stats.prefix_bytes_copied = self
            .stats
            .prefix_bytes_copied
            .checked_add(
                u64::try_from(bytes.len()).map_err(|_| CodestreamStreamError::SizeOverflow)?,
            )
            .ok_or(CodestreamStreamError::SizeOverflow)?;
        self.stats.buffered_prefix_bytes = projected;
        self.stats.peak_buffered_prefix_bytes =
            self.stats.peak_buffered_prefix_bytes.max(projected);
        Ok(())
    }

    const fn prefix_limit(&self, phase: CodestreamStreamPhase) -> u64 {
        match phase {
            CodestreamStreamPhase::ImageHeader => self.limits.max_image_prefix_bytes,
            CodestreamStreamPhase::FrameHeader => self.limits.max_frame_prefix_bytes,
            CodestreamStreamPhase::FrameSections | CodestreamStreamPhase::AwaitEnd => 0,
        }
    }

    fn advance_probe(&mut self, phase: CodestreamStreamPhase) -> Result<(), CodestreamStreamError> {
        let current =
            u64::try_from(self.prefix.len()).map_err(|_| CodestreamStreamError::SizeOverflow)?;
        let limit = self.prefix_limit(phase);
        if current >= limit {
            return Err(CodestreamStreamError::PrefixSizeLimit {
                phase,
                bytes: current
                    .checked_add(1)
                    .ok_or(CodestreamStreamError::SizeOverflow)?,
                limit,
            });
        }
        let doubled = self.next_probe_bytes.saturating_mul(2);
        let at_least_one_more = self.prefix.len().saturating_add(1);
        let requested = doubled.max(at_least_one_more);
        let limit_usize = usize::try_from(limit).unwrap_or(usize::MAX);
        self.next_probe_bytes = requested.min(limit_usize);
        Ok(())
    }

    fn try_parse_prefix(
        &mut self,
        events: &mut Vec<CodestreamInventoryEvent>,
    ) -> Result<Option<PendingSlice>, CodestreamStreamError> {
        match self.phase {
            Phase::ImagePrefix => self.try_parse_image(events),
            Phase::FramePrefix => self.try_parse_frame(events),
            _ => Err(CodestreamStreamError::Contract(
                "prefix parser entered from a non-prefix phase",
            )),
        }
    }

    fn try_parse_image(
        &mut self,
        events: &mut Vec<CodestreamInventoryEvent>,
    ) -> Result<Option<PendingSlice>, CodestreamStreamError> {
        if self.prefix_base != 0 {
            return Err(CodestreamStreamError::Contract(
                "image header does not begin at codestream byte zero",
            ));
        }
        let parsed = parse_image_header(&self.prefix, self.limits.inventory)?;
        let frame_start_byte = aligned_frame_start(&self.prefix, parsed.frame_start_bits)?;
        let header = Arc::new(parsed.inventory);
        self.image_context = Some(parsed.context);
        self.is_preview = header.preview_size.is_some();
        events.push(CodestreamInventoryEvent::ImageHeader(header));
        self.phase = Phase::FramePrefix;
        self.next_probe_bytes = INITIAL_PREFIX_PROBE_BYTES;
        self.take_prefix_tail(frame_start_byte)
    }

    fn try_parse_frame(
        &mut self,
        events: &mut Vec<CodestreamInventoryEvent>,
    ) -> Result<Option<PendingSlice>, CodestreamStreamError> {
        if usize::try_from(self.frame_count).unwrap_or(usize::MAX)
            >= self.limits.inventory.max_frames
        {
            return Err(InventoryError::ResourceLimit("frame count").into());
        }
        let context = self
            .image_context
            .ok_or(CodestreamStreamError::Contract(
                "frame prefix has no image-header context",
            ))?
            .frame_context(self.is_preview)?;
        let ParsedFramePrefix {
            frame,
            section_start_byte,
            section_end_byte,
            progress,
        } = parse_frame_prefix(
            &self.prefix,
            self.prefix_base,
            self.frame_count,
            context,
            self.is_preview,
            self.limits.inventory,
            self.progress,
        )?;
        let frame = Arc::new(frame);
        self.progress = progress;
        self.frame_count = self
            .frame_count
            .checked_add(1)
            .ok_or(CodestreamStreamError::SizeOverflow)?;
        self.stats.frames_started = self.frame_count;
        events.push(CodestreamInventoryEvent::FrameStart(Arc::clone(&frame)));
        self.active_frame = Some(ActiveFrame {
            frame,
            section_index: 0,
            section_offset: 0,
            section_end_byte,
        });
        self.phase = Phase::Sections;
        self.next_probe_bytes = INITIAL_PREFIX_PROBE_BYTES;
        let tail = self.take_prefix_tail(section_start_byte)?;
        self.advance_empty_sections(events)?;
        Ok(tail)
    }

    fn take_prefix_tail(
        &mut self,
        tail_offset: u64,
    ) -> Result<Option<PendingSlice>, CodestreamStreamError> {
        let relative =
            tail_offset
                .checked_sub(self.prefix_base)
                .ok_or(CodestreamStreamError::Contract(
                    "parsed metadata ends before its prefix base",
                ))?;
        let relative =
            usize::try_from(relative).map_err(|_| CodestreamStreamError::SizeOverflow)?;
        if relative > self.prefix.len() {
            return Err(CodestreamStreamError::Contract(
                "parsed metadata ends beyond its buffered prefix",
            ));
        }
        let storage = StreamSlice::owned(std::mem::take(&mut self.prefix));
        self.stats.buffered_prefix_bytes = 0;
        self.prefix_base = tail_offset;
        if relative == storage.len() {
            return Ok(None);
        }
        let length = storage.len();
        Ok(Some(PendingSlice {
            logical_offset: tail_offset,
            bytes: storage
                .slice(relative..length)
                .ok_or(CodestreamStreamError::Contract(
                    "metadata tail is outside its owned StreamSlice",
                ))?,
        }))
    }

    fn consume_sections(
        &mut self,
        item: &PendingSlice,
        cursor: &mut usize,
        events: &mut Vec<CodestreamInventoryEvent>,
    ) -> Result<(), CodestreamStreamError> {
        self.advance_empty_sections(events)?;
        if !matches!(self.phase, Phase::Sections) {
            return Ok(());
        }
        let active = self
            .active_frame
            .as_mut()
            .ok_or(CodestreamStreamError::Contract(
                "section phase has no active frame",
            ))?;
        let section = *active.frame.sections.get(active.section_index).ok_or(
            CodestreamStreamError::Contract("active section index exceeds the frame TOC"),
        )?;
        let expected = section
            .bytes
            .offset
            .checked_add(active.section_offset)
            .ok_or(CodestreamStreamError::SizeOverflow)?;
        let actual = item
            .logical_offset
            .checked_add(u64::try_from(*cursor).map_err(|_| CodestreamStreamError::SizeOverflow)?)
            .ok_or(CodestreamStreamError::SizeOverflow)?;
        if expected != actual {
            return Err(CodestreamStreamError::Contract(
                "section bytes are not contiguous with the declared TOC range",
            ));
        }
        let section_remaining = section
            .bytes
            .length
            .checked_sub(active.section_offset)
            .ok_or(CodestreamStreamError::SizeOverflow)?;
        let input_remaining = u64::try_from(item.bytes.len().saturating_sub(*cursor))
            .map_err(|_| CodestreamStreamError::SizeOverflow)?;
        let take = section_remaining.min(input_remaining);
        let take_usize = usize::try_from(take).map_err(|_| CodestreamStreamError::SizeOverflow)?;
        let end = cursor
            .checked_add(take_usize)
            .ok_or(CodestreamStreamError::SizeOverflow)?;
        if take != 0 {
            events.push(CodestreamInventoryEvent::SectionChunk {
                frame_index: active.frame.frame_index,
                section,
                section_offset: active.section_offset,
                bytes: item
                    .bytes
                    .slice(*cursor..end)
                    .ok_or(CodestreamStreamError::Contract(
                        "section range is outside its input StreamSlice",
                    ))?,
            });
            active.section_offset = active
                .section_offset
                .checked_add(take)
                .ok_or(CodestreamStreamError::SizeOverflow)?;
            self.stats.section_bytes_emitted = self
                .stats
                .section_bytes_emitted
                .checked_add(take)
                .ok_or(CodestreamStreamError::SizeOverflow)?;
            *cursor = end;
        }
        if active.section_offset == section.bytes.length {
            active.section_index = active
                .section_index
                .checked_add(1)
                .ok_or(CodestreamStreamError::SizeOverflow)?;
            active.section_offset = 0;
            self.advance_empty_sections(events)?;
        }
        Ok(())
    }

    fn advance_empty_sections(
        &mut self,
        events: &mut Vec<CodestreamInventoryEvent>,
    ) -> Result<(), CodestreamStreamError> {
        loop {
            let Some(active) = self.active_frame.as_mut() else {
                return Ok(());
            };
            match active.frame.sections.get(active.section_index) {
                Some(section) if section.bytes.length == 0 => {
                    active.section_index = active
                        .section_index
                        .checked_add(1)
                        .ok_or(CodestreamStreamError::SizeOverflow)?;
                }
                Some(_) => return Ok(()),
                None => {
                    self.complete_frame(events)?;
                    return Ok(());
                }
            }
        }
    }

    fn complete_frame(
        &mut self,
        events: &mut Vec<CodestreamInventoryEvent>,
    ) -> Result<(), CodestreamStreamError> {
        let active = self
            .active_frame
            .take()
            .ok_or(CodestreamStreamError::Contract(
                "cannot complete an absent frame",
            ))?;
        let frame_index = active.frame.frame_index;
        events.push(CodestreamInventoryEvent::FrameEnd { frame_index });
        self.stats.frames_completed = self
            .stats
            .frames_completed
            .checked_add(1)
            .ok_or(CodestreamStreamError::SizeOverflow)?;
        self.prefix_base = active.section_end_byte;
        if active.frame.is_last {
            if active.frame.is_preview {
                self.is_preview = false;
                self.phase = Phase::FramePrefix;
            } else {
                self.phase = Phase::AwaitEnd;
            }
        } else {
            self.phase = Phase::FramePrefix;
        }
        Ok(())
    }

    fn finish_inner(
        &mut self,
        codestream_bytes: u64,
        events: &mut Vec<CodestreamInventoryEvent>,
    ) -> Result<(), CodestreamStreamError> {
        loop {
            match self.phase {
                Phase::ImagePrefix | Phase::FramePrefix => {
                    let tail = self.try_parse_prefix(events)?;
                    if let Some(tail) = tail {
                        self.process_slices(VecDeque::from([tail]), events)?;
                    }
                }
                Phase::Sections => {
                    self.advance_empty_sections(events)?;
                    if matches!(self.phase, Phase::Sections) {
                        let active =
                            self.active_frame
                                .as_ref()
                                .ok_or(CodestreamStreamError::Contract(
                                    "section phase has no active frame at end of input",
                                ))?;
                        let section = active.frame.sections.get(active.section_index).ok_or(
                            CodestreamStreamError::Contract(
                                "active section index exceeds the frame TOC at end of input",
                            ),
                        )?;
                        let byte_offset = section
                            .bytes
                            .offset
                            .checked_add(active.section_offset)
                            .ok_or(CodestreamStreamError::SizeOverflow)?;
                        return Err(InventoryError::UnexpectedEndOfBits {
                            bit_offset: byte_offset
                                .checked_mul(8)
                                .ok_or(CodestreamStreamError::SizeOverflow)?,
                        }
                        .into());
                    }
                }
                Phase::AwaitEnd => {
                    events.push(CodestreamInventoryEvent::End {
                        codestream_bytes,
                        frame_count: self.frame_count,
                    });
                    self.phase = Phase::Finished;
                    return Ok(());
                }
                Phase::Finished => return Err(CodestreamStreamError::AlreadyFinished),
                Phase::Failed => return Err(CodestreamStreamError::Failed),
            }
        }
    }
}

fn aligned_frame_start(bytes: &[u8], frame_start_bits: u64) -> Result<u64, InventoryError> {
    let mut reader = BitReader::new(bytes);
    reader
        .skip_bits(frame_start_bits)
        .map_err(|error| map_bit_reader_error(error, reader.bit_offset()))?;
    reader
        .zero_pad_to_byte()
        .map_err(|error| map_bit_reader_error(error, reader.bit_offset()))?;
    Ok(reader.bit_offset() / 8)
}

fn map_bit_reader_error(error: BitReaderError, bit_offset: u64) -> InventoryError {
    match error {
        BitReaderError::UnexpectedEndOfBits => InventoryError::UnexpectedEndOfBits { bit_offset },
        BitReaderError::NonZeroPadding => InventoryError::NonZeroPadding { bit_offset },
        BitReaderError::SizeOverflow | BitReaderError::InvalidBitCount(_) => {
            InventoryError::SizeOverflow
        }
        _ => InventoryError::InvalidFrame("invalid image-header padding"),
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::*;
    use crate::{
        CONTAINER_FILE_TYPE_BOX_V0, CONTAINER_SIGNATURE_BOX, ContainerBox, ContainerStreamLimits,
        ContainerStreamScanner, JXLP, ParseLimits, parse, write_container_with_boxes,
    };

    #[derive(Default)]
    struct CollectedInventory {
        image: Option<ImageHeaderInventory>,
        frames: Vec<FrameInventory>,
        section_bytes: Vec<Vec<Vec<u8>>>,
        frame_ends: Vec<u32>,
        end: Option<(u64, u32)>,
        saw_shared_section: bool,
    }

    impl CollectedInventory {
        fn push(&mut self, event: CodestreamInventoryEvent, source: Option<&Arc<[u8]>>) {
            match event {
                CodestreamInventoryEvent::ImageHeader(header) => {
                    assert!(self.image.replace((*header).clone()).is_none());
                }
                CodestreamInventoryEvent::FrameStart(frame) => {
                    assert_eq!(frame.frame_index as usize, self.frames.len());
                    self.section_bytes
                        .push(vec![Vec::new(); frame.sections.len()]);
                    self.frames.push((*frame).clone());
                }
                CodestreamInventoryEvent::SectionChunk {
                    frame_index,
                    section,
                    section_offset,
                    bytes,
                } => {
                    let output = &mut self.section_bytes[frame_index as usize]
                        [section.bitstream_index as usize];
                    assert_eq!(section_offset, output.len() as u64);
                    output.extend_from_slice(bytes.bytes());
                    if let (Some(source), Some(storage)) = (source, bytes.shared_storage()) {
                        self.saw_shared_section |= Arc::ptr_eq(source, &storage);
                    }
                }
                CodestreamInventoryEvent::FrameEnd { frame_index } => {
                    self.frame_ends.push(frame_index);
                }
                CodestreamInventoryEvent::End {
                    codestream_bytes,
                    frame_count,
                } => {
                    assert!(self.end.replace((codestream_bytes, frame_count)).is_none());
                }
            }
        }

        fn assert_matches(&self, expected: &crate::CodestreamInventory, codestream: &[u8]) {
            assert_eq!(self.image.as_ref(), Some(&expected.image_header));
            assert_eq!(self.frames, expected.frames);
            assert_eq!(self.frame_ends.len(), expected.frames.len());
            assert_eq!(
                self.end,
                Some((expected.codestream_bytes, expected.frames.len() as u32))
            );
            for (frame, sections) in expected.frames.iter().zip(&self.section_bytes) {
                assert_eq!(sections.len(), frame.sections.len());
                for (section, actual) in frame.sections.iter().zip(sections) {
                    let start = section.bytes.offset as usize;
                    let end = section.bytes.end().unwrap() as usize;
                    assert_eq!(actual, &codestream[start..end]);
                }
            }
        }
    }

    fn scan_transport(
        input: &[u8],
        ranges: impl IntoIterator<Item = Range<usize>>,
    ) -> (CollectedInventory, CodestreamStreamStats) {
        let mut transport = ContainerStreamScanner::new(ContainerStreamLimits::default());
        let mut inventory = CodestreamStreamScanner::new(CodestreamStreamLimits::default());
        let mut collected = CollectedInventory::default();
        for range in ranges {
            let events = transport
                .push_chunk(Arc::from(&input[range]))
                .expect("transport chunk is valid");
            for transport_event in events {
                for event in inventory
                    .push_transport_event(&transport_event)
                    .expect("codestream chunk is valid")
                {
                    collected.push(event, None);
                }
            }
        }
        for transport_event in transport.finish_input().expect("transport end is valid") {
            for event in inventory
                .push_transport_event(&transport_event)
                .expect("codestream end is valid")
            {
                collected.push(event, None);
            }
        }
        (collected, inventory.stats())
    }

    #[test]
    fn every_basic_two_chunk_split_matches_the_contiguous_inventory() {
        let input = crate::test_fixtures::basic();
        let parsed = parse(&input, ParseLimits::default()).unwrap();
        let expected = parsed
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        for split in 0..=input.len() {
            let (actual, _) = scan_transport(&input, [0..split, split..input.len()]);
            actual.assert_matches(&expected, parsed.codestream());
        }
    }

    #[test]
    fn byte_drip_fragmented_animation_matches_every_frame_and_section() {
        let input = crate::test_fixtures::fragmented_animation();
        let parsed = parse(&input, ParseLimits::default()).unwrap();
        let expected = parsed
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let (actual, stats) =
            scan_transport(&input, (0..input.len()).map(|index| index..index + 1));
        actual.assert_matches(&expected, parsed.codestream());
        assert_eq!(stats.frames_started as usize, expected.frames.len());
        assert_eq!(stats.frames_completed, stats.frames_started);
        let expected_section_bytes = expected
            .frames
            .iter()
            .flat_map(|frame| &frame.sections)
            .map(|section| section.bytes.length)
            .sum::<u64>();
        assert_eq!(stats.section_bytes_emitted, expected_section_bytes);
    }

    #[test]
    fn entropy_permuted_toc_streams_in_physical_order_with_logical_indices() {
        let input = crate::test_fixtures::has_permutation();
        let parsed = parse(&input, ParseLimits::default()).unwrap();
        let expected = parsed
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        assert!(expected.frames[0].toc_permuted);
        let (actual, _) = scan_transport(&input, (0..input.len()).map(|index| index..index + 1));
        actual.assert_matches(&expected, parsed.codestream());
    }

    #[test]
    fn out_of_order_version_one_fragments_feed_the_same_incremental_inventory() {
        let codestream = crate::test_fixtures::basic();
        let expected = parse(&codestream, ParseLimits::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let mut input = CONTAINER_SIGNATURE_BOX.to_vec();
        let mut file_type = CONTAINER_FILE_TYPE_BOX_V0;
        file_type[15] = 1;
        input.extend_from_slice(&file_type);
        push_fragment(&mut input, 1 | (1 << 31), &codestream[7..]);
        push_fragment(&mut input, 0, &codestream[..7]);
        let (actual, _) = scan_transport(&input, (0..input.len()).map(|index| index..index + 1));
        actual.assert_matches(&expected, &codestream);
    }

    #[test]
    fn observing_transport_keeps_auxiliary_events_owned_by_the_caller() {
        let input = write_container_with_boxes(
            &crate::test_fixtures::basic(),
            &[ContainerBox {
                box_type: *b"Exif",
                payload: b"opaque",
            }],
        )
        .unwrap();
        let mut transport = ContainerStreamScanner::new(ContainerStreamLimits::default());
        let mut inventory = CodestreamStreamScanner::new(CodestreamStreamLimits::default());
        let mut saw_auxiliary = false;
        for transport_event in transport.push_chunk(Arc::from(input)).unwrap() {
            let inventory_events = inventory
                .push_transport_event(&transport_event)
                .expect("observing the transport event is valid");
            if let ContainerStreamEvent::AuxiliaryBoxChunk { bytes, .. } = &transport_event {
                assert_eq!(bytes.bytes(), b"opaque");
                assert!(inventory_events.is_empty());
                saw_auxiliary = true;
            }
        }
        assert!(saw_auxiliary);
        for transport_event in transport.finish_input().unwrap() {
            inventory.push_transport_event(&transport_event).unwrap();
        }
        assert!(inventory.is_finished());
    }

    #[test]
    fn large_section_tail_shares_input_and_prefix_retention_is_sublinear() {
        let input: Arc<[u8]> = Arc::from(crate::test_fixtures::green_queen_vardct());
        let parsed = parse(&input, ParseLimits::default()).unwrap();
        let expected = parsed
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let mut transport = ContainerStreamScanner::new(ContainerStreamLimits::default());
        let mut inventory = CodestreamStreamScanner::new(CodestreamStreamLimits::default());
        let mut collected = CollectedInventory::default();
        for transport_event in transport.push_chunk(Arc::clone(&input)).unwrap() {
            for event in inventory.push_transport_event(&transport_event).unwrap() {
                collected.push(event, Some(&input));
            }
        }
        for transport_event in transport.finish_input().unwrap() {
            for event in inventory.push_transport_event(&transport_event).unwrap() {
                collected.push(event, Some(&input));
            }
        }
        collected.assert_matches(&expected, parsed.codestream());
        assert!(collected.saw_shared_section);
        assert!(inventory.stats().peak_buffered_prefix_bytes < expected.codestream_bytes / 4);
        assert_eq!(inventory.stats().buffered_prefix_bytes, 0);
    }

    #[test]
    fn prefix_limit_and_offset_errors_are_typed_and_poison_the_scanner() {
        let mut limited = CodestreamStreamScanner::new(CodestreamStreamLimits {
            max_image_prefix_bytes: 1,
            ..CodestreamStreamLimits::default()
        });
        assert_eq!(
            limited
                .push_chunk(0, StreamSlice::from_shared(Arc::from([0xff, 0x0a])))
                .unwrap_err(),
            CodestreamStreamError::PrefixSizeLimit {
                phase: CodestreamStreamPhase::ImageHeader,
                bytes: 2,
                limit: 1,
            }
        );
        assert_eq!(limited.stats(), CodestreamStreamStats::default());
        assert_eq!(
            limited
                .push_chunk(0, StreamSlice::from_shared(Arc::from([0xff])))
                .unwrap_err(),
            CodestreamStreamError::Failed
        );

        let mut offset = CodestreamStreamScanner::new(CodestreamStreamLimits::default());
        assert_eq!(
            offset
                .push_chunk(1, StreamSlice::from_shared(Arc::from([0xff])))
                .unwrap_err(),
            CodestreamStreamError::UnexpectedOffset {
                expected: 0,
                actual: 1,
            }
        );
    }

    #[test]
    fn same_call_trailing_data_discards_events_and_rolls_back_statistics() {
        let mut input = crate::test_fixtures::basic();
        let valid_bytes = input.len() as u64;
        input.push(0);
        let mut scanner = CodestreamStreamScanner::new(CodestreamStreamLimits::default());
        assert_eq!(
            scanner
                .push_chunk(0, StreamSlice::from_shared(Arc::from(input)))
                .unwrap_err(),
            CodestreamStreamError::TrailingData {
                byte_offset: valid_bytes,
            }
        );
        assert_eq!(scanner.stats(), CodestreamStreamStats::default());
        assert_eq!(
            scanner
                .finish_input(valid_bytes + 1)
                .expect_err("a poisoned scanner stays failed"),
            CodestreamStreamError::Failed
        );
    }

    #[test]
    fn every_truncated_basic_prefix_fails_before_authoritative_end() {
        let input = crate::test_fixtures::basic();
        for length in 0..input.len() {
            let mut transport = ContainerStreamScanner::new(ContainerStreamLimits::default());
            let mut inventory = CodestreamStreamScanner::new(CodestreamStreamLimits::default());
            let mut saw_end = false;
            if let Ok(events) = transport.push_chunk(Arc::from(&input[..length])) {
                for transport_event in events {
                    match inventory.push_transport_event(&transport_event) {
                        Ok(events) => {
                            saw_end |= events
                                .iter()
                                .any(|event| matches!(event, CodestreamInventoryEvent::End { .. }));
                        }
                        Err(_) => break,
                    }
                }
                if let Ok(events) = transport.finish_input() {
                    for transport_event in events {
                        if let Ok(events) = inventory.push_transport_event(&transport_event) {
                            saw_end |= events
                                .iter()
                                .any(|event| matches!(event, CodestreamInventoryEvent::End { .. }));
                        }
                    }
                }
            }
            assert!(!saw_end, "truncated prefix {length} became authoritative");
        }
    }

    fn push_fragment(output: &mut Vec<u8>, index: u32, payload: &[u8]) {
        let size = 12u32 + u32::try_from(payload.len()).unwrap();
        output.extend_from_slice(&size.to_be_bytes());
        output.extend_from_slice(&JXLP);
        output.extend_from_slice(&index.to_be_bytes());
        output.extend_from_slice(payload);
    }
}
