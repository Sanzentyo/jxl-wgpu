//! Incremental JPEG XL transport scanning without whole-codestream accumulation.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use crate::{
    CODESTREAM_SIGNATURE, CONTAINER_FILE_TYPE_BOX_V0, CONTAINER_SIGNATURE_BOX, Error, FTYP, JXLC,
    JXLP, ParseLimits,
};

const SIGNATURE_BOX: [u8; 4] = *b"JXL ";
const MAX_HEADER_BYTES: usize = 20;

/// Independent limits for incremental transport ingestion.
///
/// `parse` limits the complete logical input. `max_chunk_bytes` bounds one caller-owned input
/// allocation retained by emitted zero-copy slices. `max_buffered_fragment_bytes` bounds the only
/// potentially large scanner-owned storage: future `jxlp` fragments received before a missing
/// earlier fragment in file-type version 1 containers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainerStreamLimits {
    pub parse: ParseLimits,
    pub max_chunk_bytes: u64,
    pub max_buffered_fragment_bytes: u64,
}

impl Default for ContainerStreamLimits {
    fn default() -> Self {
        let parse = ParseLimits::default();
        Self {
            parse,
            max_chunk_bytes: parse.max_input_bytes,
            max_buffered_fragment_bytes: parse.max_codestream_bytes,
        }
    }
}

/// A logical byte range backed by shared input, inline signature bytes, or a future-fragment buffer.
///
/// Except for the two-byte codestream signature reconstructed across arbitrary chunk boundaries,
/// ordered raw, `jxlc`, and `jxlp` payloads use the original `Arc` allocation. Only out-of-order
/// fragment payloads are coalesced into one retained buffer per fragment so retained object count
/// cannot grow with arbitrary caller chunking. Statistics report logical payload bytes; allocator
/// capacity and collection metadata are not included.
#[derive(Clone)]
enum StreamStorage {
    Shared(Arc<[u8]>),
    Buffered(Arc<Vec<u8>>),
    InlineSignature([u8; 2]),
}

#[derive(Clone)]
pub struct StreamSlice {
    storage: StreamStorage,
    range: Range<usize>,
}

impl StreamSlice {
    fn shared(storage: Arc<[u8]>, range: Range<usize>) -> Self {
        debug_assert!(range.start <= range.end && range.end <= storage.len());
        Self {
            storage: StreamStorage::Shared(storage),
            range,
        }
    }

    fn inline_signature(bytes: [u8; 2]) -> Self {
        Self {
            storage: StreamStorage::InlineSignature(bytes),
            range: 0..bytes.len(),
        }
    }

    fn owned(bytes: Vec<u8>) -> Self {
        let length = bytes.len();
        Self {
            storage: StreamStorage::Buffered(Arc::new(bytes)),
            range: 0..length,
        }
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        let storage = match &self.storage {
            StreamStorage::Shared(storage) => storage.as_ref(),
            StreamStorage::Buffered(storage) => storage.as_slice(),
            StreamStorage::InlineSignature(storage) => storage.as_slice(),
        };
        &storage[self.range.clone()]
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.range.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Returns the `Arc<[u8]>` backing this slice when it has one.
    ///
    /// Direct payload ranges preserve the caller allocation. The reconstructed two-byte
    /// codestream signature and coalesced future fragments return `None`; their payload remains
    /// borrowable through [`Self::bytes`] without another copy.
    #[must_use]
    pub fn shared_storage(&self) -> Option<Arc<[u8]>> {
        match &self.storage {
            StreamStorage::Shared(storage) => Some(Arc::clone(storage)),
            StreamStorage::Buffered(_) | StreamStorage::InlineSignature(_) => None,
        }
    }

    #[must_use]
    pub fn storage_range(&self) -> Range<usize> {
        self.range.clone()
    }
}

impl fmt::Debug for StreamSlice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (storage_kind, storage_bytes) = match &self.storage {
            StreamStorage::Shared(storage) => ("shared", storage.len()),
            StreamStorage::Buffered(storage) => ("buffered", storage.len()),
            StreamStorage::InlineSignature(storage) => ("inline-signature", storage.len()),
        };
        formatter
            .debug_struct("StreamSlice")
            .field("range", &self.range)
            .field("storage_kind", &storage_kind)
            .field("storage_bytes", &storage_bytes)
            .finish()
    }
}

/// Exact size encoding retained from an auxiliary ISO BMFF box header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContainerBoxSizeEncoding {
    Compact,
    Extended,
    ToEnd,
}

/// Original auxiliary-box header, including its exact serialized bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ContainerStreamBoxHeader {
    pub box_type: [u8; 4],
    /// `None` denotes a size-zero box extending through end of input.
    pub payload_bytes: Option<u64>,
    pub size_encoding: ContainerBoxSizeEncoding,
    raw: [u8; 16],
    raw_len: u8,
}

impl ContainerStreamBoxHeader {
    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw[..usize::from(self.raw_len)]
    }
}

impl fmt::Debug for ContainerStreamBoxHeader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ContainerStreamBoxHeader")
            .field("box_type", &self.box_type)
            .field("payload_bytes", &self.payload_bytes)
            .field("size_encoding", &self.size_encoding)
            .field("raw_bytes", &self.raw_bytes())
            .finish()
    }
}

/// Incremental transport event. A codestream is authoritative only after [`Self::End`].
#[derive(Clone, Debug)]
pub enum ContainerStreamEvent {
    /// One contiguous logical codestream range in delivery order.
    CodestreamChunk {
        logical_offset: u64,
        bytes: StreamSlice,
    },
    /// Begins one non-transport box. Header bytes are retained exactly for byte-preserving relay.
    AuxiliaryBoxStart(ContainerStreamBoxHeader),
    /// One payload range from the current auxiliary box.
    AuxiliaryBoxChunk {
        box_type: [u8; 4],
        payload_offset: u64,
        bytes: StreamSlice,
    },
    AuxiliaryBoxEnd {
        box_type: [u8; 4],
    },
    /// The complete transport, fragment order, codestream signature, and end-of-input were valid.
    End {
        codestream_bytes: u64,
        is_container: bool,
    },
}

/// Stable context field for a typed truncated incremental-input error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContainerStreamContext {
    Signature,
    FileType,
    BoxHeader,
    BoxPayload { box_type: [u8; 4] },
}

/// Observable logical and retained-memory accounting for one scanner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContainerStreamStats {
    pub input_bytes: u64,
    pub box_count: usize,
    pub fragment_count: usize,
    pub codestream_bytes_received: u64,
    pub codestream_bytes_emitted: u64,
    /// Logical future-fragment payload bytes currently retained by the scanner.
    pub buffered_fragment_bytes: u64,
    /// Maximum logical future-fragment payload bytes retained at once.
    pub peak_buffered_fragment_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Detect,
    FileType,
    BoxHeader,
    BoxPayload,
    Raw,
    Finished,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CodestreamTransport {
    Jxlc,
    Jxlp,
}

#[derive(Debug)]
enum ActiveBoxKind {
    Jxlc,
    Jxlp {
        index_bytes: [u8; 4],
        index_len: u8,
        fragment: Option<ActiveFragment>,
    },
    Auxiliary,
}

#[derive(Debug)]
struct ActiveFragment {
    index: u32,
    buffering: bool,
    buffered: Vec<u8>,
}

#[derive(Debug)]
struct ActiveBox {
    header: ContainerStreamBoxHeader,
    remaining: Option<u64>,
    payload_offset: u64,
    kind: ActiveBoxKind,
}

#[derive(Debug)]
struct BufferedFragment {
    payload: StreamSlice,
}

/// Bounded incremental scanner for raw JPEG XL, `jxlc`, and ordered/out-of-order `jxlp` input.
///
/// This layer validates transport and emits codestream bytes in logical order. It does not parse
/// frame grammar or decode entropy. Dropping it releases every retained chunk; no callback or
/// asynchronous runtime is involved.
#[derive(Debug)]
pub struct ContainerStreamScanner {
    limits: ContainerStreamLimits,
    phase: Phase,
    scratch: [u8; MAX_HEADER_BYTES],
    scratch_len: usize,
    file_type_version: u32,
    active_box: Option<ActiveBox>,
    transport: Option<CodestreamTransport>,
    signature: [u8; 2],
    signature_len: u8,
    seen_fragments: BTreeSet<u32>,
    buffered_fragments: BTreeMap<u32, BufferedFragment>,
    next_fragment: u32,
    last_fragment: Option<u32>,
    stats: ContainerStreamStats,
}

impl ContainerStreamScanner {
    #[must_use]
    pub const fn new(limits: ContainerStreamLimits) -> Self {
        Self {
            limits,
            phase: Phase::Detect,
            scratch: [0; MAX_HEADER_BYTES],
            scratch_len: 0,
            file_type_version: 0,
            active_box: None,
            transport: None,
            signature: [0; 2],
            signature_len: 0,
            seen_fragments: BTreeSet::new(),
            buffered_fragments: BTreeMap::new(),
            next_fragment: 0,
            last_fragment: None,
            stats: ContainerStreamStats {
                input_bytes: 0,
                box_count: 0,
                fragment_count: 0,
                codestream_bytes_received: 0,
                codestream_bytes_emitted: 0,
                buffered_fragment_bytes: 0,
                peak_buffered_fragment_bytes: 0,
            },
        }
    }

    #[must_use]
    pub const fn limits(&self) -> ContainerStreamLimits {
        self.limits
    }

    #[must_use]
    pub const fn stats(&self) -> ContainerStreamStats {
        self.stats
    }

    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(self.phase, Phase::Finished)
    }

    /// Consumes one caller-owned chunk and emits zero-copy logical slices where ordering permits.
    pub fn push_chunk(&mut self, storage: Arc<[u8]>) -> Result<Vec<ContainerStreamEvent>, Error> {
        if matches!(self.phase, Phase::Finished) {
            return Err(Error::StreamAlreadyFinished);
        }
        if matches!(self.phase, Phase::Failed) {
            return Err(Error::StreamFailed);
        }
        let chunk_bytes = match u64::try_from(storage.len()) {
            Ok(bytes) => bytes,
            Err(_) => return self.fail(Error::SizeOverflow),
        };
        if chunk_bytes > self.limits.max_chunk_bytes {
            return self.fail(Error::InputChunkSizeLimit {
                bytes: chunk_bytes,
                limit: self.limits.max_chunk_bytes,
            });
        }
        let input_bytes = match self.stats.input_bytes.checked_add(chunk_bytes) {
            Some(bytes) => bytes,
            None => return self.fail(Error::SizeOverflow),
        };
        if input_bytes > self.limits.parse.max_input_bytes {
            return self.fail(Error::InputSizeLimit {
                bytes: input_bytes,
                limit: self.limits.parse.max_input_bytes,
            });
        }
        self.stats.input_bytes = input_bytes;
        if storage.is_empty() {
            return Ok(Vec::new());
        }
        let stats_before_process = self.stats;
        match self.process_chunk(storage) {
            Ok(events) => Ok(events),
            Err(error) => {
                self.stats = stats_before_process;
                self.fail(error)
            }
        }
    }

    /// Declares end of input and returns the sole authoritative [`ContainerStreamEvent::End`].
    pub fn finish_input(&mut self) -> Result<Vec<ContainerStreamEvent>, Error> {
        if matches!(self.phase, Phase::Finished) {
            return Err(Error::StreamAlreadyFinished);
        }
        if matches!(self.phase, Phase::Failed) {
            return Err(Error::StreamFailed);
        }
        let stats_before_finish = self.stats;
        match self.finish_input_inner() {
            Ok(events) => Ok(events),
            Err(error) => {
                self.stats = stats_before_finish;
                self.fail(error)
            }
        }
    }

    fn fail<T>(&mut self, error: Error) -> Result<T, Error> {
        self.phase = Phase::Failed;
        self.active_box = None;
        self.buffered_fragments.clear();
        self.stats.buffered_fragment_bytes = 0;
        Err(error)
    }

    fn process_chunk(&mut self, storage: Arc<[u8]>) -> Result<Vec<ContainerStreamEvent>, Error> {
        let mut events = Vec::new();
        let mut cursor = 0usize;
        while cursor < storage.len() {
            match self.phase {
                Phase::Detect => {
                    self.fill_scratch(&storage, &mut cursor, 2);
                    if self.scratch_len < 2 {
                        continue;
                    }
                    if self.scratch[..2] == CODESTREAM_SIGNATURE {
                        self.phase = Phase::Raw;
                        self.accept_codestream_bytes(
                            StreamSlice::inline_signature(CODESTREAM_SIGNATURE),
                            false,
                            &mut events,
                        )?;
                        self.reset_scratch();
                    } else {
                        if self.scratch[..2] != CONTAINER_SIGNATURE_BOX[..2] {
                            return Err(Error::InvalidSignature);
                        }
                        self.fill_scratch(&storage, &mut cursor, CONTAINER_SIGNATURE_BOX.len());
                        if self.scratch_len == CONTAINER_SIGNATURE_BOX.len() {
                            if self.scratch[..CONTAINER_SIGNATURE_BOX.len()]
                                != CONTAINER_SIGNATURE_BOX
                            {
                                return Err(Error::InvalidSignature);
                            }
                            self.phase = Phase::FileType;
                            self.reset_scratch();
                        }
                    }
                }
                Phase::FileType => {
                    self.fill_scratch(&storage, &mut cursor, CONTAINER_FILE_TYPE_BOX_V0.len());
                    if self.scratch_len == CONTAINER_FILE_TYPE_BOX_V0.len() {
                        self.file_type_version = validate_file_type(&self.scratch)?;
                        if self.limits.parse.max_boxes < 2 {
                            return Err(Error::BoxCountLimit {
                                boxes: 2,
                                limit: self.limits.parse.max_boxes,
                            });
                        }
                        if CONTAINER_FILE_TYPE_BOX_V0.len() as u64 > self.limits.parse.max_box_bytes
                        {
                            return Err(Error::BoxSizeLimit {
                                box_type: FTYP,
                                bytes: CONTAINER_FILE_TYPE_BOX_V0.len() as u64,
                                limit: self.limits.parse.max_box_bytes,
                            });
                        }
                        self.stats.box_count = 2;
                        self.phase = Phase::BoxHeader;
                        self.reset_scratch();
                    }
                }
                Phase::BoxHeader => {
                    self.fill_scratch(&storage, &mut cursor, 8);
                    if self.scratch_len < 8 {
                        continue;
                    }
                    let size32 =
                        u32::from_be_bytes(self.scratch[..4].try_into().map_err(|_| {
                            Error::StreamContract("box-size prefix is not four bytes")
                        })?);
                    let target = if size32 == 1 { 16 } else { 8 };
                    self.fill_scratch(&storage, &mut cursor, target);
                    if self.scratch_len == target {
                        let header = self.parse_stream_box_header(target)?;
                        self.reset_scratch();
                        self.begin_box(header, &mut events)?;
                        if self
                            .active_box
                            .as_ref()
                            .is_some_and(|active| active.remaining == Some(0))
                        {
                            self.finish_active_box(&mut events)?;
                        } else {
                            self.phase = Phase::BoxPayload;
                        }
                    }
                }
                Phase::BoxPayload => {
                    self.consume_active_payload(Arc::clone(&storage), &mut cursor, &mut events)?;
                }
                Phase::Raw => {
                    let start = cursor;
                    cursor = storage.len();
                    self.accept_codestream_bytes(
                        StreamSlice::shared(Arc::clone(&storage), start..cursor),
                        false,
                        &mut events,
                    )?;
                }
                Phase::Finished => return Err(Error::StreamAlreadyFinished),
                Phase::Failed => return Err(Error::StreamFailed),
            }
        }
        Ok(events)
    }

    fn fill_scratch(&mut self, storage: &[u8], cursor: &mut usize, target: usize) {
        let needed = target.saturating_sub(self.scratch_len);
        let copied = needed.min(storage.len().saturating_sub(*cursor));
        self.scratch[self.scratch_len..self.scratch_len + copied]
            .copy_from_slice(&storage[*cursor..*cursor + copied]);
        self.scratch_len += copied;
        *cursor += copied;
    }

    fn reset_scratch(&mut self) {
        self.scratch_len = 0;
    }

    fn parse_stream_box_header(
        &self,
        header_bytes: usize,
    ) -> Result<ContainerStreamBoxHeader, Error> {
        let size32 = u32::from_be_bytes(
            self.scratch[..4]
                .try_into()
                .map_err(|_| Error::StreamContract("box-size prefix is not four bytes"))?,
        );
        let box_type = self.scratch[4..8]
            .try_into()
            .map_err(|_| Error::StreamContract("box type is not four bytes"))?;
        let (payload_bytes, size_encoding) = match size32 {
            0 => (None, ContainerBoxSizeEncoding::ToEnd),
            1 => {
                let size =
                    u64::from_be_bytes(self.scratch[8..16].try_into().map_err(|_| {
                        Error::StreamContract("extended box size is not eight bytes")
                    })?);
                if size < 16 {
                    return Err(Error::InvalidBoxSize { box_type, size });
                }
                (Some(size - 16), ContainerBoxSizeEncoding::Extended)
            }
            size if size < 8 => {
                return Err(Error::InvalidBoxSize {
                    box_type,
                    size: u64::from(size),
                });
            }
            size => (Some(u64::from(size) - 8), ContainerBoxSizeEncoding::Compact),
        };
        let total_size = payload_bytes
            .and_then(|bytes| bytes.checked_add(header_bytes as u64))
            .unwrap_or(header_bytes as u64);
        if payload_bytes.is_some() && total_size > self.limits.parse.max_box_bytes {
            return Err(Error::BoxSizeLimit {
                box_type,
                bytes: total_size,
                limit: self.limits.parse.max_box_bytes,
            });
        }
        let mut raw = [0; 16];
        raw[..header_bytes].copy_from_slice(&self.scratch[..header_bytes]);
        Ok(ContainerStreamBoxHeader {
            box_type,
            payload_bytes,
            size_encoding,
            raw,
            raw_len: header_bytes as u8,
        })
    }

    fn begin_box(
        &mut self,
        header: ContainerStreamBoxHeader,
        events: &mut Vec<ContainerStreamEvent>,
    ) -> Result<(), Error> {
        let box_count = self
            .stats
            .box_count
            .checked_add(1)
            .ok_or(Error::SizeOverflow)?;
        if box_count > self.limits.parse.max_boxes {
            return Err(Error::BoxCountLimit {
                boxes: box_count,
                limit: self.limits.parse.max_boxes,
            });
        }
        self.stats.box_count = box_count;
        let kind = match header.box_type {
            FTYP => return Err(Error::MisplacedFileTypeBox),
            SIGNATURE_BOX => return Err(Error::MisplacedSignatureBox),
            JXLC => {
                if self.transport.is_some() {
                    return Err(Error::ConflictingCodestreamBoxes);
                }
                self.transport = Some(CodestreamTransport::Jxlc);
                ActiveBoxKind::Jxlc
            }
            JXLP => {
                if self.transport == Some(CodestreamTransport::Jxlc) {
                    return Err(Error::ConflictingCodestreamBoxes);
                }
                self.transport = Some(CodestreamTransport::Jxlp);
                ActiveBoxKind::Jxlp {
                    index_bytes: [0; 4],
                    index_len: 0,
                    fragment: None,
                }
            }
            _ => {
                events.push(ContainerStreamEvent::AuxiliaryBoxStart(header));
                ActiveBoxKind::Auxiliary
            }
        };
        self.active_box = Some(ActiveBox {
            header,
            remaining: header.payload_bytes,
            payload_offset: 0,
            kind,
        });
        Ok(())
    }

    fn consume_active_payload(
        &mut self,
        storage: Arc<[u8]>,
        cursor: &mut usize,
        events: &mut Vec<ContainerStreamEvent>,
    ) -> Result<(), Error> {
        let mut active = self
            .active_box
            .take()
            .ok_or(Error::StreamContract("box-payload state has no active box"))?;
        let available = u64::try_from(storage.len().saturating_sub(*cursor))
            .map_err(|_| Error::SizeOverflow)?;
        let take = active
            .remaining
            .map_or(available, |bytes| bytes.min(available));
        let end = cursor
            .checked_add(usize::try_from(take).map_err(|_| Error::SizeOverflow)?)
            .ok_or(Error::SizeOverflow)?;
        let total_box_bytes = u64::from(active.header.raw_len)
            .checked_add(active.payload_offset)
            .and_then(|bytes| bytes.checked_add(take))
            .ok_or(Error::SizeOverflow)?;
        if total_box_bytes > self.limits.parse.max_box_bytes {
            return Err(Error::BoxSizeLimit {
                box_type: active.header.box_type,
                bytes: total_box_bytes,
                limit: self.limits.parse.max_box_bytes,
            });
        }

        let mut payload_cursor = *cursor;
        match &mut active.kind {
            ActiveBoxKind::Jxlc => {
                self.accept_codestream_bytes(
                    StreamSlice::shared(Arc::clone(&storage), payload_cursor..end),
                    false,
                    events,
                )?;
                payload_cursor = end;
            }
            ActiveBoxKind::Jxlp {
                index_bytes,
                index_len,
                fragment,
            } => {
                if *index_len < 4 {
                    let needed = 4usize - usize::from(*index_len);
                    let copied = needed.min(end.saturating_sub(payload_cursor));
                    index_bytes[usize::from(*index_len)..usize::from(*index_len) + copied]
                        .copy_from_slice(&storage[payload_cursor..payload_cursor + copied]);
                    *index_len += copied as u8;
                    payload_cursor += copied;
                    if *index_len == 4 {
                        *fragment = Some(self.begin_fragment(*index_bytes)?);
                    }
                }
                if payload_cursor < end {
                    let fragment = fragment.as_mut().ok_or(Error::StreamContract(
                        "jxlp payload data arrived before its fragment index",
                    ))?;
                    let span = StreamSlice::shared(Arc::clone(&storage), payload_cursor..end);
                    self.accept_codestream_bytes(span.clone(), fragment.buffering, events)?;
                    if fragment.buffering {
                        fragment
                            .buffered
                            .try_reserve(span.len())
                            .map_err(|_| Error::AllocationFailed("out-of-order jxlp fragment"))?;
                        fragment.buffered.extend_from_slice(span.bytes());
                    }
                    payload_cursor = end;
                }
            }
            ActiveBoxKind::Auxiliary => {
                if payload_cursor < end {
                    events.push(ContainerStreamEvent::AuxiliaryBoxChunk {
                        box_type: active.header.box_type,
                        payload_offset: active.payload_offset,
                        bytes: StreamSlice::shared(Arc::clone(&storage), payload_cursor..end),
                    });
                    payload_cursor = end;
                }
            }
        }
        debug_assert_eq!(payload_cursor, end);
        active.payload_offset = active
            .payload_offset
            .checked_add(take)
            .ok_or(Error::SizeOverflow)?;
        if let Some(remaining) = &mut active.remaining {
            *remaining = remaining.checked_sub(take).ok_or(Error::SizeOverflow)?;
        }
        *cursor = end;
        let complete = active.remaining == Some(0);
        self.active_box = Some(active);
        if complete {
            self.finish_active_box(events)?;
        }
        Ok(())
    }

    fn begin_fragment(&mut self, index_bytes: [u8; 4]) -> Result<ActiveFragment, Error> {
        let raw_index = u32::from_be_bytes(index_bytes);
        let is_last = raw_index & (1 << 31) != 0;
        let index = raw_index & !(1 << 31);
        if self.file_type_version == 0 && index != self.next_fragment {
            return Err(Error::OutOfOrderFragment {
                expected: self.next_fragment,
                actual: index,
            });
        }
        if !self.seen_fragments.insert(index) {
            return Err(Error::DuplicateFragment(index));
        }
        if let Some(last) = self.last_fragment
            && index > last
        {
            return Err(Error::FragmentAfterLast(index));
        }
        if is_last {
            if self.last_fragment.replace(index).is_some() {
                return Err(Error::MultipleLastFragments);
            }
            if let Some(&after_last) = self.seen_fragments.range(index.saturating_add(1)..).next() {
                return Err(Error::FragmentAfterLast(after_last));
            }
        }
        self.stats.fragment_count = self
            .stats
            .fragment_count
            .checked_add(1)
            .ok_or(Error::SizeOverflow)?;
        Ok(ActiveFragment {
            index,
            buffering: index != self.next_fragment,
            buffered: Vec::new(),
        })
    }

    fn finish_active_box(&mut self, events: &mut Vec<ContainerStreamEvent>) -> Result<(), Error> {
        let active = self.active_box.take().ok_or(Error::StreamContract(
            "cannot finish an absent container box",
        ))?;
        match active.kind {
            ActiveBoxKind::Jxlc => {
                if self.signature_len != 2 {
                    return Err(Error::InvalidCodestreamSignature);
                }
            }
            ActiveBoxKind::Jxlp {
                index_len,
                fragment,
                ..
            } => {
                if index_len != 4 {
                    return Err(Error::TruncatedFragmentIndex);
                }
                let fragment = fragment.ok_or(Error::StreamContract(
                    "complete jxlp index has no fragment state",
                ))?;
                if fragment.buffering {
                    let payload = StreamSlice::owned(fragment.buffered);
                    let buffered = BufferedFragment { payload };
                    if self
                        .buffered_fragments
                        .insert(fragment.index, buffered)
                        .is_some()
                    {
                        return Err(Error::DuplicateFragment(fragment.index));
                    }
                } else {
                    self.next_fragment = self
                        .next_fragment
                        .checked_add(1)
                        .ok_or(Error::SizeOverflow)?;
                    self.flush_buffered_fragments(events)?;
                }
            }
            ActiveBoxKind::Auxiliary => {
                events.push(ContainerStreamEvent::AuxiliaryBoxEnd {
                    box_type: active.header.box_type,
                });
            }
        }
        self.phase = Phase::BoxHeader;
        Ok(())
    }

    fn accept_codestream_bytes(
        &mut self,
        bytes: StreamSlice,
        buffer: bool,
        events: &mut Vec<ContainerStreamEvent>,
    ) -> Result<(), Error> {
        let length = u64::try_from(bytes.len()).map_err(|_| Error::SizeOverflow)?;
        let received = self
            .stats
            .codestream_bytes_received
            .checked_add(length)
            .ok_or(Error::SizeOverflow)?;
        if received > self.limits.parse.max_codestream_bytes {
            return Err(Error::CodestreamSizeLimit {
                bytes: received,
                limit: self.limits.parse.max_codestream_bytes,
            });
        }
        self.stats.codestream_bytes_received = received;
        if buffer {
            let buffered = self
                .stats
                .buffered_fragment_bytes
                .checked_add(length)
                .ok_or(Error::SizeOverflow)?;
            if buffered > self.limits.max_buffered_fragment_bytes {
                return Err(Error::BufferedFragmentSizeLimit {
                    bytes: buffered,
                    limit: self.limits.max_buffered_fragment_bytes,
                });
            }
            self.stats.buffered_fragment_bytes = buffered;
            self.stats.peak_buffered_fragment_bytes =
                self.stats.peak_buffered_fragment_bytes.max(buffered);
        } else {
            self.emit_codestream_slice(bytes, events)?;
        }
        Ok(())
    }

    fn emit_codestream_slice(
        &mut self,
        bytes: StreamSlice,
        events: &mut Vec<ContainerStreamEvent>,
    ) -> Result<(), Error> {
        let mut start = 0usize;
        if self.signature_len < 2 {
            let needed = 2usize - usize::from(self.signature_len);
            let copied = needed.min(bytes.len());
            self.signature
                [usize::from(self.signature_len)..usize::from(self.signature_len) + copied]
                .copy_from_slice(&bytes.bytes()[..copied]);
            self.signature_len += copied as u8;
            start = copied;
            if self.signature_len == 2 {
                if self.signature != CODESTREAM_SIGNATURE {
                    return Err(Error::InvalidCodestreamSignature);
                }
                self.push_emitted(StreamSlice::inline_signature(self.signature), events)?;
            }
        }
        if start < bytes.len() {
            let range = bytes.storage_range();
            let tail_start = range.start.checked_add(start).ok_or(Error::SizeOverflow)?;
            self.push_emitted(
                StreamSlice {
                    storage: bytes.storage.clone(),
                    range: tail_start..range.end,
                },
                events,
            )?;
        }
        Ok(())
    }

    fn push_emitted(
        &mut self,
        bytes: StreamSlice,
        events: &mut Vec<ContainerStreamEvent>,
    ) -> Result<(), Error> {
        if bytes.is_empty() {
            return Ok(());
        }
        let logical_offset = self.stats.codestream_bytes_emitted;
        self.stats.codestream_bytes_emitted = logical_offset
            .checked_add(u64::try_from(bytes.len()).map_err(|_| Error::SizeOverflow)?)
            .ok_or(Error::SizeOverflow)?;
        events.push(ContainerStreamEvent::CodestreamChunk {
            logical_offset,
            bytes,
        });
        Ok(())
    }

    fn flush_buffered_fragments(
        &mut self,
        events: &mut Vec<ContainerStreamEvent>,
    ) -> Result<(), Error> {
        while let Some(fragment) = self.buffered_fragments.remove(&self.next_fragment) {
            let bytes = u64::try_from(fragment.payload.len()).map_err(|_| Error::SizeOverflow)?;
            self.stats.buffered_fragment_bytes = self
                .stats
                .buffered_fragment_bytes
                .checked_sub(bytes)
                .ok_or(Error::SizeOverflow)?;
            self.emit_codestream_slice(fragment.payload, events)?;
            self.next_fragment = self
                .next_fragment
                .checked_add(1)
                .ok_or(Error::SizeOverflow)?;
        }
        Ok(())
    }

    fn finish_input_inner(&mut self) -> Result<Vec<ContainerStreamEvent>, Error> {
        let mut events = Vec::new();
        match self.phase {
            Phase::Detect => {
                return Err(Error::UnexpectedEndOfInput {
                    context: ContainerStreamContext::Signature,
                });
            }
            Phase::FileType => {
                return Err(Error::UnexpectedEndOfInput {
                    context: ContainerStreamContext::FileType,
                });
            }
            Phase::BoxHeader if self.scratch_len != 0 => {
                return Err(Error::UnexpectedEndOfInput {
                    context: ContainerStreamContext::BoxHeader,
                });
            }
            Phase::BoxPayload => {
                let active = self.active_box.as_ref().ok_or(Error::StreamContract(
                    "box-payload state has no active box at end of input",
                ))?;
                if active.remaining.is_some() {
                    return Err(Error::UnexpectedEndOfInput {
                        context: ContainerStreamContext::BoxPayload {
                            box_type: active.header.box_type,
                        },
                    });
                }
                self.finish_active_box(&mut events)?;
            }
            Phase::Raw | Phase::BoxHeader => {}
            Phase::Finished => return Err(Error::StreamAlreadyFinished),
            Phase::Failed => return Err(Error::StreamFailed),
        }
        let is_container = !matches!(self.phase, Phase::Raw);
        if is_container {
            match self.transport {
                None => return Err(Error::MissingCodestream),
                Some(CodestreamTransport::Jxlc) => {}
                Some(CodestreamTransport::Jxlp) => {
                    let last = self.last_fragment.ok_or(Error::MissingFinalFragment)?;
                    let expected = last.checked_add(1).ok_or(Error::SizeOverflow)?;
                    if self.next_fragment != expected || !self.buffered_fragments.is_empty() {
                        return Err(Error::MissingFragment);
                    }
                }
            }
        }
        if self.signature_len != 2 {
            return Err(Error::InvalidCodestreamSignature);
        }
        if self.stats.codestream_bytes_received != self.stats.codestream_bytes_emitted {
            return Err(Error::StreamContract(
                "validated transport retained unemitted codestream bytes",
            ));
        }
        events.push(ContainerStreamEvent::End {
            codestream_bytes: self.stats.codestream_bytes_emitted,
            is_container,
        });
        self.phase = Phase::Finished;
        Ok(events)
    }
}

fn validate_file_type(bytes: &[u8]) -> Result<u32, Error> {
    if bytes.len() != CONTAINER_FILE_TYPE_BOX_V0.len()
        || bytes[..12] != CONTAINER_FILE_TYPE_BOX_V0[..12]
        || bytes[16..20] != CONTAINER_FILE_TYPE_BOX_V0[16..20]
    {
        return Err(Error::InvalidFileTypeBox);
    }
    let version = u32::from_be_bytes(
        bytes[12..16]
            .try_into()
            .map_err(|_| Error::StreamContract("file-type version is not four bytes"))?,
    );
    if version > 1 {
        return Err(Error::UnsupportedFileTypeVersion(version));
    }
    Ok(version)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContainerBox, FragmentedContainerWriter, write_container_with_boxes};

    fn collect_codestream(events: impl IntoIterator<Item = ContainerStreamEvent>) -> Vec<u8> {
        events
            .into_iter()
            .filter_map(|event| match event {
                ContainerStreamEvent::CodestreamChunk { bytes, .. } => Some(bytes),
                _ => None,
            })
            .flat_map(|bytes| bytes.bytes().to_vec())
            .collect()
    }

    fn scan_chunks(input: &[u8], chunks: &[Range<usize>]) -> Result<(Vec<u8>, bool), Error> {
        let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits::default());
        let mut codestream = Vec::new();
        for range in chunks {
            codestream.extend(collect_codestream(
                scanner.push_chunk(Arc::from(&input[range.clone()]))?,
            ));
        }
        let end = scanner.finish_input()?;
        let is_container = end.iter().find_map(|event| match event {
            ContainerStreamEvent::End { is_container, .. } => Some(*is_container),
            _ => None,
        });
        codestream.extend(collect_codestream(end));
        Ok((codestream, is_container.expect("finish emits End")))
    }

    #[test]
    fn raw_codestream_is_identical_at_every_two_chunk_split() {
        let input = crate::test_fixtures::basic();
        for split in 0..=input.len() {
            let (actual, is_container) =
                scan_chunks(&input, &[0..split, split..input.len()]).unwrap();
            assert_eq!(actual, input, "split {split}");
            assert!(!is_container);
        }
    }

    #[test]
    fn container_byte_drip_preserves_codestream_and_auxiliary_box_bytes() {
        let codestream = crate::test_fixtures::basic();
        let input = write_container_with_boxes(
            &codestream,
            &[ContainerBox {
                box_type: *b"Exif",
                payload: b"opaque-metadata",
            }],
        )
        .unwrap();
        let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits::default());
        let mut actual_codestream = Vec::new();
        let mut auxiliary = Vec::new();
        let mut raw_header = Vec::new();
        for &byte in &input {
            for event in scanner.push_chunk(Arc::from([byte])).unwrap() {
                match event {
                    ContainerStreamEvent::CodestreamChunk {
                        logical_offset,
                        bytes,
                    } => {
                        assert_eq!(logical_offset, actual_codestream.len() as u64);
                        actual_codestream.extend_from_slice(bytes.bytes());
                    }
                    ContainerStreamEvent::AuxiliaryBoxStart(header) => {
                        raw_header.extend_from_slice(header.raw_bytes());
                    }
                    ContainerStreamEvent::AuxiliaryBoxChunk { bytes, .. } => {
                        auxiliary.extend_from_slice(bytes.bytes());
                    }
                    ContainerStreamEvent::AuxiliaryBoxEnd { .. }
                    | ContainerStreamEvent::End { .. } => {}
                }
            }
        }
        let end = scanner.finish_input().unwrap();
        assert!(matches!(
            end.as_slice(),
            [ContainerStreamEvent::End {
                is_container: true,
                ..
            }]
        ));
        assert_eq!(actual_codestream, codestream);
        assert_eq!(auxiliary, b"opaque-metadata");
        assert_eq!(&raw_header[4..8], b"Exif");
        assert_eq!(scanner.stats().buffered_fragment_bytes, 0);
    }

    #[test]
    fn ordered_fragments_stream_at_every_box_and_index_boundary() {
        let codestream = crate::test_fixtures::basic();
        let mut writer = FragmentedContainerWriter::new();
        writer.push_fragment(&codestream[..1], false).unwrap();
        writer.push_fragment(&codestream[1..7], false).unwrap();
        writer.push_fragment(&codestream[7..], true).unwrap();
        let input = writer.finish().unwrap();
        for split in 0..=input.len() {
            let (actual, is_container) =
                scan_chunks(&input, &[0..split, split..input.len()]).unwrap();
            assert_eq!(actual, codestream, "split {split}");
            assert!(is_container);
        }
    }

    #[test]
    fn extended_and_to_end_box_headers_survive_every_split() {
        let codestream = crate::test_fixtures::basic();
        let mut input = CONTAINER_SIGNATURE_BOX.to_vec();
        input.extend_from_slice(&CONTAINER_FILE_TYPE_BOX_V0);
        input.extend_from_slice(&1_u32.to_be_bytes());
        input.extend_from_slice(b"Exif");
        input.extend_from_slice(&19_u64.to_be_bytes());
        input.extend_from_slice(b"abc");
        input.extend_from_slice(&0_u32.to_be_bytes());
        input.extend_from_slice(&JXLC);
        input.extend_from_slice(&codestream);

        for split in 0..=input.len() {
            let (actual, is_container) =
                scan_chunks(&input, &[0..split, split..input.len()]).unwrap();
            assert_eq!(actual, codestream, "split {split}");
            assert!(is_container);
        }
    }

    #[test]
    fn ordered_payload_events_share_the_caller_allocation() {
        let codestream = crate::test_fixtures::basic();
        let storage: Arc<[u8]> = Arc::from(codestream.clone());
        let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits::default());
        let events = scanner.push_chunk(Arc::clone(&storage)).unwrap();
        let shared_tail = events.into_iter().find_map(|event| match event {
            ContainerStreamEvent::CodestreamChunk { bytes, .. }
                if bytes.storage_range().start >= 2 =>
            {
                bytes.shared_storage()
            }
            _ => None,
        });
        assert!(Arc::ptr_eq(
            &storage,
            &shared_tail.expect("raw payload tail is emitted")
        ));
        scanner.finish_input().unwrap();
    }

    #[test]
    fn ordered_container_payloads_share_the_complete_input_allocation() {
        let codestream = crate::test_fixtures::basic();
        let input = write_container_with_boxes(
            &codestream,
            &[ContainerBox {
                box_type: *b"Exif",
                payload: b"opaque",
            }],
        )
        .unwrap();
        let storage: Arc<[u8]> = Arc::from(input);
        let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits::default());
        let events = scanner.push_chunk(Arc::clone(&storage)).unwrap();
        let mut saw_inline_signature = false;
        let mut saw_shared_codestream = false;
        let mut saw_shared_auxiliary = false;
        for event in events {
            match event {
                ContainerStreamEvent::CodestreamChunk {
                    logical_offset: 0,
                    bytes,
                } => {
                    assert_eq!(bytes.bytes(), CODESTREAM_SIGNATURE);
                    assert!(bytes.shared_storage().is_none());
                    saw_inline_signature = true;
                }
                ContainerStreamEvent::CodestreamChunk { bytes, .. } => {
                    assert!(Arc::ptr_eq(
                        &storage,
                        &bytes
                            .shared_storage()
                            .expect("ordered jxlc range is shared")
                    ));
                    saw_shared_codestream = true;
                }
                ContainerStreamEvent::AuxiliaryBoxChunk { bytes, .. } => {
                    assert!(Arc::ptr_eq(
                        &storage,
                        &bytes
                            .shared_storage()
                            .expect("ordered auxiliary range is shared")
                    ));
                    saw_shared_auxiliary = true;
                }
                ContainerStreamEvent::AuxiliaryBoxStart(_)
                | ContainerStreamEvent::AuxiliaryBoxEnd { .. }
                | ContainerStreamEvent::End { .. } => {}
            }
        }
        assert!(saw_inline_signature && saw_shared_codestream && saw_shared_auxiliary);
        scanner.finish_input().unwrap();
    }

    #[test]
    fn ordered_fragment_payloads_share_the_complete_input_allocation() {
        let codestream = crate::test_fixtures::basic();
        let mut writer = FragmentedContainerWriter::new();
        writer.push_fragment(&codestream[..1], false).unwrap();
        writer.push_fragment(&codestream[1..], true).unwrap();
        let storage: Arc<[u8]> = Arc::from(writer.finish().unwrap());
        let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits::default());
        let events = scanner.push_chunk(Arc::clone(&storage)).unwrap();
        let mut saw_inline_signature = false;
        let mut shared_payload_events = 0usize;
        for event in events {
            if let ContainerStreamEvent::CodestreamChunk {
                logical_offset,
                bytes,
            } = event
            {
                if logical_offset == 0 {
                    assert_eq!(bytes.bytes(), CODESTREAM_SIGNATURE);
                    assert!(bytes.shared_storage().is_none());
                    saw_inline_signature = true;
                } else {
                    assert!(Arc::ptr_eq(
                        &storage,
                        &bytes
                            .shared_storage()
                            .expect("ordered jxlp range is shared")
                    ));
                    shared_payload_events += 1;
                }
            }
        }
        assert!(saw_inline_signature);
        assert!(shared_payload_events > 0);
        scanner.finish_input().unwrap();
    }

    #[test]
    fn fixed_container_boxes_obey_typed_count_and_size_limits() {
        let mut input = CONTAINER_SIGNATURE_BOX.to_vec();
        input.extend_from_slice(&CONTAINER_FILE_TYPE_BOX_V0);

        let mut count_limited = ContainerStreamScanner::new(ContainerStreamLimits {
            parse: ParseLimits {
                max_boxes: 1,
                ..ParseLimits::default()
            },
            ..ContainerStreamLimits::default()
        });
        assert_eq!(
            count_limited
                .push_chunk(Arc::from(input.clone()))
                .unwrap_err(),
            Error::BoxCountLimit { boxes: 2, limit: 1 }
        );

        let mut size_limited = ContainerStreamScanner::new(ContainerStreamLimits {
            parse: ParseLimits {
                max_box_bytes: 19,
                ..ParseLimits::default()
            },
            ..ContainerStreamLimits::default()
        });
        assert_eq!(
            size_limited.push_chunk(Arc::from(input)).unwrap_err(),
            Error::BoxSizeLimit {
                box_type: FTYP,
                bytes: 20,
                limit: 19,
            }
        );

        let mut to_end = CONTAINER_SIGNATURE_BOX.to_vec();
        to_end.extend_from_slice(&CONTAINER_FILE_TYPE_BOX_V0);
        to_end.extend_from_slice(&0_u32.to_be_bytes());
        to_end.extend_from_slice(b"Exif");
        to_end.extend_from_slice(&[0_u8; 13]);
        let mut to_end_limited = ContainerStreamScanner::new(ContainerStreamLimits {
            parse: ParseLimits {
                max_box_bytes: 20,
                ..ParseLimits::default()
            },
            ..ContainerStreamLimits::default()
        });
        assert_eq!(
            to_end_limited.push_chunk(Arc::from(to_end)).unwrap_err(),
            Error::BoxSizeLimit {
                box_type: *b"Exif",
                bytes: 21,
                limit: 20,
            }
        );
    }

    #[test]
    fn chunk_input_and_codestream_limits_report_exact_attempted_bytes() {
        let mut chunk_limited = ContainerStreamScanner::new(ContainerStreamLimits {
            max_chunk_bytes: 2,
            ..ContainerStreamLimits::default()
        });
        assert_eq!(
            chunk_limited.push_chunk(Arc::from([0_u8; 3])).unwrap_err(),
            Error::InputChunkSizeLimit { bytes: 3, limit: 2 }
        );

        let mut input_limited = ContainerStreamScanner::new(ContainerStreamLimits {
            parse: ParseLimits {
                max_input_bytes: 3,
                ..ParseLimits::default()
            },
            max_chunk_bytes: 3,
            ..ContainerStreamLimits::default()
        });
        input_limited
            .push_chunk(Arc::from(CODESTREAM_SIGNATURE))
            .unwrap();
        assert_eq!(
            input_limited.push_chunk(Arc::from([1_u8, 2])).unwrap_err(),
            Error::InputSizeLimit { bytes: 4, limit: 3 }
        );

        let mut codestream_limited = ContainerStreamScanner::new(ContainerStreamLimits {
            parse: ParseLimits {
                max_codestream_bytes: 2,
                ..ParseLimits::default()
            },
            ..ContainerStreamLimits::default()
        });
        assert_eq!(
            codestream_limited
                .push_chunk(Arc::from([0xff_u8, 0x0a, 1]))
                .unwrap_err(),
            Error::CodestreamSizeLimit { bytes: 3, limit: 2 }
        );
    }

    #[test]
    fn same_chunk_failure_does_not_report_discarded_events_as_emitted() {
        let mut input = CONTAINER_SIGNATURE_BOX.to_vec();
        input.extend_from_slice(&CONTAINER_FILE_TYPE_BOX_V0);
        for _ in 0..2 {
            input.extend_from_slice(&10_u32.to_be_bytes());
            input.extend_from_slice(&JXLC);
            input.extend_from_slice(&CODESTREAM_SIGNATURE);
        }
        let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits::default());
        assert_eq!(
            scanner.push_chunk(Arc::from(input.clone())).unwrap_err(),
            Error::ConflictingCodestreamBoxes
        );
        assert_eq!(scanner.stats().input_bytes, input.len() as u64);
        assert_eq!(scanner.stats().codestream_bytes_received, 0);
        assert_eq!(scanner.stats().codestream_bytes_emitted, 0);
    }

    #[test]
    fn version_one_out_of_order_fragments_are_bounded_and_reordered() {
        let mut input = CONTAINER_SIGNATURE_BOX.to_vec();
        let mut file_type = CONTAINER_FILE_TYPE_BOX_V0;
        file_type[15] = 1;
        input.extend_from_slice(&file_type);
        push_fragment(&mut input, 1 | (1 << 31), b"cdef");
        let future_fragment_end = input.len();
        push_fragment(&mut input, 0, &[0xff, 0x0a, b'a', b'b']);

        let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits {
            max_buffered_fragment_bytes: 4,
            ..ContainerStreamLimits::default()
        });
        let mut actual = Vec::new();
        for &byte in &input[..future_fragment_end] {
            actual.extend(collect_codestream(
                scanner.push_chunk(Arc::from([byte])).unwrap(),
            ));
        }
        assert!(actual.is_empty());
        let buffered = scanner
            .buffered_fragments
            .get(&1)
            .expect("future fragment is complete");
        assert_eq!(buffered.payload.bytes(), b"cdef");
        assert!(buffered.payload.shared_storage().is_none());
        assert!(matches!(
            &buffered.payload.storage,
            StreamStorage::Buffered(storage) if storage.len() == 4
        ));
        for &byte in &input[future_fragment_end..] {
            actual.extend(collect_codestream(
                scanner.push_chunk(Arc::from([byte])).unwrap(),
            ));
        }
        assert_eq!(actual, b"\xff\x0aabcdef");
        assert_eq!(scanner.stats().peak_buffered_fragment_bytes, 4);
        assert_eq!(scanner.stats().buffered_fragment_bytes, 0);
        scanner.finish_input().unwrap();
    }

    #[test]
    fn exact_buffer_limit_rejects_future_fragment_before_unbounded_retention() {
        let mut input = CONTAINER_SIGNATURE_BOX.to_vec();
        let mut file_type = CONTAINER_FILE_TYPE_BOX_V0;
        file_type[15] = 1;
        input.extend_from_slice(&file_type);
        push_fragment(&mut input, 1 | (1 << 31), b"five!");
        let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits {
            max_buffered_fragment_bytes: 4,
            ..ContainerStreamLimits::default()
        });
        assert_eq!(
            scanner.push_chunk(Arc::from(input)).unwrap_err(),
            Error::BufferedFragmentSizeLimit { bytes: 5, limit: 4 }
        );
        assert_eq!(scanner.stats().buffered_fragment_bytes, 0);
        assert_eq!(
            scanner.push_chunk(Arc::from([0_u8])).unwrap_err(),
            Error::StreamFailed
        );
    }

    #[test]
    fn streamed_fragments_distinguish_an_absent_final_marker() {
        let mut input = CONTAINER_SIGNATURE_BOX.to_vec();
        input.extend_from_slice(&CONTAINER_FILE_TYPE_BOX_V0);
        push_fragment(&mut input, 0, &[0xff, 0x0a, 1]);
        let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits::default());
        scanner.push_chunk(Arc::from(input)).unwrap();
        assert_eq!(
            scanner.finish_input().unwrap_err(),
            Error::MissingFinalFragment
        );
    }

    #[test]
    fn every_container_prefix_is_a_typed_error_or_an_exact_to_end_payload() {
        let input = crate::test_fixtures::fragmented_animation();
        for length in 0..input.len() {
            let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits::default());
            let mut codestream = Vec::new();
            match scanner.push_chunk(Arc::from(&input[..length])) {
                Ok(events) => {
                    codestream.extend(collect_codestream(events));
                    if let Ok(events) = scanner.finish_input() {
                        codestream.extend(collect_codestream(events));
                        let parsed =
                            crate::parse(&input[..length], ParseLimits::default()).unwrap();
                        assert_eq!(codestream, parsed.codestream(), "prefix {length}");
                    }
                }
                Err(_) => continue,
            }
        }
        let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits::default());
        scanner.push_chunk(Arc::from(input)).unwrap();
        scanner.finish_input().unwrap();
    }

    fn push_fragment(output: &mut Vec<u8>, index: u32, payload: &[u8]) {
        let size = 12u32 + u32::try_from(payload.len()).unwrap();
        output.extend_from_slice(&size.to_be_bytes());
        output.extend_from_slice(&JXLP);
        output.extend_from_slice(&index.to_be_bytes());
        output.extend_from_slice(payload);
    }
}
