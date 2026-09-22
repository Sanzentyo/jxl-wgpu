use super::{FRAME_INDEX_BOX_TYPE, FrameIndex, FrameIndexError, FrameIndexLimits};
use crate::{ContainerStreamBoxHeader, ContainerStreamEvent};

#[cfg(test)]
mod tests;

/// Logical metadata retained by a collector, excluding allocator capacity and fixed-size state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameIndexCollectorStats {
    /// Encoded bytes of the currently incomplete index; released after structural parsing.
    pub retained_payload_bytes: u64,
    /// Parsed entries awaiting authoritative transport end or handoff.
    pub retained_entries: usize,
    pub authoritative_end: bool,
    pub failed: bool,
}

/// Observes borrowed [`crate::ContainerStreamScanner`] events without retaining caller storage.
/// Only one bounded plain `jxli` payload is copied. Other boxes use fixed-size state, including
/// a four-byte `brob` type probe. A parsed index cannot be taken before the scanner's final End.
#[derive(Debug)]
pub struct FrameIndexCollector {
    limits: FrameIndexLimits,
    active: Option<ActiveBox>,
    index: Option<FrameIndex>,
    ended: bool,
    failed: bool,
}

#[derive(Debug)]
struct ActiveBox {
    header: ContainerStreamBoxHeader,
    received: u64,
    brob_type: [u8; 4],
    brob_type_len: usize,
    payload: Vec<u8>,
}

impl FrameIndexCollector {
    #[must_use]
    pub const fn new(limits: FrameIndexLimits) -> Self {
        Self {
            limits,
            active: None,
            index: None,
            ended: false,
            failed: false,
        }
    }

    #[must_use]
    pub fn stats(&self) -> FrameIndexCollectorStats {
        FrameIndexCollectorStats {
            retained_payload_bytes: self
                .active
                .as_ref()
                .map_or(0, |active| active.payload.len() as u64),
            retained_entries: self.index.as_ref().map_or(0, |index| index.entries().len()),
            authoritative_end: self.ended && !self.failed,
            failed: self.failed,
        }
    }

    /// An error poisons the collector and immediately drops encoded and parsed index storage.
    pub fn push_transport_event(
        &mut self,
        event: &ContainerStreamEvent,
    ) -> Result<(), FrameIndexError> {
        if self.failed {
            return Err(FrameIndexError::CollectorFailed);
        }
        let result = if self.ended {
            Err(FrameIndexError::CollectorFinished)
        } else {
            self.observe(event)
        };
        if result.is_err() {
            self.failed = true;
            self.active = None;
            self.index = None;
        }
        result
    }

    pub fn finish(self) -> Result<Option<FrameIndex>, FrameIndexError> {
        if self.failed {
            return Err(FrameIndexError::CollectorFailed);
        }
        if !self.ended {
            return Err(FrameIndexError::IncompleteTransport);
        }
        Ok(self.index)
    }

    fn observe(&mut self, event: &ContainerStreamEvent) -> Result<(), FrameIndexError> {
        match event {
            ContainerStreamEvent::AuxiliaryBoxStart(header) => {
                if self.active.is_some() {
                    return Err(FrameIndexError::EventContract);
                }
                if header.box_type == FRAME_INDEX_BOX_TYPE {
                    if self.index.is_some() {
                        return Err(FrameIndexError::DuplicateBox);
                    }
                    if header
                        .payload_bytes
                        .is_some_and(|size| size > self.limits.max_payload_bytes)
                    {
                        return Err(FrameIndexError::PayloadLimit);
                    }
                }
                self.active = Some(ActiveBox {
                    header: *header,
                    received: 0,
                    brob_type: [0; 4],
                    brob_type_len: 0,
                    payload: Vec::new(),
                });
            }
            ContainerStreamEvent::AuxiliaryBoxChunk {
                box_type,
                payload_offset,
                bytes,
            } => {
                let active = self.active.as_mut().ok_or(FrameIndexError::EventContract)?;
                if *box_type != active.header.box_type || *payload_offset != active.received {
                    return Err(FrameIndexError::EventContract);
                }
                let received = active
                    .received
                    .checked_add(bytes.len() as u64)
                    .ok_or(FrameIndexError::Overflow)?;
                if active
                    .header
                    .payload_bytes
                    .is_some_and(|expected| received > expected)
                {
                    return Err(FrameIndexError::EventContract);
                }
                if *box_type == FRAME_INDEX_BOX_TYPE {
                    if received > self.limits.max_payload_bytes {
                        return Err(FrameIndexError::PayloadLimit);
                    }
                    let size =
                        usize::try_from(received).map_err(|_| FrameIndexError::PayloadLimit)?;
                    if size > active.payload.capacity() {
                        // Geometric growth keeps byte-drip input linear in copied payload size,
                        // while requested capacity never exceeds the caller's payload bound.
                        let limit =
                            usize::try_from(self.limits.max_payload_bytes).unwrap_or(usize::MAX);
                        let capacity = size
                            .max(active.payload.capacity().saturating_mul(2).max(32))
                            .min(limit);
                        active
                            .payload
                            .try_reserve_exact(capacity - active.payload.len())
                            .map_err(|_| FrameIndexError::Allocation)?;
                    }
                    active.payload.extend_from_slice(bytes.bytes());
                } else if *box_type == *b"brob" {
                    let count = (4 - active.brob_type_len).min(bytes.len());
                    active.brob_type[active.brob_type_len..active.brob_type_len + count]
                        .copy_from_slice(&bytes.bytes()[..count]);
                    active.brob_type_len += count;
                    if active.brob_type_len == 4 && active.brob_type == FRAME_INDEX_BOX_TYPE {
                        return Err(FrameIndexError::CompressedBox);
                    }
                }
                active.received = received;
            }
            ContainerStreamEvent::AuxiliaryBoxEnd { box_type } => {
                let active = self.active.take().ok_or(FrameIndexError::EventContract)?;
                if *box_type != active.header.box_type
                    || active
                        .header
                        .payload_bytes
                        .is_some_and(|expected| active.received != expected)
                {
                    return Err(FrameIndexError::EventContract);
                }
                if *box_type == FRAME_INDEX_BOX_TYPE {
                    self.index = Some(FrameIndex::parse(&active.payload, self.limits)?);
                }
            }
            ContainerStreamEvent::CodestreamChunk { .. } => {
                if self.active.is_some() {
                    return Err(FrameIndexError::EventContract);
                }
            }
            ContainerStreamEvent::End { .. } => {
                if self.active.is_some() {
                    return Err(FrameIndexError::IncompleteTransport);
                }
                self.ended = true;
            }
        }
        Ok(())
    }
}
