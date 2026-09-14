use crate::{ContainerStreamBoxHeader, ContainerStreamEvent};

use super::{
    BROB, Metadata, MetadataBox, MetadataError, MetadataLimits, MetadataResource,
    MetadataSelection, append, check, validate_type,
};

/// Observes the same borrowed transport events as `GpuDecodeStream`.
///
/// Only selected payloads are copied, into one vector per box rather than one retained object
/// per input chunk. Unselected `brob` boxes use a four-byte inline type probe. No codestream or
/// caller-owned input allocation is retained. An error drops all accumulated metadata immediately.
#[derive(Debug)]
pub struct MetadataCollector {
    selection: MetadataSelection,
    limits: MetadataLimits,
    metadata: Metadata,
    active: Option<ActiveBox>,
    ended: bool,
    failed: bool,
}

#[derive(Debug)]
struct ActiveBox {
    header: ContainerStreamBoxHeader,
    received: u64,
    prefix: [u8; 4],
    prefix_len: usize,
    logical_type: Option<[u8; 4]>,
    selected: bool,
    payload: Vec<u8>,
}

impl MetadataCollector {
    #[must_use]
    pub fn new(selection: MetadataSelection, limits: MetadataLimits) -> Self {
        Self {
            selection,
            limits,
            metadata: Metadata::default(),
            active: None,
            ended: false,
            failed: false,
        }
    }

    /// Logical payload bytes copied so far, including the current selected box.
    #[must_use]
    pub fn retained_bytes(&self) -> u64 {
        self.metadata.retained_bytes
            + self
                .active
                .as_ref()
                .map_or(0, |active| active.payload.len() as u64)
    }

    pub fn push_transport_event(
        &mut self,
        event: &ContainerStreamEvent,
    ) -> Result<(), MetadataError> {
        if self.failed {
            return Err(MetadataError::CollectorFailed);
        }
        if self.ended {
            return Err(MetadataError::CollectorFinished);
        }
        if let Err(error) = self.observe(event) {
            self.failed = true;
            self.active = None;
            self.metadata = Metadata::default();
            return Err(error);
        }
        Ok(())
    }

    /// Requires the scanner's final `End`; completed earlier boxes alone are not whole-file proof.
    pub fn finish(self) -> Result<Metadata, MetadataError> {
        if self.failed {
            return Err(MetadataError::CollectorFailed);
        }
        if !self.ended {
            return Err(MetadataError::IncompleteTransport);
        }
        Ok(self.metadata)
    }

    fn observe(&mut self, event: &ContainerStreamEvent) -> Result<(), MetadataError> {
        match event {
            ContainerStreamEvent::AuxiliaryBoxStart(header) => {
                if self.active.is_some() {
                    return Err(MetadataError::EventContract);
                }
                validate_type(header.box_type, false)?;
                let logical_type = (header.box_type != BROB).then_some(header.box_type);
                let mut active = ActiveBox {
                    header: *header,
                    received: 0,
                    prefix: [0; 4],
                    prefix_len: 0,
                    logical_type,
                    selected: false,
                    payload: Vec::new(),
                };
                if let Some(box_type) = logical_type {
                    self.select(&mut active, box_type)?;
                }
                self.active = Some(active);
            }
            ContainerStreamEvent::AuxiliaryBoxChunk {
                box_type,
                payload_offset,
                bytes,
            } => {
                let mut active = self.active.take().ok_or(MetadataError::EventContract)?;
                if *box_type != active.header.box_type || *payload_offset != active.received {
                    return Err(MetadataError::EventContract);
                }
                active.received = active
                    .received
                    .checked_add(bytes.len() as u64)
                    .ok_or(MetadataError::SizeOverflow)?;
                if active
                    .header
                    .payload_bytes
                    .is_some_and(|expected| active.received > expected)
                {
                    return Err(MetadataError::EventContract);
                }
                let mut payload = bytes.bytes();
                if active.logical_type.is_none() {
                    let copied = (4 - active.prefix_len).min(payload.len());
                    active.prefix[active.prefix_len..active.prefix_len + copied]
                        .copy_from_slice(&payload[..copied]);
                    active.prefix_len += copied;
                    payload = &payload[copied..];
                    if active.prefix_len == 4 {
                        let inner = active.prefix;
                        validate_type(inner, true)?;
                        self.select(&mut active, inner)?;
                        active.logical_type = Some(inner);
                        if active.selected {
                            self.append(&mut active, &inner)?;
                        }
                    }
                }
                if active.selected {
                    self.append(&mut active, payload)?;
                }
                self.active = Some(active);
            }
            ContainerStreamEvent::AuxiliaryBoxEnd { box_type } => {
                let active = self.active.take().ok_or(MetadataError::EventContract)?;
                if *box_type != active.header.box_type
                    || active
                        .header
                        .payload_bytes
                        .is_some_and(|expected| active.received != expected)
                {
                    return Err(MetadataError::EventContract);
                }
                let logical_type = active
                    .logical_type
                    .ok_or(MetadataError::TruncatedBrotliType)?;
                if active.selected {
                    self.metadata.push(
                        MetadataBox {
                            wire_type: *box_type,
                            logical_type,
                            payload: active.payload,
                        },
                        self.limits,
                    )?;
                }
            }
            ContainerStreamEvent::CodestreamChunk { .. } => {
                if self.active.is_some() {
                    return Err(MetadataError::EventContract);
                }
            }
            ContainerStreamEvent::End { .. } => {
                if self.active.is_some() {
                    return Err(MetadataError::IncompleteTransport);
                }
                self.ended = true;
            }
        }
        Ok(())
    }

    fn select(&self, active: &mut ActiveBox, box_type: [u8; 4]) -> Result<(), MetadataError> {
        active.selected = self.selection.includes(box_type);
        if active.selected {
            check(
                MetadataResource::BoxCount,
                self.metadata.boxes.len() as u64 + 1,
                self.limits.max_boxes as u64,
            )?;
            if let Some(length) = active.header.payload_bytes {
                check(
                    MetadataResource::EncodedBoxBytes,
                    length,
                    self.limits.max_encoded_box_bytes,
                )?;
                let total = self
                    .metadata
                    .retained_bytes
                    .checked_add(length)
                    .ok_or(MetadataError::SizeOverflow)?;
                check(
                    MetadataResource::RetainedBytes,
                    total,
                    self.limits.max_retained_bytes,
                )?;
            }
        }
        Ok(())
    }

    fn append(&self, active: &mut ActiveBox, bytes: &[u8]) -> Result<(), MetadataError> {
        let size = (active.payload.len() as u64)
            .checked_add(bytes.len() as u64)
            .ok_or(MetadataError::SizeOverflow)?;
        let total = self
            .metadata
            .retained_bytes
            .checked_add(size)
            .ok_or(MetadataError::SizeOverflow)?;
        check(
            MetadataResource::RetainedBytes,
            total,
            self.limits.max_retained_bytes,
        )?;
        append(
            &mut active.payload,
            bytes,
            self.limits.max_encoded_box_bytes,
            MetadataResource::EncodedBoxBytes,
        )
    }
}
