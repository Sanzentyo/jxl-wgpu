use std::sync::Arc;

use super::{IccError, IccLimits, IccRenderingIntent, IccSignature};

/// Header values used by transform selection. Original bytes remain available on the profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IccHeader {
    pub version: u32,
    pub class: IccSignature,
    pub device_space: IccSignature,
    pub pcs: IccSignature,
    pub rendering_intent: IccRenderingIntent,
    /// Exact signed s15Fixed16 PCS illuminant (the ICC D50 encoding).
    pub illuminant: [i32; 3],
}

/// Checked tag range in the original profile. Different tags may share one entire element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IccTag {
    pub signature: IccSignature,
    pub kind: IccSignature,
    pub offset: u32,
    pub size: u32,
}

#[derive(Clone, Debug)]
pub struct IccProfile {
    bytes: Arc<[u8]>,
    header: IccHeader,
    tags: Arc<[IccTag]>,
    max_curve_samples: u32,
}

// A resource policy does not change the profile's color meaning or identity.
impl PartialEq for IccProfile {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}
impl Eq for IccProfile {}
impl std::hash::Hash for IccProfile {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.bytes, state);
    }
}

impl IccProfile {
    /// Validates an owned profile without copying its payload. Tag metadata allocations are
    /// bounded first. Uninterpreted private tags remain accessible as their original bytes.
    pub fn parse(bytes: Arc<[u8]>, limits: IccLimits) -> Result<Self, IccError> {
        let data = Reader(&bytes);
        limit(
            "profile bytes",
            bytes.len() as u64,
            limits.max_profile_bytes,
        )?;
        data.slice(0, 132, "header and tag count")?;
        if u64::from(data.u32(0)?) != bytes.len() as u64 {
            return invalid("declared profile size", 0);
        }
        if data.signature(36)? != IccSignature(*b"acsp") {
            return invalid("profile signature", 36);
        }
        let version = data.u32(8)?;
        let major = version >> 24;
        if !matches!(major, 2 | 4) || (major == 4 && version > 0x0440_0000) {
            return Err(IccError::Version { version });
        }
        if version & 0xffff != 0 || (version >> 20) & 15 > 9 || (version >> 16) & 15 > 9 {
            return invalid("version encoding", 8);
        }
        data.zeros(if major == 2 { 84 } else { 100 }, 128, "reserved header")?;
        let rendering_intent = match data.u32(64)? {
            0 => IccRenderingIntent::Perceptual,
            1 => IccRenderingIntent::Relative,
            2 => IccRenderingIntent::Saturation,
            3 => IccRenderingIntent::Absolute,
            _ => return invalid("rendering intent", 64),
        };
        let illuminant = [data.i32(68)?, data.i32(72)?, data.i32(76)?];
        if illuminant != [0xf6d6, 0x10000, 0xd32d] {
            return invalid("PCS D50 illuminant", 68);
        }
        let header = IccHeader {
            version,
            class: data.signature(12)?,
            device_space: data.signature(16)?,
            pcs: data.signature(20)?,
            rendering_intent,
            illuminant,
        };
        let count = data.u32(128)?;
        limit("tag count", u64::from(count), u64::from(limits.max_tags))?;
        let table_end = 132 + u64::from(count) * 12;
        data.slice(132, table_end, "tag table")?;
        let mut tags = Vec::with_capacity(count as usize);
        for index in 0..u64::from(count) {
            let entry = 132 + 12 * index;
            let signature = data.signature(entry)?;
            let offset = data.u32(entry + 4)?;
            let size = data.u32(entry + 8)?;
            if !offset.is_multiple_of(4) || u64::from(offset) < table_end || size < 8 {
                return invalid("tag range", entry + 4);
            }
            data.slice(
                u64::from(offset),
                u64::from(offset) + u64::from(size),
                "tag element",
            )?;
            let kind = data.signature(u64::from(offset))?;
            data.zeros(u64::from(offset) + 4, u64::from(offset) + 8, "reserved tag")?;
            tags.push(IccTag {
                signature,
                kind,
                offset,
                size,
            });
        }
        tags.sort_unstable_by_key(|tag| (tag.offset, tag.size));
        let mut previous: Option<IccTag> = None;
        let mut contiguous_end = table_end;
        for tag in &tags {
            if let Some(prior) = previous {
                let prior_end = u64::from(prior.offset) + u64::from(prior.size);
                if u64::from(tag.offset) < prior_end {
                    if (tag.offset, tag.size) == (prior.offset, prior.size) {
                        continue;
                    }
                    return Err(IccError::TagOverlap {
                        first: prior.signature,
                        second: tag.signature,
                    });
                }
            }
            if version >= 0x0440_0000 && u64::from(tag.offset) != contiguous_end {
                return invalid("noncontiguous v4.4 tag data", u64::from(tag.offset));
            }
            let end = u64::from(tag.offset) + u64::from(tag.size);
            contiguous_end = end.next_multiple_of(4);
            data.zeros(end, contiguous_end.min(bytes.len() as u64), "tag padding")?;
            previous = Some(*tag);
        }
        if version >= 0x0440_0000 && bytes.len() as u64 != contiguous_end {
            return invalid("v4.4 profile padding", contiguous_end);
        }
        tags.sort_unstable_by_key(|tag| tag.signature);
        for pair in tags.windows(2) {
            if pair[0].signature == pair[1].signature {
                return Err(IccError::DuplicateTag {
                    tag: pair[0].signature,
                });
            }
        }
        Ok(Self {
            bytes,
            header,
            tags: tags.into(),
            max_curve_samples: limits.max_curve_samples,
        })
    }

    #[must_use]
    pub const fn header(&self) -> &IccHeader {
        &self.header
    }

    #[must_use]
    pub fn bytes(&self) -> &Arc<[u8]> {
        &self.bytes
    }

    /// Tags ordered by signature, independent of their physical payload order.
    #[must_use]
    pub fn tags(&self) -> &[IccTag] {
        &self.tags
    }

    #[must_use]
    pub fn tag(&self, signature: IccSignature) -> Option<&IccTag> {
        self.tags
            .binary_search_by_key(&signature, |tag| tag.signature)
            .ok()
            .map(|i| &self.tags[i])
    }

    #[must_use]
    pub fn tag_data(&self, signature: IccSignature) -> Option<&[u8]> {
        self.tag(signature).map(|tag| {
            &self.bytes[tag.offset as usize..(u64::from(tag.offset) + u64::from(tag.size)) as usize]
        })
    }

    pub(super) fn required(&self, signature: IccSignature) -> Result<Reader<'_>, IccError> {
        self.tag_data(signature)
            .map(Reader)
            .ok_or(IccError::MissingTag { tag: signature })
    }

    pub(super) const fn max_curve_samples(&self) -> u32 {
        self.max_curve_samples
    }
}

pub(super) struct Reader<'a>(pub(super) &'a [u8]);

impl<'a> Reader<'a> {
    pub(super) fn slice(
        &self,
        start: u64,
        end: u64,
        field: &'static str,
    ) -> Result<&'a [u8], IccError> {
        if start > end || end > self.0.len() as u64 {
            return Err(IccError::Truncated {
                field,
                required: end,
                available: self.0.len() as u64,
            });
        }
        Ok(&self.0[start as usize..end as usize])
    }

    pub(super) fn u32(&self, offset: u64) -> Result<u32, IccError> {
        Ok(u32::from_be_bytes(
            self.slice(offset, offset + 4, "u32")?
                .try_into()
                .expect("checked four bytes"),
        ))
    }

    pub(super) fn i32(&self, offset: u64) -> Result<i32, IccError> {
        self.u32(offset).map(|v| v as i32)
    }

    pub(super) fn u16(&self, offset: u64) -> Result<u16, IccError> {
        Ok(u16::from_be_bytes(
            self.slice(offset, offset + 2, "u16")?
                .try_into()
                .expect("checked two bytes"),
        ))
    }

    pub(super) fn signature(&self, offset: u64) -> Result<IccSignature, IccError> {
        Ok(IccSignature(self.u32(offset)?.to_be_bytes()))
    }

    pub(super) fn zeros(&self, start: u64, end: u64, field: &'static str) -> Result<(), IccError> {
        if self.slice(start, end, field)?.iter().any(|&v| v != 0) {
            return invalid(field, start);
        }
        Ok(())
    }
}

pub(super) fn invalid<T>(field: &'static str, offset: u64) -> Result<T, IccError> {
    Err(IccError::Invalid { field, offset })
}

pub(super) fn limit(resource: &'static str, required: u64, limit: u64) -> Result<(), IccError> {
    if required > limit {
        return Err(IccError::Limit {
            resource,
            required,
            limit,
        });
    }
    Ok(())
}
