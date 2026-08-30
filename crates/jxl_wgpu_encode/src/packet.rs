use std::collections::BTreeMap;

use jxl_gpu_bitstream::BitWriter;

use crate::PacketError;

const MAX_TOC_ENTRIES: usize = 65_536;
const TOC_BUCKETS: [(u32, u8); 4] = [(0, 10), (1_024, 14), (17_408, 22), (4_211_712, 30)];

/// An LSB-first JPEG XL bit fragment with validated zero padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitFragment {
    bytes: Vec<u8>,
    bit_len: usize,
}

impl BitFragment {
    pub fn new(bytes: Vec<u8>, bit_len: usize) -> Result<Self, PacketError> {
        let expected = bit_len.checked_add(7).ok_or(PacketError::SizeOverflow)? / 8;
        if bytes.len() != expected {
            return Err(PacketError::BitLength {
                bit_len,
                expected,
                actual: bytes.len(),
            });
        }
        let used_bits = bit_len % 8;
        if used_bits != 0 {
            let valid_mask = (1u8 << used_bits) - 1;
            if bytes.last().is_some_and(|last| last & !valid_mask != 0) {
                return Err(PacketError::NonZeroPadding);
            }
        }
        Ok(Self { bytes, bit_len })
    }

    #[must_use]
    pub fn byte_aligned(bytes: Vec<u8>) -> Self {
        let bit_len = bytes.len().saturating_mul(8);
        Self { bytes, bit_len }
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub const fn bit_len(&self) -> usize {
        self.bit_len
    }

    #[must_use]
    pub const fn is_byte_aligned(&self) -> bool {
        self.bit_len.is_multiple_of(8)
    }
}

/// The canonical JPEG XL section layout of one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameGroupLayout {
    dc_groups: u32,
    ac_groups: u32,
    passes: u8,
    fused_single_group: bool,
}

impl FrameGroupLayout {
    pub fn new(dc_groups: u32, ac_groups: u32, passes: u8) -> Result<Self, PacketError> {
        if dc_groups == 0 || ac_groups == 0 || passes == 0 {
            return Err(PacketError::EmptyLayout);
        }
        let fused_single_group = ac_groups == 1 && passes == 1;
        let entries = if fused_single_group {
            1u64
        } else {
            2u64.checked_add(u64::from(dc_groups))
                .and_then(|value| value.checked_add(u64::from(ac_groups) * u64::from(passes)))
                .ok_or(PacketError::TooManyGroups)?
        };
        if entries > MAX_TOC_ENTRIES as u64 {
            return Err(PacketError::TooManyGroups);
        }
        Ok(Self {
            dc_groups,
            ac_groups,
            passes,
            fused_single_group,
        })
    }

    #[must_use]
    pub const fn dc_groups(self) -> u32 {
        self.dc_groups
    }

    #[must_use]
    pub const fn ac_groups(self) -> u32 {
        self.ac_groups
    }

    #[must_use]
    pub const fn passes(self) -> u8 {
        self.passes
    }

    #[must_use]
    pub const fn is_fused_single_group(self) -> bool {
        self.fused_single_group
    }

    #[must_use]
    pub fn toc_entries(self) -> usize {
        if self.fused_single_group {
            1
        } else {
            2 + self.dc_groups as usize + self.ac_groups as usize * usize::from(self.passes)
        }
    }

    fn canonical_index(self, kind: GroupPacketKind) -> Option<usize> {
        if self.fused_single_group {
            return (kind == GroupPacketKind::Single).then_some(0);
        }
        match kind {
            GroupPacketKind::DcGlobal => Some(0),
            GroupPacketKind::DcGroup(group) if group < self.dc_groups => Some(1 + group as usize),
            GroupPacketKind::AcGlobal => Some(1 + self.dc_groups as usize),
            GroupPacketKind::AcGroup { pass, group }
                if pass < self.passes && group < self.ac_groups =>
            {
                Some(
                    2 + self.dc_groups as usize
                        + usize::from(pass) * self.ac_groups as usize
                        + group as usize,
                )
            }
            _ => None,
        }
    }

    fn kind_at(self, index: usize) -> GroupPacketKind {
        if self.fused_single_group {
            return GroupPacketKind::Single;
        }
        if index == 0 {
            return GroupPacketKind::DcGlobal;
        }
        let dc_end = 1 + self.dc_groups as usize;
        if index < dc_end {
            return GroupPacketKind::DcGroup((index - 1) as u32);
        }
        if index == dc_end {
            return GroupPacketKind::AcGlobal;
        }
        let ac_index = index - dc_end - 1;
        GroupPacketKind::AcGroup {
            pass: (ac_index / self.ac_groups as usize) as u8,
            group: (ac_index % self.ac_groups as usize) as u32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GroupPacketKind {
    /// Optimized one-group/one-pass frame where all sections share one packet.
    Single,
    DcGlobal,
    DcGroup(u32),
    AcGlobal,
    AcGroup {
        pass: u8,
        group: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupPacket {
    pub kind: GroupPacketKind,
    pub payload: Vec<u8>,
}

impl GroupPacket {
    #[must_use]
    pub fn new(kind: GroupPacketKind, payload: Vec<u8>) -> Self {
        Self { kind, payload }
    }
}

/// Validated packets stored in canonical TOC order, independent of GPU job
/// completion order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FramePacketSet {
    pub frame_header: BitFragment,
    pub layout: FrameGroupLayout,
    packets: Vec<GroupPacket>,
}

impl FramePacketSet {
    pub fn new(
        frame_header: BitFragment,
        layout: FrameGroupLayout,
        packets: impl IntoIterator<Item = GroupPacket>,
    ) -> Result<Self, PacketError> {
        let mut indexed = BTreeMap::new();
        for packet in packets {
            let index = layout
                .canonical_index(packet.kind)
                .ok_or(PacketError::InvalidKind {
                    kind: packet.kind,
                    layout,
                })?;
            if indexed.insert(index, packet).is_some() {
                return Err(PacketError::Duplicate(layout.kind_at(index)));
            }
        }
        let mut ordered = Vec::with_capacity(layout.toc_entries());
        for index in 0..layout.toc_entries() {
            ordered.push(
                indexed
                    .remove(&index)
                    .ok_or_else(|| PacketError::Missing(layout.kind_at(index)))?,
            );
        }
        Ok(Self {
            frame_header,
            layout,
            packets: ordered,
        })
    }

    #[must_use]
    pub fn packets(&self) -> &[GroupPacket] {
        &self.packets
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedFrame {
    bytes: Vec<u8>,
    packet_count: usize,
}

impl EncodedFrame {
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub const fn packet_count(&self) -> usize {
        self.packet_count
    }

    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Serializes frame header bits, an absent TOC permutation, canonical TOC
/// sizes, and byte-aligned group packets. Image and coefficient work is not
/// performed here.
pub fn assemble_frame(packet_set: FramePacketSet) -> Result<EncodedFrame, PacketError> {
    let mut writer = BitWriter::new();
    append_fragment(&mut writer, &packet_set.frame_header)?;
    writer
        .write_bits(0, 1)
        .map_err(|_| PacketError::SizeOverflow)?;
    writer
        .align_to_byte()
        .map_err(|_| PacketError::SizeOverflow)?;
    for packet in &packet_set.packets {
        let size = u32::try_from(packet.payload.len()).map_err(|_| PacketError::PacketTooLarge)?;
        write_toc_size(&mut writer, size)?;
    }
    writer
        .align_to_byte()
        .map_err(|_| PacketError::SizeOverflow)?;
    let mut bytes = writer.into_bytes();
    let payload_bytes = packet_set
        .packets
        .iter()
        .try_fold(0usize, |total, packet| {
            total
                .checked_add(packet.payload.len())
                .ok_or(PacketError::SizeOverflow)
        })?;
    bytes
        .try_reserve(payload_bytes)
        .map_err(|_| PacketError::SizeOverflow)?;
    for packet in packet_set.packets {
        bytes.extend_from_slice(&packet.payload);
    }
    Ok(EncodedFrame {
        bytes,
        packet_count: packet_set.layout.toc_entries(),
    })
}

fn append_fragment(writer: &mut BitWriter, fragment: &BitFragment) -> Result<(), PacketError> {
    for bit_index in 0..fragment.bit_len {
        let bit = (fragment.bytes[bit_index / 8] >> (bit_index % 8)) & 1;
        writer
            .write_bits(u64::from(bit), 1)
            .map_err(|_| PacketError::SizeOverflow)?;
    }
    Ok(())
}

fn write_toc_size(writer: &mut BitWriter, size: u32) -> Result<(), PacketError> {
    let (selector, offset, bits) = TOC_BUCKETS
        .iter()
        .copied()
        .enumerate()
        .find(|(_, (offset, bits))| size >= *offset && u64::from(size - *offset) < (1u64 << *bits))
        .map(|(selector, (offset, bits))| (selector as u64, offset, bits))
        .ok_or(PacketError::PacketTooLarge)?;
    writer
        .write_bits(selector, 2)
        .and_then(|()| writer.write_bits(u64::from(size - offset), bits))
        .map_err(|_| PacketError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use jxl_gpu_bitstream::BitReader;

    use super::*;

    #[test]
    fn rejects_non_zero_fragment_padding() {
        assert_eq!(
            BitFragment::new(vec![0b1110_0101], 3).unwrap_err(),
            PacketError::NonZeroPadding
        );
    }

    #[test]
    fn canonicalizes_parallel_group_completion_order() {
        let layout = FrameGroupLayout::new(1, 2, 1).unwrap();
        let set = FramePacketSet::new(
            BitFragment::new(Vec::new(), 0).unwrap(),
            layout,
            [
                GroupPacket::new(GroupPacketKind::AcGroup { pass: 0, group: 1 }, vec![4]),
                GroupPacket::new(GroupPacketKind::AcGlobal, vec![2]),
                GroupPacket::new(GroupPacketKind::DcGlobal, vec![0]),
                GroupPacket::new(GroupPacketKind::AcGroup { pass: 0, group: 0 }, vec![3]),
                GroupPacket::new(GroupPacketKind::DcGroup(0), vec![1]),
            ],
        )
        .unwrap();
        let kinds: Vec<_> = set.packets().iter().map(|packet| packet.kind).collect();
        assert_eq!(
            kinds,
            [
                GroupPacketKind::DcGlobal,
                GroupPacketKind::DcGroup(0),
                GroupPacketKind::AcGlobal,
                GroupPacketKind::AcGroup { pass: 0, group: 0 },
                GroupPacketKind::AcGroup { pass: 0, group: 1 },
            ]
        );
        let encoded = assemble_frame(set).unwrap();
        assert_eq!(
            &encoded.bytes()[encoded.bytes().len() - 5..],
            &[0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn toc_uses_reference_bucket_boundaries() {
        let mut writer = BitWriter::new();
        for size in [0, 1_023, 1_024, 17_407, 17_408, 4_211_711, 4_211_712] {
            write_toc_size(&mut writer, size).unwrap();
        }
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        for expected in [0, 1_023, 1_024, 17_407, 17_408, 4_211_711, 4_211_712] {
            let selector = reader.read_bits(2).unwrap() as usize;
            let (offset, bits) = TOC_BUCKETS[selector];
            let actual = offset + reader.read_bits(bits).unwrap() as u32;
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn one_group_one_pass_uses_fused_packet() {
        let layout = FrameGroupLayout::new(1, 1, 1).unwrap();
        assert!(layout.is_fused_single_group());
        assert_eq!(layout.toc_entries(), 1);
        FramePacketSet::new(
            BitFragment::new(Vec::new(), 0).unwrap(),
            layout,
            [GroupPacket::new(GroupPacketKind::Single, vec![1, 2, 3])],
        )
        .unwrap();
    }
}
