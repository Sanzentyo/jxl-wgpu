use super::*;

fn packets(layout: FrameGroupLayout) -> FramePacketSet {
    FramePacketSet::new(
        BitFragment::new(vec![21], 5).unwrap(),
        layout,
        (0..layout.toc_entries()).map(|index| {
            let mut payload = (index as u32).to_le_bytes().to_vec();
            payload.resize(index % 11, index as u8);
            GroupPacket::new(layout.kind_at(index), payload)
        }),
    )
    .unwrap()
}

#[test]
fn physical_toc_orders_round_trip_through_independent_entropy_decoder() {
    for (dc, ac, passes) in [
        (1, 1, 1),
        (1, 1, 11),
        (2, 7, 3),
        (64, 4096, 11),
        (1, 65_533, 1),
    ] {
        let layout = FrameGroupLayout::new(dc, ac, passes).unwrap();
        let original = packets(layout);
        let count = layout.toc_entries();
        for pattern in 0..4 {
            let mut physical: Vec<_> = (0..count).collect();
            match pattern {
                0 => {}
                1 => physical.reverse(),
                2 => physical.rotate_left(count / 3),
                _ => {
                    let mut state = 0x3141_5926u32;
                    for index in (1..count).rev() {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        physical.swap(index, state as usize % (index + 1));
                    }
                }
            }
            let ordered = original
                .clone()
                .with_order(physical.iter().map(|&i| layout.kind_at(i)))
                .unwrap();
            assert_eq!(ordered.packets(), original.packets());
            assert_eq!(
                ordered
                    .packets_in_file_order()
                    .map(|p| p.kind)
                    .collect::<Vec<_>>(),
                physical
                    .iter()
                    .map(|&i| layout.kind_at(i))
                    .collect::<Vec<_>>()
            );
            let encoded = assemble_frame(ordered).unwrap();
            assert_eq!(encoded.packet_count(), count);
            let mut bits = jxl_bitstream::Bitstream::new(encoded.bytes());
            assert_eq!(bits.read_bits(5).unwrap(), 21);
            let permutation = if bits.read_bits(1).unwrap() != 0 {
                let mut decoder = jxl_coding::Decoder::parse(&mut bits, 8).unwrap();
                decoder.begin(&mut bits).unwrap();
                let permutation =
                    jxl_coding::read_permutation(&mut bits, &mut decoder, count as u32, 0).unwrap();
                decoder.finalize().unwrap();
                permutation
            } else {
                assert!(physical.iter().copied().eq(0..count));
                assert_eq!(encoded, assemble_frame(original.clone()).unwrap());
                (0..count).collect()
            };
            let mut inverse = vec![0; count];
            for (position, &canonical) in physical.iter().enumerate() {
                inverse[canonical] = position;
            }
            assert_eq!(permutation, inverse);
            bits.zero_pad_to_byte().unwrap();
            let sizes = (0..count)
                .map(|_| {
                    let (offset, width) = TOC_BUCKETS[bits.read_bits(2).unwrap() as usize];
                    (offset + bits.read_bits(usize::from(width)).unwrap()) as usize
                })
                .collect::<Vec<_>>();
            bits.zero_pad_to_byte().unwrap();
            let payload_start = bits.num_read_bits() / 8;
            let mut offsets = vec![payload_start];
            for (position, size) in sizes.iter().copied().enumerate() {
                assert_eq!(size, original.packets()[physical[position]].payload.len());
                offsets.push(offsets.last().unwrap() + size);
            }
            assert_eq!(*offsets.last().unwrap(), encoded.bytes().len());
            for (canonical, packet) in original.packets().iter().enumerate() {
                let position = permutation[canonical];
                assert_eq!(
                    &encoded.bytes()[offsets[position]..offsets[position + 1]],
                    packet.payload
                );
            }
        }
    }
}

#[test]
fn packet_order_rejects_missing_duplicate_invalid_and_unbounded_entries() {
    let layout = FrameGroupLayout::new(1, 2, 2).unwrap();
    let set = packets(layout);
    let count = layout.toc_entries();
    let kinds: Vec<_> = set.packets().iter().map(|p| p.kind).collect();
    assert_eq!(
        set.clone()
            .with_order(kinds[..count - 1].iter().copied())
            .unwrap_err(),
        PacketError::OrderLength {
            expected: count,
            actual: count - 1
        }
    );
    let mut duplicate = kinds.clone();
    duplicate[count - 1] = duplicate[0];
    assert_eq!(
        set.clone().with_order(duplicate).unwrap_err(),
        PacketError::DuplicateOrder(kinds[0])
    );
    for invalid in [
        GroupPacketKind::Single,
        GroupPacketKind::DcGroup(1),
        GroupPacketKind::AcGroup { pass: 2, group: 0 },
        GroupPacketKind::AcGroup { pass: 0, group: 2 },
    ] {
        let mut order = kinds.clone();
        order[count - 1] = invalid;
        assert_eq!(
            set.clone().with_order(order).unwrap_err(),
            PacketError::InvalidKind {
                kind: invalid,
                layout
            }
        );
    }
    // An unbounded producer is rejected after the first extra entry.
    let reads = std::cell::Cell::new(0);
    let infinite = kinds
        .iter()
        .copied()
        .cycle()
        .inspect(|_| reads.set(reads.get() + 1));
    assert_eq!(
        set.with_order(infinite).unwrap_err(),
        PacketError::OrderLength {
            expected: count,
            actual: count + 1
        }
    );
    assert_eq!(reads.get(), count + 1);
}

#[test]
fn packet_order_cannot_be_reinterpreted_by_mutating_public_layout() {
    let layout = FrameGroupLayout::new(1, 2, 2).unwrap();
    for replacement in [
        FrameGroupLayout::new(1, 3, 2).unwrap(),
        FrameGroupLayout::new(1, 4, 1).unwrap(),
    ] {
        let mut set = packets(layout)
            .with_order((0..layout.toc_entries()).rev().map(|i| layout.kind_at(i)))
            .unwrap();
        set.layout = replacement;
        assert!(assemble_frame(set.clone()).is_err());
        assert!(set.with_order(std::iter::empty()).is_err());
    }
}
