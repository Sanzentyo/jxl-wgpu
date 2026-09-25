//! Independent adversarial source packing shared by encoder conformance targets.
use jxl_gpu_formats::{
    ByteOrder, Channel, ImageLayout, PackingField, PackingFieldKind, PackingWord, PixelFormat,
    PlaneFormat, PlaneSampling, Swizzle, SwizzleComponent,
};
use jxl_gpu_protocol::Extent2d;

#[derive(Clone, Copy, Debug)]
pub enum Storage {
    Packed,
    Planar,
    Split,
    SharedWord,
    ThreeBytes,
    MixedWords,
}

/// Physical storage choices; logical samples remain in canonical channel order.
#[derive(Clone, Copy, Debug)]
pub struct Packing {
    pub storage: Storage,
    pub reversed: bool,
    pub shifted: bool,
}

struct Sample {
    logical: usize,
    word: usize,
    shift: u8,
}

impl Packing {
    /// Pack canonical logical samples independently of the encoder's source planner.
    /// Source padding, row gaps and inter-plane gaps are deliberately poisoned.
    pub fn pack(
        self,
        mut pixel_format: PixelFormat,
        extent: Extent2d,
        expected: &[u32],
        plane_gap: u64,
    ) -> (ImageLayout, Vec<u8>) {
        assert_eq!(pixel_format.planes.len(), 1);
        let ids: Vec<_> = pixel_format.planes[0]
            .words
            .iter()
            .flat_map(|word| &word.fields)
            .filter_map(|field| match field.kind {
                PackingFieldKind::Channel(channel) => Some(channel),
                PackingFieldKind::Padding => None,
            })
            .collect();
        let channels = ids.len();
        let gray_alpha = ids == [Channel::X, Channel::W];
        let bits = pixel_format.planes[0].words[0]
            .fields
            .iter()
            .find(|field| matches!(field.kind, PackingFieldKind::Channel(_)))
            .unwrap()
            .bits;
        let byte_order = pixel_format.byte_order;
        assert_eq!(
            expected.len(),
            extent.width as usize * extent.height as usize * channels
        );
        let mut order: Vec<_> = (0..channels).collect();
        if self.reversed && gray_alpha {
            order.reverse();
            pixel_format.swizzle = Swizzle::Xyzw([
                SwizzleComponent::W,
                SwizzleComponent::Zero,
                SwizzleComponent::Zero,
                SwizzleComponent::X,
            ]);
        }
        if self.reversed && channels >= 3 {
            order[..3].reverse();
            pixel_format.swizzle = if channels == 4 {
                Swizzle::ZYXW
            } else {
                Swizzle::ZYX1
            };
            if matches!(self.storage, Storage::Split) && channels == 4 {
                order = vec![3, 0, 1, 2];
                pixel_format.swizzle = Swizzle::Xyzw([
                    SwizzleComponent::Y,
                    SwizzleComponent::Z,
                    SwizzleComponent::W,
                    SwizzleComponent::X,
                ]);
            }
        }
        let physical_planes: Vec<Vec<usize>> = match self.storage {
            Storage::Planar => (0..channels).map(|index| vec![index]).collect(),
            Storage::Split if channels > 1 => vec![(0..channels - 1).collect(), vec![channels - 1]],
            _ => vec![(0..channels).collect()],
        };
        let mut assignments = Vec::new();
        pixel_format.planes = physical_planes
            .iter()
            .map(|physical| {
                let mut samples = Vec::new();
                let words = if matches!(self.storage, Storage::SharedWord) {
                    let storage_bits = (channels as u8 * bits + 1).div_ceil(8) * 8;
                    assert!(storage_bits <= 32);
                    let mut fields = Vec::new();
                    for (index, &position) in physical.iter().enumerate() {
                        let shift = storage_bits - bits * (index as u8 + 1);
                        fields.push(PackingField::channel(ids[position], bits));
                        samples.push(Sample {
                            logical: order[position],
                            word: 0,
                            shift,
                        });
                    }
                    fields.push(PackingField::padding(storage_bits - channels as u8 * bits));
                    vec![PackingWord { fields }]
                } else {
                    physical
                        .iter()
                        .enumerate()
                        .map(|(word, &position)| {
                            let storage_bits = match self.storage {
                                Storage::ThreeBytes => 24,
                                Storage::MixedWords => [8, 16, 24, 32][position],
                                _ if self.shifted && bits == 16 => 32,
                                _ => bits.next_power_of_two().max(8),
                            };
                            let shift = if self.shifted {
                                (storage_bits - bits).div_ceil(2)
                            } else {
                                0
                            };
                            let mut fields = Vec::new();
                            if storage_bits > bits + shift {
                                fields.push(PackingField::padding(storage_bits - bits - shift));
                            }
                            fields.push(PackingField::channel(ids[position], bits));
                            if shift != 0 {
                                fields.push(PackingField::padding(shift));
                            }
                            samples.push(Sample {
                                logical: order[position],
                                word,
                                shift,
                            });
                            PackingWord { fields }
                        })
                        .collect()
                };
                assignments.push(samples);
                PlaneFormat {
                    sampling: PlaneSampling::FULL,
                    pixels_per_element: 1,
                    words,
                }
            })
            .collect();
        let mut layout = ImageLayout::packed(extent, pixel_format.clone()).unwrap();
        // Store planes in reverse physical order, with unrelated unaligned pitches and poisoned gaps.
        let mut end = 5;
        for index in (0..layout.planes.len()).rev() {
            let plane = &mut layout.planes[index];
            plane.offset = end;
            plane.row_stride = plane.row_bytes + 5 + 2 * index as u64;
            end = plane.end_offset().unwrap() + plane_gap + 3;
        }
        let layout = ImageLayout::from_planes(extent, pixel_format, layout.planes).unwrap();
        let mut bytes = vec![0xa5; layout.logical_size.div_ceil(4) as usize * 4];
        let mask = u32::MAX >> (32 - bits);
        for (plane_index, plane) in layout.planes.iter().enumerate() {
            let word_widths: Vec<_> = layout.format.planes[plane_index]
                .words
                .iter()
                .map(|word| word.bits() as usize / 8)
                .collect();
            let pixel_bytes: usize = word_widths.iter().sum();
            for y in 0..extent.height {
                for x in 0..extent.width {
                    let mut packed = vec![u32::MAX; word_widths.len()];
                    for sample in &assignments[plane_index] {
                        let value =
                            expected[((y * extent.width + x) as usize) * channels + sample.logical];
                        packed[sample.word] = (packed[sample.word] & !(mask << sample.shift))
                            | (value << sample.shift);
                    }
                    let mut address = (plane.offset + u64::from(y) * plane.row_stride) as usize
                        + x as usize * pixel_bytes;
                    for (word, &width) in packed.into_iter().zip(&word_widths) {
                        if byte_order == ByteOrder::Big {
                            bytes[address..address + width]
                                .copy_from_slice(&word.to_be_bytes()[4 - width..]);
                        } else {
                            bytes[address..address + width]
                                .copy_from_slice(&word.to_le_bytes()[..width]);
                        }
                        address += width;
                    }
                }
            }
        }
        (layout, bytes)
    }
}
