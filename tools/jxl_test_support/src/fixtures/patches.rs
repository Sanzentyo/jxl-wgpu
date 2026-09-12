//! Patch dictionaries and reference/LF scenarios built from explicit frame-feature inputs.
use super::frame_features::{
    Frame, assemble_frame_sequence, assemble_frames, copy_bits, dictionary, flags, header,
    packet_frame_prefix,
};
use jxl_gpu_bitstream::{BitReader, BitWriter, FrameInventory};
use jxl_wgpu_encode::BitFragment;

pub fn values(frame: &FrameInventory, extras: usize, count: u32) -> Vec<u32> {
    values_using_modes(frame, extras, count, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Equivalent RGB operations for an image without any alpha or extra channels. Source-over
/// reduces to replacement and multiply-add reduces to addition when alpha is implicitly one.
pub fn arithmetic_values(frame: &FrameInventory, count: u32) -> Vec<u32> {
    values_using_modes(frame, 0, count, [0, 1, 2, 3, 1, 1, 2, 2])
}

fn values_using_modes(
    frame: &FrameInventory,
    extras: usize,
    count: u32,
    modes: [u32; 8],
) -> Vec<u32> {
    if count == 0 {
        return vec![0];
    }
    let width = frame.width.min(7);
    let height = frame.height.min(5);
    let mut result = vec![1, 3, 0, 0, width - 1, height - 1, count - 1];
    let mut previous = [0i32; 2];
    for index in 0..count {
        let position = [
            (index * 3 % (frame.width - width + 1)) as i32,
            (index * 2 % (frame.height - height + 1)) as i32,
        ];
        for axis in 0..2 {
            let value = if index == 0 {
                position[axis] as u32
            } else {
                let delta = position[axis] - previous[axis];
                ((delta << 1) ^ (delta >> 31)) as u32
            };
            result.push(value);
        }
        previous = position;
        for channel in 0..=extras {
            let mode = modes[(index as usize + channel) % 8];
            result.push(mode);
            if mode >= 4 && extras > 1 {
                result.push((index as usize + channel) as u32 % extras as u32);
            }
            if mode >= 3 {
                result.push((index / 8) % 2);
            }
        }
    }
    result
}

pub fn assemble(bytes: &[u8], patch_values: &[u32]) -> Vec<u8> {
    assemble_frames(&[
        Frame {
            codestream: bytes,
            reference: Some((3, true)),
            patches: None,
            splines: None,
        },
        Frame {
            codestream: bytes,
            reference: None,
            patches: Some(patch_values),
            splines: None,
        },
    ])
}

/// LF dependencies precede both the hidden patch reference and its visible consumer.
pub fn assemble_shared_lf(bytes: &[u8], patch_values: &[u32]) -> Vec<u8> {
    assemble_frame_sequence(
        &[
            Frame {
                codestream: bytes,
                reference: Some((3, true)),
                patches: None,
                splines: None,
            },
            Frame {
                codestream: bytes,
                reference: None,
                patches: Some(patch_values),
                splines: None,
            },
        ],
        false,
    )
}

/// First decode an unchanged LF chain and save its main frame as a component reference. Then
/// replace the LF slots with patch-bearing producers and decode the unchanged main consumer.
pub fn assemble_lf_producers(bytes: &[u8], dictionaries: &[Option<&[u32]>]) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(bytes, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let bytes = parsed.codestream();
    let main = inventory.frames.last().unwrap();
    let dependencies = &inventory.frames[..inventory.frames.len() - 1];
    assert_eq!(dependencies.len(), dictionaries.len());
    assert!(dependencies.iter().all(|frame| frame.lf_level != 0));
    let mut result = bytes[..main.header_bits.offset as usize / 8].to_vec();
    result.extend(packet_frame(
        bytes,
        main,
        header(&inventory.image_header, main, Some((3, true)), 0),
        None,
    ));
    for (frame, &values) in dependencies.iter().zip(dictionaries) {
        let header = if values.is_some() {
            let mut old_flags = BitWriter::new();
            flags(&mut old_flags, frame.flags);
            let mut writer = BitWriter::new();
            // Non-default, frame type and encoding precede the U64 flags. Preserve every
            // remaining header bit, including the LF level and native filter parameters.
            let start = frame.header_bits.offset;
            let mut reader = BitReader::new(bytes);
            reader.skip_bits(start).unwrap();
            assert_eq!(reader.read_bits(1).unwrap(), 0);
            reader.skip_bits(3).unwrap();
            let mut expected = BitReader::new(old_flags.as_bytes());
            assert_eq!(
                reader.read_bits(old_flags.bit_len() as u8).unwrap(),
                expected.read_bits(old_flags.bit_len() as u8).unwrap()
            );
            copy_bits(&mut writer, bytes, start, start + 4);
            flags(&mut writer, frame.flags | 2);
            copy_bits(
                &mut writer,
                bytes,
                start + 4 + old_flags.bit_len() as u64,
                frame.header_bits.end().unwrap(),
            );
            BitFragment::new(writer.as_bytes().to_vec(), writer.bit_len()).unwrap()
        } else {
            let mut writer = BitWriter::new();
            copy_bits(
                &mut writer,
                bytes,
                frame.header_bits.offset,
                frame.header_bits.end().unwrap(),
            );
            BitFragment::new(writer.as_bytes().to_vec(), writer.bit_len()).unwrap()
        };
        result.extend(packet_frame(bytes, frame, header, values));
    }
    result.extend_from_slice(&bytes[main.header_bits.offset as usize / 8..]);
    result
}

fn packet_frame(
    bytes: &[u8],
    frame: &FrameInventory,
    header: BitFragment,
    patch_values: Option<&[u32]>,
) -> Vec<u8> {
    packet_frame_prefix(bytes, frame, header, patch_values.map(dictionary).as_ref())
}
