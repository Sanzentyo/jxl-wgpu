//! Build preview conformance streams from independent frame entropy without pixel decoding.
use jxl_bitstream::{Bitstream, U};
use jxl_gpu_bitstream::{BitReader, BitWriter, CodestreamInventory, FrameEncoding};
use jxl_image::{
    AnimationHeader, BitDepth, ExtraChannelInfo, SizeHeader,
    color::{ColourEncoding, ToneMapping},
};
use jxl_oxide_common::Bundle;
use std::ops::Range;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub div8: bool,
    pub ratio: u32,
    pub is_last: bool,
    pub duration: u32,
    pub timecode: u32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            div8: false,
            ratio: 0,
            is_last: true,
            duration: 0,
            timecode: 0,
        }
    }
}

pub fn inventory(data: &[u8]) -> CodestreamInventory {
    jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
}

fn copy(writer: &mut BitWriter, data: &[u8], range: Range<u64>) {
    let mut reader = BitReader::new(data);
    reader.skip_bits(range.start).unwrap();
    while reader.bit_offset() < range.end {
        let count = (range.end - reader.bit_offset()).min(56) as u8;
        writer
            .write_bits(reader.read_bits(count).unwrap(), count)
            .unwrap();
    }
}

fn offset(reader: &Bitstream<'_>) -> u64 {
    reader.num_read_bits() as u64
}

fn dimension(writer: &mut BitWriter, size: u32, div8: bool) {
    let value = if div8 {
        assert!(size.is_multiple_of(8));
        size / 8
    } else {
        size
    };
    let (selector, base, count) = if div8 {
        match value {
            16 => (0, 16, 0),
            32 => (1, 32, 0),
            1..=32 => (2, 1, 5),
            33..=544 => (3, 33, 9),
            _ => panic!("preview size"),
        }
    } else {
        match value {
            1..=64 => (0, 1, 6),
            65..=320 => (1, 65, 8),
            321..=1344 => (2, 321, 10),
            1345..=5440 => (3, 1345, 12),
            _ => panic!("preview size"),
        }
    };
    writer.write_bits(selector, 2).unwrap();
    writer.write_bits(u64::from(value - base), count).unwrap();
}

fn image_header(
    data: &[u8],
    info: &CodestreamInventory,
    width: u32,
    height: u32,
    options: Options,
) -> Vec<u8> {
    assert!(info.image_header.preview_size.is_none() && info.image_header.embedded_icc.is_none());
    let mut reader = Bitstream::new(data);
    assert_eq!(reader.read_bits(16).unwrap(), 0x0aff);
    SizeHeader::parse(&mut reader, ()).unwrap();
    let metadata_start = offset(&reader);
    let all_default = reader.read_bool().unwrap();
    let extra_fields = !all_default && reader.read_bool().unwrap();
    let mut intrinsic = None;
    let mut animation = None;
    if extra_fields {
        reader.read_bits(3).unwrap();
        let start = offset(&reader);
        if reader.read_bool().unwrap() {
            SizeHeader::parse(&mut reader, ()).unwrap();
        }
        intrinsic = Some(start..offset(&reader));
        assert!(!reader.read_bool().unwrap());
        let start = offset(&reader);
        if reader.read_bool().unwrap() {
            AnimationHeader::parse(&mut reader, ()).unwrap();
        }
        animation = Some(start..offset(&reader));
    }
    let body_start = offset(&reader);
    if !all_default {
        BitDepth::parse(&mut reader, ()).unwrap();
        reader.read_bool().unwrap();
        let count = reader.read_u32(0, 1, 2 + U(4), 1 + U(12)).unwrap();
        for _ in 0..count {
            ExtraChannelInfo::parse(&mut reader, ()).unwrap();
        }
        reader.read_bool().unwrap();
        ColourEncoding::parse(&mut reader, ()).unwrap();
    }
    let body_end = offset(&reader);
    if extra_fields {
        ToneMapping::parse(&mut reader, ()).unwrap();
    }
    let tail_start = offset(&reader);
    let mut writer = BitWriter::new();
    copy(&mut writer, data, 0..metadata_start);
    writer.write_bits(0, 1).unwrap(); // explicit metadata
    writer.write_bits(1, 1).unwrap(); // extra fields
    writer
        .write_bits(u64::from(info.image_header.orientation - 1), 3)
        .unwrap();
    if let Some(range) = intrinsic {
        copy(&mut writer, data, range);
    } else {
        writer.write_bits(0, 1).unwrap();
    }
    writer.write_bits(1, 1).unwrap(); // preview
    writer.write_bits(u64::from(options.div8), 1).unwrap();
    dimension(&mut writer, height, options.div8);
    writer.write_bits(u64::from(options.ratio), 3).unwrap();
    if options.ratio == 0 {
        dimension(&mut writer, width, options.div8);
    }
    if let Some(range) = animation {
        copy(&mut writer, data, range);
    } else {
        writer.write_bits(0, 1).unwrap();
    }
    if all_default {
        writer.write_bits(0, 3).unwrap(); // 8-bit integer
        writer.write_bits(1, 1).unwrap(); // 16-bit Modular buffers
        writer.write_bits(0, 2).unwrap(); // no extras
        writer.write_bits(3, 2).unwrap(); // XYB and default color
    } else {
        copy(&mut writer, data, body_start..body_end);
    }
    if extra_fields {
        copy(&mut writer, data, body_end..tail_start);
    } else {
        writer.write_bits(1, 1).unwrap();
    }
    if all_default {
        writer.write_bits(0, 2).unwrap();
    }
    copy(
        &mut writer,
        data,
        tail_start..info.image_header.bit_range.end().unwrap(),
    );
    writer.align_to_byte().unwrap();
    writer.into_bytes()
}

fn toc_size(writer: &mut BitWriter, size: u64) {
    for (selector, (base, count)) in [(0, 10), (1024, 14), (17408, 22), (4211712, 30)]
        .into_iter()
        .enumerate()
    {
        if (base..base + (1 << count)).contains(&size) {
            writer.write_bits(selector as u64, 2).unwrap();
            writer.write_bits(size - base, count).unwrap();
            return;
        }
    }
    panic!("TOC size");
}

fn preview_frame(
    data: &[u8],
    source: &CodestreamInventory,
    main: &CodestreamInventory,
    options: Options,
) -> Vec<u8> {
    assert_eq!(source.frames.len(), 1);
    assert!(source.image_header.animation.is_none());
    let frame = &source.frames[0];
    assert!(!frame.have_crop && !frame.uses_lf_frame() && !frame.toc_permuted && frame.is_last);
    assert_eq!(frame.num_passes, 1);
    let start = frame.header_bits.offset;
    let mut reader = Bitstream::new(data);
    reader.skip_bits(start as usize).unwrap();
    assert!(!reader.read_bool().unwrap());
    assert_eq!(reader.read_bits(2).unwrap(), 0);
    reader.read_bool().unwrap();
    reader.read_u64().unwrap();
    if !source.image_header.xyb_encoded && reader.read_bool().unwrap() {
        reader.skip_bits(6).unwrap();
    }
    reader
        .skip_bits(2 * (1 + source.image_header.extra_channels.len()))
        .unwrap();
    if frame.encoding == FrameEncoding::Modular {
        reader.skip_bits(2).unwrap();
    } else if source.image_header.xyb_encoded {
        reader.skip_bits(6).unwrap();
    }
    assert_eq!(reader.read_bits(2).unwrap(), 0); // one pass
    assert!(!reader.read_bool().unwrap()); // no crop
    for _ in 0..=source.image_header.extra_channels.len() {
        assert_eq!(reader.read_bits(2).unwrap(), 0);
    }
    let timing_start = offset(&reader);
    assert!(reader.read_bool().unwrap());
    let suffix_start = offset(&reader);
    let mut writer = BitWriter::new();
    copy(&mut writer, data, start..timing_start);
    if let Some(animation) = main.image_header.animation {
        writer.write_bits(3, 2).unwrap();
        writer.write_bits(u64::from(options.duration), 32).unwrap();
        if animation.have_timecodes {
            writer.write_bits(u64::from(options.timecode), 32).unwrap();
        }
    } else {
        assert_eq!(options.duration, 0);
    }
    writer.write_bits(u64::from(options.is_last), 1).unwrap();
    if !options.is_last {
        writer.write_bits(0, 2).unwrap(); // no retained reference
        if options.duration == 0 {
            writer.write_bits(0, 1).unwrap();
        }
    }
    copy(
        &mut writer,
        data,
        suffix_start..frame.header_bits.end().unwrap(),
    );
    writer.write_bits(0, 1).unwrap();
    writer.align_to_byte().unwrap();
    for section in &frame.sections {
        toc_size(&mut writer, section.bytes.length);
    }
    writer.align_to_byte().unwrap();
    let mut output = writer.into_bytes();
    for section in &frame.sections {
        output.extend_from_slice(
            &data[section.bytes.offset as usize..section.bytes.end().unwrap() as usize],
        );
    }
    output
}

pub fn combine(main: &[u8], preview: &[u8], options: Options) -> Vec<u8> {
    let main_file = jxl_gpu_bitstream::parse(main, Default::default()).unwrap();
    let preview_file = jxl_gpu_bitstream::parse(preview, Default::default()).unwrap();
    let main = main_file.codestream();
    let preview = preview_file.codestream();
    let main_info = inventory(main);
    let preview_info = inventory(preview);
    assert_eq!(
        main_info.image_header.bit_depth,
        preview_info.image_header.bit_depth
    );
    assert_eq!(
        main_info.image_header.xyb_encoded,
        preview_info.image_header.xyb_encoded
    );
    assert_eq!(
        main_info.image_header.colour_encoding,
        preview_info.image_header.colour_encoding
    );
    assert_eq!(
        main_info.image_header.extra_channels,
        preview_info.image_header.extra_channels
    );
    assert_eq!(
        main_info.image_header.upsampling_weights,
        preview_info.image_header.upsampling_weights
    );
    assert_eq!(
        main_info.image_header.opsin_inverse_matrix,
        preview_info.image_header.opsin_inverse_matrix
    );
    let (width, height) = (
        preview_info.image_header.width,
        preview_info.image_header.height,
    );
    let mut output = image_header(main, &main_info, width, height, options);
    output.extend(preview_frame(preview, &preview_info, &main_info, options));
    output.extend_from_slice(&main[main_info.frames[0].header_bits.offset as usize / 8..]);
    let checked = inventory(&output);
    assert_eq!(checked.image_header.preview_size, Some((width, height)));
    assert_eq!(checked.frames.len(), main_info.frames.len() + 1);
    assert_eq!(checked.frames[0].noise_seed, [0, 1]);
    assert_eq!(checked.frames[0].is_last, options.is_last);
    for (frame, (data, original)) in checked.frames.iter().zip(
        std::iter::once((preview, &preview_info.frames[0]))
            .chain(main_info.frames.iter().map(|f| (main, f))),
    ) {
        assert_eq!(frame.sections.len(), original.sections.len());
        for (new, old) in frame.sections.iter().zip(&original.sections) {
            assert_eq!(
                &output[new.bytes.offset as usize..new.bytes.end().unwrap() as usize],
                &data[old.bytes.offset as usize..old.bytes.end().unwrap() as usize]
            );
        }
    }
    for (frame, original) in checked.frames[1..].iter().zip(&main_info.frames) {
        let mut expected_seed = original.noise_seed;
        // The preview advances the nonvisible counter. It survives into leading LF/hidden
        // main frames and is reset only when the first visible main frame advances.
        if expected_seed[0] == 0 {
            expected_seed[1] += 1;
        }
        assert_eq!(frame.noise_seed, expected_seed);
        assert_eq!(
            frame.lf_source_frame,
            original.lf_source_frame.map(|id| id + 1)
        );
    }
    output
}

pub fn orient(data: &[u8], orientation: u32) -> Vec<u8> {
    assert!((1..=8).contains(&orientation));
    let mut reader = Bitstream::new(data);
    reader.skip_bits(16).unwrap();
    SizeHeader::parse(&mut reader, ()).unwrap();
    assert!(!reader.read_bool().unwrap() && reader.read_bool().unwrap());
    let start = reader.num_read_bits();
    let mut output = data.to_vec();
    for bit in 0..3 {
        let at = start + bit;
        output[at / 8] = (output[at / 8] & !(1 << (at % 8)))
            | ((((orientation - 1) >> bit) as u8 & 1) << (at % 8));
    }
    output
}

pub fn zero_noise(data: &[u8]) -> Vec<u8> {
    let info = inventory(data);
    let mut output = data.to_vec();
    for frame in info.frames.iter().filter(|frame| frame.flags & 1 != 0) {
        assert_eq!(
            frame.flags & (2 | 16),
            0,
            "no preceding patch/spline syntax"
        );
        let start = frame.sections[0].bits.offset as usize;
        for bit in start..start + 80 {
            output[bit / 8] &= !(1 << (bit % 8));
        }
    }
    output
}
