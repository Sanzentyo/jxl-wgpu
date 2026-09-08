//! Reframe independent libjxl entropy into LF chains with alpha and depth planes.
//! `cargo run -p jxl_wgpu_decode --example regenerate_lf_extra_channels -- OUTPUT_DIRECTORY`
//!
//! libjxl's encoder disables progressive DC with extras. Retain an ordinary VarDCT frame's
//! global extras and HF entropy, remove its LF coefficient substream, and reference an independent
//! small image. Offline Rust decoding finds entropy boundaries; libjxl validates every result.
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use jxl_bitstream::Bitstream;
use jxl_frame::data::{LfGlobal, LfGlobalParams, PassGroupParams, PassGroupParamsVardct};
use jxl_gpu_bitstream::{BitReader, BitWriter, FrameEncoding, FrameInventory, FrameSectionKind};
use jxl_grid::AlignedGrid;
use jxl_oxide_common::Bundle;
use jxl_wgpu_encode::{
    BitFragment, FrameGroupLayout, FramePacketSet, GroupPacket, GroupPacketKind, assemble_frame,
};

#[path = "support/offline/hex.rs"]
mod hex;

fn run(command: &mut Command) -> Vec<u8> {
    let result = command.output().expect("offline libjxl tool");
    assert!(
        result.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    result.stdout
}

fn compile(source: &Path, binary: &Path, libraries: &[&str]) {
    let flags = run(Command::new("pkg-config")
        .args(["--cflags", "--libs"])
        .args(libraries));
    run(Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror"])
        .arg(source)
        .args(std::str::from_utf8(&flags).unwrap().split_whitespace())
        .arg("-o")
        .arg(binary));
}

fn copy_bits(writer: &mut BitWriter, data: &[u8], range: Range<u64>) {
    let mut reader = BitReader::new(data);
    reader.skip_bits(range.start).unwrap();
    let mut remaining = range.end - range.start;
    while remaining != 0 {
        let count = remaining.min(56) as u8;
        writer
            .write_bits(reader.read_bits(count).unwrap(), count)
            .unwrap();
        remaining -= u64::from(count);
    }
}

/// This independent sample reconstruction runs only in the offline generator.
fn boundaries(data: &[u8], inventory: &FrameInventory) -> (Range<u64>, u64) {
    let mut bits = Bitstream::new(data);
    let image = Arc::new(jxl_image::ImageHeader::parse(&mut bits, ()).unwrap());
    let pool = jxl_threadpool::JxlThreadPool::none();
    let mut frame = jxl_frame::Frame::parse(
        &mut bits,
        jxl_frame::FrameContext {
            image_header: Arc::clone(&image),
            tracker: None,
            pool: pool.clone(),
        },
    )
    .unwrap();
    frame.feed_bytes(&data[bits.num_read_bits() / 8..]).unwrap();
    assert!(frame.is_loading_done());
    let mut bits = Bitstream::new(data);
    bits.skip_bits(inventory.sections[0].bits.offset as usize)
        .unwrap();
    let global = LfGlobal::<i32>::parse(
        &mut bits,
        LfGlobalParams::new(&image, frame.header(), None, false),
    )
    .unwrap();
    let start = bits.num_read_bits() as u64;
    let Some(vardct) = &global.vardct else {
        return (start..start, start);
    };
    // Removing quantized LF must not change HF entropy's LF-context selection.
    assert!(vardct.hf_block_ctx.lf_thresholds.iter().all(Vec::is_empty));
    jxl_vardct::LfCoeff::<i32>::parse(
        &mut bits,
        jxl_vardct::LfCoeffParams {
            lf_group_idx: 0,
            lf_width: inventory.width,
            lf_height: inventory.height,
            jpeg_upsampling: inventory.jpeg_upsampling,
            bits_per_sample: 8,
            global_ma_config: global.gmodular.ma_config(),
            allow_partial: false,
            tracker: None,
            pool: &pool,
        },
    )
    .unwrap();
    let end = bits.num_read_bits() as u64;
    let mut gmodular = global.gmodular.try_clone().unwrap();
    let groups = gmodular
        .modular
        .image_mut()
        .unwrap()
        .prepare_groups(frame.pass_shifts())
        .unwrap();
    assert!(groups.lf_groups.iter().all(|group| group.is_empty()));
    assert!(
        groups
            .pass_groups
            .iter()
            .flatten()
            .all(|group| group.is_empty())
    );
    let lf = frame
        .try_parse_lf_group::<i32>(Some(vardct), global.gmodular.ma_config(), None, 0)
        .unwrap()
        .unwrap();
    let hf = frame.try_parse_hf_global(Some(&global)).unwrap().unwrap();
    let mut pass = frame.pass_group_bitstream(0, 0).unwrap().unwrap().bitstream;
    let mut grids = std::array::from_fn::<_, 3, _>(|_| {
        AlignedGrid::<i32>::with_alloc_tracker(
            inventory.width.div_ceil(8) as usize * 8,
            inventory.height.div_ceil(8) as usize * 8,
            None,
        )
        .unwrap()
    });
    let mut output = grids.each_mut().map(AlignedGrid::as_subgrid_mut);
    jxl_frame::data::decode_pass_group(
        &mut pass,
        PassGroupParams {
            frame_header: frame.header(),
            lf_group: &lf,
            pass_idx: 0,
            group_idx: 0,
            global_ma_config: global.gmodular.ma_config(),
            modular: None,
            vardct: Some(PassGroupParamsVardct {
                lf_vardct: vardct,
                hf_global: &hf,
                hf_coeff_output: &mut output,
            }),
            allow_partial: false,
            tracker: None,
            pool: &pool,
        },
    )
    .unwrap();
    let syntax_end = inventory.sections[0].bits.offset + pass.num_read_bits() as u64;
    assert_eq!(
        syntax_end.div_ceil(8) * 8,
        inventory.sections[0].bits.end().unwrap()
    );
    (start..end, syntax_end)
}

fn physical(data: &[u8], lf_level: u32, uses_lf: bool, gaborish: bool) -> Vec<u8> {
    let inventory = jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(inventory.frames.len(), 1);
    let frame = &inventory.frames[0];
    assert_eq!(frame.sections.len(), 1);
    assert_eq!(frame.sections[0].kind, FrameSectionKind::Single);
    assert_eq!(frame.num_passes, 1);
    assert_eq!(frame.flags, 0);
    assert_eq!(inventory.image_header.extra_channels.len(), 2);
    let modular = frame.encoding == FrameEncoding::Modular;
    let lf = lf_level != 0;
    assert!(!uses_lf || !modular);
    let mut header = BitWriter::new();
    header.write_bits(0, 1).unwrap(); // non-default
    header.write_bits(u64::from(lf), 2).unwrap(); // LF or regular
    header.write_bits(u64::from(modular), 1).unwrap();
    if !uses_lf {
        header.write_bits(0, 2).unwrap(); // flags=0
        header.write_bits(0, 6).unwrap(); // color and both extras: 1x upsampling
    } else {
        header.write_bits(2, 2).unwrap();
        header.write_bits(32 - 17, 8).unwrap(); // UseLfFrame, omits upsampling
    }
    if modular {
        header
            .write_bits(u64::from(frame.group_size_shift), 2)
            .unwrap();
    } else {
        header.write_bits(u64::from(frame.x_qm_scale), 3).unwrap();
        header.write_bits(u64::from(frame.b_qm_scale), 3).unwrap();
    }
    header.write_bits(0, 2).unwrap(); // one pass
    if lf {
        header.write_bits(u64::from(lf_level - 1), 2).unwrap();
    } else {
        header.write_bits(0, 1).unwrap(); // no crop
        header.write_bits(0, 6).unwrap(); // replace for color and both extras
        header.write_bits(1, 1).unwrap(); // last frame
    }
    header.write_bits(0, 2).unwrap(); // name
    header.write_bits(0, 1).unwrap(); // custom restoration
    header.write_bits(u64::from(gaborish), 1).unwrap();
    if gaborish {
        header.write_bits(0, 1).unwrap();
    } // default weights
    header.write_bits(0, 2).unwrap(); // EPF off
    header.write_bits(0, 4).unwrap(); // restoration/frame extensions
    let header_bits = header.bit_len();
    let (lf_coefficients, end) = boundaries(data, frame);
    let mut packet = BitWriter::new();
    let start = frame.sections[0].bits.offset;
    if uses_lf {
        copy_bits(&mut packet, data, start..lf_coefficients.start);
        copy_bits(&mut packet, data, lf_coefficients.end..end);
    } else {
        copy_bits(&mut packet, data, start..end);
    }
    packet.align_to_byte().unwrap();
    assemble_frame(
        FramePacketSet::new(
            BitFragment::new(header.into_bytes(), header_bits).unwrap(),
            FrameGroupLayout::new(1, 1, 1).unwrap(),
            [GroupPacket::new(
                GroupPacketKind::Single,
                packet.into_bytes(),
            )],
        )
        .unwrap(),
    )
    .unwrap()
    .into_bytes()
}

fn main() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = PathBuf::from(std::env::args_os().nth(1).expect("output directory"));
    std::fs::create_dir_all(&output).unwrap();
    let temporary = std::env::temp_dir().join(format!("jxl-lf-extras-{}", std::process::id()));
    std::fs::create_dir_all(&temporary).unwrap();
    let generator = temporary.join("generate");
    compile(
        &source.join("generate_lf_extra_channels.c"),
        &generator,
        &["libjxl"],
    );
    run(Command::new(generator).arg(&temporary));
    let decoder = temporary.join("decode");
    compile(
        &source.join("decode_extra_channels.c"),
        &decoder,
        &["libjxl", "libjxl_cms"],
    );
    let seed = std::fs::read(temporary.join("extras.jxl")).unwrap();
    let inventory = jxl_gpu_bitstream::parse(&seed, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let consumer = physical(&seed, 0, true, false);
    let middle = physical(
        &std::fs::read(temporary.join("vardct_root.jxl")).unwrap(),
        1,
        true,
        true,
    );
    for (root, nested) in [
        ("modular", false),
        ("vardct", false),
        ("modular", true),
        ("vardct", true),
    ] {
        let seed_root = std::fs::read(
            temporary.join(format!("{root}_root{}.jxl", if nested { "2" } else { "" })),
        )
        .unwrap();
        for gaborish in [false, true].into_iter().filter(|&gab| !nested || gab) {
            let name = format!(
                "{}{root}_gab{}",
                if nested { "nested_" } else { "" },
                u8::from(gaborish)
            );
            let mut stream = seed[..inventory.frames[0].header_bits.offset as usize / 8].to_vec();
            stream.extend(physical(
                &seed_root,
                if nested { 2 } else { 1 },
                false,
                gaborish,
            ));
            if nested {
                stream.extend_from_slice(&middle);
            }
            stream.extend_from_slice(&consumer);
            let encoded = temporary.join("oracle.jxl");
            std::fs::write(&encoded, &stream).unwrap();
            let reference = run(Command::new(&decoder).arg(&encoded));
            assert_eq!(reference.len(), 65 * 33 * 6 * 4);
            std::fs::write(output.join(format!("{name}.jxl.hex")), hex::hex(&stream)).unwrap();
            assert_eq!(hex::unhex(&hex::hex(&stream)), stream);
            // Binary32 RGBA followed by alpha and depth, all in the reference's sRGB domain.
            let words: Vec<_> = reference
                .chunks_exact(4)
                .map(|v| format!("{:08x}", u32::from_le_bytes(v.try_into().unwrap())))
                .collect();
            let text = words
                .chunks(8)
                .map(|line| line.join(" ") + "\n")
                .collect::<String>();
            std::fs::write(output.join(format!("{name}.f32.hex")), text).unwrap();
            eprintln!(
                "{name}: {} bytes, independent libjxl reference accepted",
                stream.len()
            );
        }
    }
    std::fs::remove_dir_all(temporary).unwrap();
}
