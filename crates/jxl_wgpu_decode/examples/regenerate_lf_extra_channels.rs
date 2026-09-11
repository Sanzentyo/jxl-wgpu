//! Reframe independent libjxl entropy into LF chains with alpha and depth planes.
//! `cargo run -p jxl_wgpu_decode --example regenerate_lf_extra_channels -- OUTPUT_DIRECTORY`
//! Add `--conformance` for mixed-precision LF1–LF4, alpha, orientation and resampling cases.
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
#[path = "support/lf_conformance.rs"]
mod lf_conformance;
#[path = "support/lf_extra.rs"]
mod lf_extra;

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

fn write_lf_oracle(
    output: &Path,
    temporary: &Path,
    decoder: &Path,
    name: &str,
    level: u32,
    encoded: &[u8],
) {
    let input = temporary.join("lf-oracle.jxl");
    std::fs::write(&input, encoded).unwrap();
    let reference = run(Command::new(decoder).arg(&input).args([
        "--linear",
        "--preserve-alpha",
        "--keep-orientation",
    ]));
    let image = jxl_gpu_bitstream::parse(encoded, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .image_header;
    assert_eq!(
        reference.len(),
        image.width as usize * image.height as usize * 6 * 4
    );
    std::fs::write(
        output.join(format!("{name}.lf{level}.jxl.hex")),
        hex::hex(encoded),
    )
    .unwrap();
    let words = reference
        .chunks_exact(4)
        .map(|word| format!("{:08x}", u32::from_le_bytes(word.try_into().unwrap())))
        .collect::<Vec<_>>();
    std::fs::write(
        output.join(format!("{name}.lf{level}.linear.f32.hex")),
        words
            .chunks(8)
            .map(|line| line.join(" ") + "\n")
            .collect::<String>(),
    )
    .unwrap();
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
            bits_per_sample: frame.header().bit_depth.bits_per_sample(),
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Final,
    Background,
    Blend,
}

fn physical(data: &[u8], lf_level: u32, uses_lf: bool, gaborish: bool) -> Vec<u8> {
    physical_with_role(data, lf_level, uses_lf, gaborish, Role::Final)
}

fn physical_with_role(
    data: &[u8],
    lf_level: u32,
    uses_lf: bool,
    gaborish: bool,
    role: Role,
) -> Vec<u8> {
    let inventory = jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(inventory.frames.len(), 1);
    let frame = &inventory.frames[0];
    assert_eq!(frame.num_passes, 1);
    assert_eq!(frame.flags, 0);
    assert_eq!(inventory.image_header.extra_channels.len(), 2);
    assert!(inventory.image_header.xyb_encoded);
    assert!(
        inventory
            .image_header
            .extra_channels
            .iter()
            .all(|extra| extra.dimension_shift == 0),
        "this offline header writer does not reframe shifted extra metadata"
    );
    let modular = frame.encoding == FrameEncoding::Modular;
    let lf = lf_level != 0;
    assert!(!uses_lf || !modular);
    let mut header = BitWriter::new();
    header.write_bits(0, 1).unwrap(); // non-default
    header.write_bits(u64::from(lf), 2).unwrap(); // LF or regular
    header.write_bits(u64::from(modular), 1).unwrap();
    if !uses_lf {
        header.write_bits(0, 2).unwrap(); // flags=0
        for factor in
            std::iter::once(frame.upsampling).chain(frame.extra_channel_upsampling.iter().copied())
        {
            assert!([1, 2, 4, 8].contains(&factor));
            header.write_bits(u64::from(factor.ilog2()), 2).unwrap();
        }
    } else {
        assert_eq!(frame.upsampling, 1);
        assert!(
            frame
                .extra_channel_upsampling
                .iter()
                .all(|&factor| factor == 1)
        );
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
        if role == Role::Blend {
            for _ in 0..2 {
                header.write_bits(2, 2).unwrap(); // Blend color and alpha
                header.write_bits(0, 2).unwrap(); // alpha index zero
                header.write_bits(1, 1).unwrap(); // clamp alpha
                header.write_bits(1, 2).unwrap(); // reference slot one
            }
            header.write_bits(1, 2).unwrap(); // Add depth
            header.write_bits(1, 2).unwrap(); // reference slot one
        } else {
            header.write_bits(0, 6).unwrap(); // replace color and both extras
        }
        header
            .write_bits(u64::from(role != Role::Background), 1)
            .unwrap();
        if role == Role::Background {
            header.write_bits(1, 2).unwrap(); // save reference one
            header.write_bits(0, 1).unwrap(); // save after color transform
        }
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
    assemble_frame(
        FramePacketSet::new(
            BitFragment::new(header.into_bytes(), header_bits).unwrap(),
            FrameGroupLayout::new(
                frame.low_frequency_group_count.try_into().unwrap(),
                frame.group_count.try_into().unwrap(),
                1,
            )
            .unwrap(),
            lf_extra::packets(data, frame, uses_lf),
        )
        .unwrap(),
    )
    .unwrap()
    .into_bytes()
}

fn main() {
    let option = std::env::args().nth(2);
    assert!(matches!(
        option.as_deref(),
        None | Some("--distributed" | "--conformance")
    ));
    let distributed = option.as_deref() == Some("--distributed");
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
    let mut generate = Command::new(generator);
    generate.arg(&temporary);
    if let Some(option) = &option {
        generate.arg(option);
    }
    run(&mut generate);
    let decoder = temporary.join("decode");
    compile(
        &source.join("decode_extra_channels.c"),
        &decoder,
        &["libjxl", "libjxl_cms"],
    );
    if option.as_deref() == Some("--conformance") {
        lf_conformance::generate(&temporary, &output, &decoder);
        std::fs::remove_dir_all(temporary).unwrap();
        return;
    }
    let seed = std::fs::read(temporary.join("extras.jxl")).unwrap();
    let inventory = jxl_gpu_bitstream::parse(&seed, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let consumer = physical(&seed, 0, true, false);
    let middle = (!distributed).then(|| {
        physical(
            &std::fs::read(temporary.join("vardct_root.jxl")).unwrap(),
            1,
            true,
            true,
        )
    });
    for (root, nested) in [
        ("modular", false),
        ("vardct", false),
        ("modular", true),
        ("vardct", true),
    ]
    .into_iter()
    .filter(|&(_, nested)| !distributed || !nested)
    {
        let seed_root = std::fs::read(
            temporary.join(format!("{root}_root{}.jxl", if nested { "2" } else { "" })),
        )
        .unwrap();
        for gaborish in [false, true]
            .into_iter()
            .filter(|&gab| !(nested || distributed) || gab)
        {
            let name = format!(
                "{}{root}_gab{}",
                if distributed {
                    "distributed_"
                } else if nested {
                    "nested_"
                } else {
                    ""
                },
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
                stream.extend_from_slice(middle.as_ref().unwrap());
            }
            if nested {
                // Both LF updates precede the hidden background. LF2's prediction slot expires
                // before either presentation can run, exercising independent extra ownership.
                let mut composed = stream.clone();
                composed.extend(physical_with_role(&seed, 0, false, true, Role::Background));
                composed.extend(physical_with_role(&seed, 0, true, true, Role::Blend));
                let path = temporary.join("composed.jxl");
                std::fs::write(&path, &composed).unwrap();
                let reference = run(Command::new(&decoder).arg(&path).arg("--preserve-alpha"));
                assert_eq!(
                    reference.len(),
                    inventory.image_header.width as usize
                        * inventory.image_header.height as usize
                        * 6
                        * 4
                );
                std::fs::write(
                    output.join(format!("{name}.composed.jxl.hex")),
                    hex::hex(&composed),
                )
                .unwrap();
                // Match the composed consumer's restoration; the older final-only fixture
                // intentionally disables consumer Gaborish and is not this layer's oracle.
                let mut foreground = stream.clone();
                foreground.extend(physical(&seed, 0, true, true));
                std::fs::write(
                    output.join(format!("{name}.foreground.jxl.hex")),
                    hex::hex(&foreground),
                )
                .unwrap();
                // Reconstruct the exact hidden background as an independent ordinary image.
                let mut background =
                    seed[..inventory.frames[0].header_bits.offset as usize / 8].to_vec();
                background.extend(physical(&seed, 0, false, true));
                for layer in [&foreground, &background] {
                    std::fs::write(&path, layer).unwrap();
                    assert_eq!(
                        run(Command::new(&decoder).arg(&path).arg("--preserve-alpha")).len(),
                        reference.len()
                    );
                }
                std::fs::write(
                    output.join(format!("{name}.background.jxl.hex")),
                    hex::hex(&background),
                )
                .unwrap();
            }
            stream.extend_from_slice(&consumer);
            // Independently decodable LF producers retain the same entropy and restoration,
            // but use a smaller ordinary image canvas. A nested LF1 oracle retains its LF2
            // dependency as LF1 relative to that canvas.
            let seed_prefix = |data: &[u8]| {
                let inventory = jxl_gpu_bitstream::parse(data, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                data[..inventory.frames[0].header_bits.offset as usize / 8].to_vec()
            };
            let mut root_image = seed_prefix(&seed_root);
            root_image.extend(physical(&seed_root, 0, false, gaborish));
            write_lf_oracle(
                &output,
                &temporary,
                &decoder,
                &name,
                if nested { 2 } else { 1 },
                &root_image,
            );
            if nested {
                let middle_seed = std::fs::read(temporary.join("vardct_root.jxl")).unwrap();
                let mut middle_image = seed_prefix(&middle_seed);
                middle_image.extend(physical(&seed_root, 1, false, gaborish));
                middle_image.extend(physical(&middle_seed, 0, true, true));
                write_lf_oracle(&output, &temporary, &decoder, &name, 1, &middle_image);
            }
            let encoded = temporary.join("oracle.jxl");
            std::fs::write(&encoded, &stream).unwrap();
            let reference = run(Command::new(&decoder).arg(&encoded));
            assert_eq!(
                reference.len(),
                inventory.image_header.width as usize
                    * inventory.image_header.height as usize
                    * 6
                    * 4
            );
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
