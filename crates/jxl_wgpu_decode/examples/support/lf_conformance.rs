//! Native seeds with independent precision, orientation, alpha association and LF geometry.
use super::*;

fn prefix(seed: &[u8]) -> Vec<u8> {
    let inventory = jxl_gpu_bitstream::parse(seed, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    seed[..inventory.frames[0].header_bits.offset as usize / 8].to_vec()
}

fn chain(seeds: &[Vec<u8>], lowest: usize) -> Vec<u8> {
    let mut output = prefix(&seeds[lowest]);
    for index in (lowest..seeds.len()).rev() {
        output.extend(physical(
            &seeds[index],
            (index - lowest) as u32,
            index + 1 != seeds.len(),
            true,
        ));
    }
    output
}

fn validate(decoder: &Path, temporary: &Path, encoded: &[u8]) {
    let inventory = jxl_gpu_bitstream::parse(encoded, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let path = temporary.join("validate.jxl");
    std::fs::write(&path, encoded).unwrap();
    let reference = run(Command::new(decoder)
        .arg(&path)
        .args(["--preserve-alpha", "--keep-orientation"]));
    assert_eq!(
        reference.len(),
        inventory.image_header.width as usize * inventory.image_header.height as usize * 6 * 4
    );
    assert!(
        reference
            .chunks_exact(4)
            .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite())
    );
}

fn crops(
    temporary: &Path,
    output: &Path,
    decoder: &Path,
    name: &str,
    seeds: &[Vec<u8>],
    name_out: &str,
) {
    let path = temporary.join(format!("{name}_crop.jxl"));
    if !path.exists() {
        return;
    }
    let data = std::fs::read(path).unwrap();
    let data = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream()
        .to_vec();
    for (suffix, x0, y0, blend) in [
        ("crop_foreground", 0, 0, false),
        ("crop_left", -3, 5, true),
        ("crop_top", 5, -3, true),
    ] {
        let mut encoded = prefix(&seeds[0]);
        for level in (1..seeds.len()).rev() {
            encoded.extend(physical(
                &seeds[level],
                level as u32,
                level + 1 != seeds.len(),
                true,
            ));
        }
        if blend {
            encoded.extend(physical_with_role(
                &seeds[0],
                0,
                false,
                true,
                Role::Background,
            ));
        }
        encoded.extend(physical_with_role(
            &data,
            0,
            true,
            true,
            Role::Crop { x0, y0, blend },
        ));
        validate(decoder, temporary, &encoded);
        std::fs::write(
            output.join(format!("{name_out}.{suffix}.jxl.hex")),
            hex::hex(&encoded),
        )
        .unwrap();
    }
}

pub(super) fn generate(temporary: &Path, output: &Path, decoder: &Path) {
    let manifest = std::fs::read_to_string(temporary.join("conformance.txt")).unwrap();
    for row in manifest.lines() {
        let (name, levels) = row.split_once(' ').unwrap();
        let levels: usize = levels.parse().unwrap();
        assert!((1..=4).contains(&levels));
        for root in ["modular", "vardct"] {
            let name_out = format!("{name}_{root}");
            let seeds: Vec<_> = (0..=levels)
                .map(|level| {
                    let mode = if level == levels { root } else { "vardct" };
                    let data =
                        std::fs::read(temporary.join(format!("{name}_{mode}_lf{level}.jxl")))
                            .unwrap();
                    // Level-10 native seeds carry a container. Frame offsets and the offline
                    // entropy reader both address its extracted codestream.
                    jxl_gpu_bitstream::parse(&data, Default::default())
                        .unwrap()
                        .codestream()
                        .to_vec()
                })
                .collect();
            let complete = chain(&seeds, 0);
            validate(decoder, temporary, &complete);
            std::fs::write(
                output.join(format!("{name_out}.jxl.hex")),
                hex::hex(&complete),
            )
            .unwrap();
            for level in 1..=levels {
                write_lf_oracle(
                    output,
                    temporary,
                    decoder,
                    &name_out,
                    level as u32,
                    &chain(&seeds, level),
                );
            }
            let mut background = prefix(&seeds[0]);
            background.extend(physical(&seeds[0], 0, false, true));
            let mut composed = prefix(&seeds[0]);
            for level in (1..=levels).rev() {
                composed.extend(physical(&seeds[level], level as u32, level != levels, true));
            }
            composed.extend(physical_with_role(
                &seeds[0],
                0,
                false,
                true,
                Role::Background,
            ));
            composed.extend(physical_with_role(&seeds[0], 0, true, true, Role::Blend));
            for (suffix, data) in [("background", background), ("composed", composed)] {
                validate(decoder, temporary, &data);
                std::fs::write(
                    output.join(format!("{name_out}.{suffix}.jxl.hex")),
                    hex::hex(&data),
                )
                .unwrap();
            }
            crops(temporary, output, decoder, name, &seeds, &name_out);
            eprintln!(
                "{name_out}: LF{levels} chain, independent producers and coalesced composition accepted"
            );
        }
    }
    std::fs::write(output.join("cases.txt"), manifest).unwrap();
}
