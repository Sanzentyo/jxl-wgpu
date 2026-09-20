use super::{corpus, planes};
use jxl_gpu_bitstream::{
    BitWriter, ChromaticityInventory, ColourEncodingInventory, ExtraChannelTypeInventory,
    PrimariesInventory, SampleBitDepth, TransferFunctionInventory, WhitePointInventory,
};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    GpuDecoder, GpuOutputRequest, ModularChannels, NumericSampleMapping, WgpuDecodeEngine,
    native_modular_pixel_format,
};
use std::num::NonZeroU64;

fn depth(writer: &mut BitWriter, depth: SampleBitDepth) {
    let SampleBitDepth::Integer { bits_per_sample } = depth else {
        unreachable!()
    };
    assert!((1..=31).contains(&bits_per_sample));
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(3, 2).unwrap();
    writer
        .write_bits(u64::from(bits_per_sample - 1), 6)
        .unwrap();
}

fn enumeration(writer: &mut BitWriter, value: u64) {
    assert!(value <= 17);
    writer.write_bits(value.min(2), 2).unwrap();
    if value >= 2 {
        writer.write_bits(value - 2, 4).unwrap();
    }
}

fn chromaticity(writer: &mut BitWriter, xy: ChromaticityInventory) {
    for coordinate in [xy.x, xy.y] {
        let packed = u64::from(((coordinate as u32) << 1) ^ ((coordinate >> 31) as u32));
        let (selector, (base, bits)) = [(0, 19), (524288, 19), (1048576, 20), (2097152, 21)]
            .into_iter()
            .enumerate()
            .find(|(_, (base, bits))| (*base..base + (1u64 << bits)).contains(&packed))
            .unwrap();
        writer.write_bits(selector as u64, 2).unwrap();
        writer.write_bits(packed - base, bits).unwrap();
    }
}

/// Re-encode only the image metadata of the exact-word integer fixtures. The physical frame,
/// transforms and entropy stay byte-identical, and a full inventory equality check proves that
/// no field other than the color declaration and its bit range changed.
fn with_profile(data: &[u8], case: &corpus::Case) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let image = &inventory.image_header;
    assert!(!image.grayscale && !image.xyb_encoded && image.extra_channels.len() == 1);
    let alpha = &image.extra_channels[0];
    assert_eq!(
        alpha.channel_type,
        ExtraChannelTypeInventory::Alpha { associated: false }
    );
    let mut writer = BitWriter::new();
    writer.write_bits(0x0aff, 16).unwrap();
    writer.write_bits(0, 1).unwrap();
    for (size, ratio) in [(image.height, true), (image.width, false)] {
        assert!((1..=512).contains(&size));
        writer.write_bits(0, 2).unwrap();
        writer.write_bits(u64::from(size - 1), 9).unwrap();
        if ratio {
            writer.write_bits(0, 3).unwrap();
        }
    }
    writer.write_bits(0, 2).unwrap(); // Explicit metadata, no extra fields.
    depth(&mut writer, image.bit_depth);
    writer.write_bits(0, 1).unwrap(); // 32-bit Modular buffers.
    writer.write_bits(1, 2).unwrap(); // One extra channel.
    writer.write_bits(0, 3).unwrap(); // Explicit alpha declaration.
    depth(&mut writer, alpha.bit_depth);
    writer.write_bits(0, 5).unwrap(); // No shift, name or association.
    writer.write_bits(0, 1).unwrap(); // Original RGB.
    writer.write_bits(0, 2).unwrap(); // Explicit color, no ICC.
    enumeration(&mut writer, 0); // RGB.
    enumeration(
        &mut writer,
        match case.profile.white {
            WhitePointInventory::D65 => 1,
            WhitePointInventory::Custom(_) => 2,
            WhitePointInventory::E => 10,
            WhitePointInventory::Dci => 11,
        },
    );
    if let WhitePointInventory::Custom(xy) = case.profile.white {
        chromaticity(&mut writer, xy);
    }
    enumeration(
        &mut writer,
        match case.profile.primaries {
            PrimariesInventory::Srgb => 1,
            PrimariesInventory::Custom { .. } => 2,
            PrimariesInventory::Bt2100 => 9,
            PrimariesInventory::P3 => 11,
        },
    );
    if let PrimariesInventory::Custom { red, green, blue } = case.profile.primaries {
        for xy in [red, green, blue] {
            chromaticity(&mut writer, xy);
        }
    }
    if let TransferFunctionInventory::Gamma {
        scaled_gamma,
        inverted,
    } = case.transfer.transfer
    {
        assert!(inverted);
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(u64::from(scaled_gamma), 24).unwrap();
    } else {
        writer.write_bits(0, 1).unwrap(); // Enumerated transfer.
        enumeration(
            &mut writer,
            match case.transfer.transfer {
                TransferFunctionInventory::Linear => 8,
                TransferFunctionInventory::Srgb => 13,
                TransferFunctionInventory::Bt709 => 1,
                TransferFunctionInventory::Dci => 17,
                _ => unreachable!(),
            },
        );
    }
    enumeration(&mut writer, 1); // Relative intent.
    writer.write_bits(0, 2).unwrap(); // No extensions.
    writer.write_bits(1, 1).unwrap(); // Default transform data.
    writer.align_to_byte().unwrap();
    let mut encoded = writer.into_bytes();
    let frame_bytes = &parsed.codestream()[inventory.frames[0].header_bits.offset as usize / 8..];
    encoded.extend_from_slice(frame_bytes);
    let changed = jxl_gpu_bitstream::parse(&encoded, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let mut header = changed.image_header;
    assert!(
        matches!(header.colour_encoding, ColourEncodingInventory::Enumerated {white_point, primaries, transfer_function, ..}
        if white_point == case.profile.white && primaries == case.profile.primaries
            && transfer_function == case.transfer.transfer)
    );
    header.bit_range = image.bit_range;
    header.colour_encoding = image.colour_encoding;
    assert_eq!(&header, image);
    assert_eq!(
        &encoded[changed.frames[0].header_bits.offset as usize / 8..],
        frame_bytes
    );
    encoded
}

#[test]
fn sdr_metadata_preserves_native_modular_integer_samples() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoders = [
        GpuDecoder::wgpu(backend.clone()).unwrap(),
        GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
        ),
    ];
    for name in [
        "17-3-31-33x5-p0-r0",
        "31-3-5-33x5-p0-r0",
        "31-3-24-33x5-p0-r0",
    ] {
        let root = corpus::directory().parent().unwrap().join("integer");
        let data = jxl_test_support::offline::unhex(
            &std::fs::read_to_string(root.join(format!("{name}.jxl.hex"))).unwrap(),
        );
        let expected: Vec<_> = std::fs::read_to_string(root.join(format!("{name}.u32.hex")))
            .unwrap()
            .split_whitespace()
            .map(|v| u32::from_str_radix(v, 16).unwrap())
            .collect();
        for (case, intent) in corpus::cases()
            .into_iter()
            .chain(corpus::analytic_cases())
            .filter(|c| {
                c.mode == corpus::Mode::ModularRgb
                    && !c.profile.grayscale
                    && !c.sequence
                    && !c.floating
            })
            .flat_map(|case| corpus::intents::ALL.map(|intent| (case.clone(), intent)))
        {
            let data = with_profile(&data, &case);
            let data = corpus::intents::replace(&data, intent);
            let image = jxl_gpu_bitstream::parse(&data, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap()
                .image_header;
            let SampleBitDepth::Integer {
                bits_per_sample: color_bits,
            } = image.bit_depth
            else {
                unreachable!()
            };
            let SampleBitDepth::Integer {
                bits_per_sample: alpha_bits,
            } = image.extra_channels[0].bit_depth
            else {
                unreachable!()
            };
            for selection in 0..5 {
                let bits = if selection == 3 {
                    alpha_bits
                } else {
                    color_bits
                } as u8;
                let request = if selection == 4 {
                    GpuOutputRequest::color(
                        native_modular_pixel_format(ModularChannels::Rgb, bits).unwrap(),
                    )
                    .unwrap()
                } else {
                    let request = GpuOutputRequest::numeric(
                        native_modular_pixel_format(ModularChannels::Gray, bits).unwrap(),
                        NumericSampleMapping::NativeUnsigned,
                    )
                    .unwrap();
                    if selection == 3 {
                        request.with_extra_channel(0).unwrap()
                    } else {
                        request.with_color_channel(selection).unwrap()
                    }
                };
                let expected: Vec<_> = expected
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| {
                        if selection == 4 {
                            p[..3].to_vec()
                        } else {
                            vec![p[selection as usize]]
                        }
                    })
                    .collect();
                for (bounded, decoder) in decoders.iter().enumerate() {
                    let mut session = if bounded == 0 {
                        decoder.open(&data, request.clone()).unwrap()
                    } else {
                        planes::open_fragmented(decoder, &data, request.clone())
                    };
                    let frame = pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap();
                    let bytes = planes::read_bytes(&backend, &frame.output().outputs[0]);
                    let actual: Vec<_> = bytes
                        .chunks_exact(usize::from(bits.next_power_of_two().max(8) / 8))
                        .map(|sample| {
                            sample
                                .iter()
                                .enumerate()
                                .fold(0u32, |v, (i, &b)| v | (u32::from(b) << (8 * i)))
                        })
                        .collect();
                    assert_eq!(
                        actual, expected,
                        "{name}/{}/{intent:?}/{selection}/bounded{bounded}",
                        case.name
                    );
                    drop((frame, session));
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
}
