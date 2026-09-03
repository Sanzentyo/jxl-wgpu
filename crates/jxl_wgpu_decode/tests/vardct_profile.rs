use jxl_gpu_bitstream::{BitRange, BitWriter, FrameSectionKind, InventoryLimits, parse};
use jxl_wgpu_decode::vardct::frontend::{
    HfGlobalPrefix, HfMetadataPrefix, LfGlobalPrefix, LfGroupPrefix, ModularChannelPlan,
    StandardVarDctProfile, UnsupportedVarDctFeature, VarDctFrontendError, VarDctPacketError,
    VarDctSectionLayout,
};
use jxl_wgpu_decode::vardct::packet::BoundedVarDctPacketPlan;

fn decode_hex(source: &str) -> Vec<u8> {
    let digits: Vec<_> = source
        .bytes()
        .filter(|byte| byte.is_ascii_hexdigit())
        .collect();
    digits
        .chunks_exact(2)
        .map(|pair| {
            let high = (pair[0] as char).to_digit(16).unwrap();
            let low = (pair[1] as char).to_digit(16).unwrap();
            ((high << 4) | low) as u8
        })
        .collect()
}

fn negotiate(bytes: &[u8]) -> StandardVarDctProfile {
    StandardVarDctProfile::negotiate(&inventory(bytes)).unwrap()
}

fn inventory(bytes: &[u8]) -> jxl_gpu_bitstream::CodestreamInventory {
    parse(bytes, Default::default())
        .unwrap()
        .codestream_inventory(InventoryLimits::default())
        .unwrap()
}

fn custom_lf_global(
    dequant_bits: [u16; 3],
    colour_factor: (u8, u16),
    base_bits: [u16; 2],
    lf_factors: [u8; 2],
) -> (Vec<u8>, BitRange) {
    let mut writer = BitWriter::new();
    writer.write_bits(0, 1).unwrap();
    for bits in dequant_bits {
        writer.write_bits(u64::from(bits), 16).unwrap();
    }
    writer.write_bits(0, 2).unwrap();
    writer.write_bits(0, 11).unwrap();
    writer.write_bits(0, 2).unwrap();
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(u64::from(colour_factor.0), 2).unwrap();
    let factor_bits = match colour_factor.0 {
        0 | 1 => 0,
        2 => 8,
        3 => 16,
        _ => unreachable!(),
    };
    writer
        .write_bits(u64::from(colour_factor.1), factor_bits)
        .unwrap();
    for bits in base_bits {
        writer.write_bits(u64::from(bits), 16).unwrap();
    }
    for factor in lf_factors {
        writer.write_bits(u64::from(factor), 8).unwrap();
    }
    writer.write_bits(0, 1).unwrap();
    let packet = BitRange {
        offset: 0,
        length: writer.bit_len() as u64,
    };
    (writer.into_bytes(), packet)
}

#[test]
fn accepts_basic_single_entry_without_host_entropy_decode() {
    let bytes = decode_hex(include_str!("../test-data/basic.jxl.hex"));
    let profile = negotiate(&bytes);
    assert_eq!((profile.width(), profile.height()), (1, 1));
    assert_eq!(profile.group_count(), 1);
    assert!(profile.adaptive_lf_smoothing());
    assert_eq!(profile.lf_quant_stream_index(0).unwrap(), 1);
    assert_eq!(profile.hf_metadata_stream_index(0).unwrap(), 3);
    let VarDctSectionLayout::Single { packet } = *profile.sections() else {
        panic!("basic uses a single-entry TOC")
    };
    let prefix = LfGlobalPrefix::parse(&bytes, packet).unwrap();
    assert_eq!(prefix.lf_dequantization, Default::default());
    assert_eq!(prefix.lf_correlation, Default::default());
    assert_eq!((prefix.global_scale, prefix.quant_lf), (4587, 16));
    assert_eq!(prefix.hf_block_context.num_block_clusters, 15);
    assert_eq!(prefix.hf_block_context.block_context_map.len(), 39);
    assert!(prefix.global_ma_tree_bit_offset.unwrap() > packet.offset);
}

#[test]
fn parses_non_default_lf_dequantization_and_channel_correlation() {
    for (selector, extra, expected_colour_factor) in
        [(0, 0, 84), (1, 0, 256), (2, 40, 42), (3, 42, 300)]
    {
        let (bytes, packet) = custom_lf_global(
            [0x2c00, 0x3000, 0x3400],
            (selector, extra),
            [0xb800, 0x3800],
            [112, 160],
        );
        let prefix = LfGlobalPrefix::parse(&bytes, packet).unwrap();
        assert_eq!(prefix.lf_dequantization.multipliers, [0.0625, 0.125, 0.25]);
        assert_eq!(prefix.lf_correlation.colour_factor, expected_colour_factor);
        assert_eq!(prefix.lf_correlation.base, [-0.5, 0.5]);
        assert_eq!(prefix.lf_correlation.lf_factors, [-16, 32]);
        let inverse = 1.0 / expected_colour_factor as f32;
        assert_eq!(
            prefix.lf_correlation.lf_slopes(),
            [-0.5 - 16.0 * inverse, 0.5 + 32.0 * inverse]
        );
        assert_eq!(prefix.lf_correlation.hf_params(), [-0.5, 0.5, inverse]);
    }
}

#[test]
fn rejects_invalid_lf_dequantization_and_base_correlation() {
    let (bytes, packet) = custom_lf_global([0, 0x3000, 0x3400], (0, 0), [0, 0x3c00], [128, 128]);
    assert!(matches!(
        LfGlobalPrefix::parse(&bytes, packet),
        Err(VarDctPacketError::LfDequantizationTooSmall {
            channel: "X",
            value: 0.0,
        })
    ));

    let (bytes, packet) = custom_lf_global(
        [0x2c00, 0x3000, 0x3400],
        (0, 0),
        [0xc480, 0x3c00],
        [128, 128],
    );
    assert!(matches!(
        LfGlobalPrefix::parse(&bytes, packet),
        Err(VarDctPacketError::BaseCorrelationOutOfRange {
            channel: "X",
            value: -4.5,
        })
    ));

    let (bytes, packet) =
        custom_lf_global([0x7c00, 0x3000, 0x3400], (0, 0), [0, 0x3c00], [128, 128]);
    assert!(matches!(
        LfGlobalPrefix::parse(&bytes, packet),
        Err(VarDctPacketError::MetadataBitstream {
            stage: "LF channel dequantization multiplier",
            ..
        })
    ));
}

#[test]
fn accepts_green_queen_physical_group_packets() {
    let bytes = decode_hex(include_str!(
        "../../jxl_gpu_bitstream/test-data/green_queen_vardct_e3.jxl.hex"
    ));
    let profile = negotiate(&bytes);
    assert_eq!((profile.width(), profile.height()), (438, 589));
    assert_eq!(profile.group_count(), 6);
    assert!(profile.adaptive_lf_smoothing());
    assert_eq!(profile.lf_quant_stream_index(0).unwrap(), 1);
    assert_eq!(profile.hf_metadata_stream_index(0).unwrap(), 3);
    assert_eq!(
        profile.low_frequency_group_rect(0).unwrap(),
        jxl_wgpu_decode::vardct::frontend::VarDctGroupRect {
            x: 0,
            y: 0,
            width: 438,
            height: 589,
        }
    );
    assert_eq!(
        profile.pass_group_rect(5).unwrap(),
        jxl_wgpu_decode::vardct::frontend::VarDctGroupRect {
            x: 256,
            y: 512,
            width: 182,
            height: 77,
        }
    );
    let VarDctSectionLayout::Sections {
        lf_global,
        lf_groups,
        hf_global,
        pass_groups,
    } = profile.sections().clone()
    else {
        panic!("green queen uses a multi-entry TOC")
    };
    assert_eq!(lf_groups.len(), 1);
    assert_eq!(pass_groups.len(), 6);
    let lf_group_prefix = LfGroupPrefix::parse(&bytes, lf_groups[0], 438, 589, 1).unwrap();
    assert_eq!(lf_group_prefix.extra_precision, 0);
    assert_eq!(lf_group_prefix.lf_quant.stream_index, 1);
    assert_eq!(lf_group_prefix.lf_quant.channels.len(), 3);
    assert!(
        lf_group_prefix
            .lf_quant
            .channels
            .iter()
            .all(|channel| (channel.width, channel.height) == (55, 74))
    );
    assert_eq!(lf_group_prefix.lf_quant.sample_count(), Some(55 * 74 * 3));
    // This cursor is returned by the LF Modular GPU dispatch. Its fixture value is cross-checked
    // against the dev-only jxl-vardct scalar oracle in `inspect_vardct`.
    let hf_metadata =
        HfMetadataPrefix::parse(&bytes, 67_171, lf_groups[0].end().unwrap(), 438, 589, 3).unwrap();
    assert_eq!(
        (
            hf_metadata.block_width,
            hf_metadata.block_height,
            hf_metadata.block_count,
        ),
        (55, 74, 4_070)
    );
    assert_eq!(hf_metadata.metadata.stream_index, 3);
    assert_eq!(
        hf_metadata.metadata.channels,
        [(7, 10), (7, 10), (4_070, 2), (55, 74),]
            .map(|(width, height)| ModularChannelPlan { width, height })
    );
    let prefix = LfGlobalPrefix::parse(&bytes, lf_global).unwrap();
    assert_eq!((prefix.global_scale, prefix.quant_lf), (8813, 10));
    assert_eq!(prefix.hf_block_context.num_block_clusters, 15);
    assert_eq!(prefix.hf_block_context.block_context_map.len(), 39);
    assert!(prefix.global_ma_tree_bit_offset.unwrap() > lf_global.offset);
    let hf_prefix = HfGlobalPrefix::parse(&bytes, hf_global, 6).unwrap();
    assert_eq!(hf_prefix.num_hf_presets, 1);
    assert_ne!(hf_prefix.used_orders, 0);
    assert!(hf_prefix.order_entropy_bit_offset > hf_global.offset);
}

#[test]
fn permuted_toc_is_normalized_to_logical_pass_group_order() {
    let bytes = decode_hex(include_str!(
        "../test-data/green_queen_vardct_permuted.jxl.hex"
    ));
    let inventory = inventory(&bytes);
    let frame = &inventory.frames[0];
    assert!(frame.toc_permuted);

    let physical_group_order = frame
        .sections
        .iter()
        .filter_map(|section| match section.kind {
            FrameSectionKind::PassGroup {
                pass_index: 0,
                group_index,
            } => Some(group_index),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(physical_group_order.len() as u64, frame.group_count);
    assert_ne!(
        physical_group_order,
        (0..frame.group_count).collect::<Vec<_>>()
    );

    let profile = StandardVarDctProfile::negotiate(&inventory).unwrap();
    assert_eq!((profile.width(), profile.height()), (438, 589));
    let VarDctSectionLayout::Sections { pass_groups, .. } = profile.sections() else {
        panic!("center-first fixture must expose independent pass-group sections")
    };
    assert_eq!(pass_groups.len() as u64, frame.group_count);
    for (group_index, &range) in pass_groups.iter().enumerate() {
        let physical = frame
            .sections
            .iter()
            .find(|section| {
                section.kind
                    == FrameSectionKind::PassGroup {
                        pass_index: 0,
                        group_index: group_index as u64,
                    }
            })
            .unwrap();
        assert_eq!(range, physical.bits);
    }

    let packet = BoundedVarDctPacketPlan::parse(&bytes, &inventory, &profile).unwrap();
    assert_eq!(
        packet.hf_coefficients.unwrap().pass_groups,
        pass_groups.as_slice()
    );
}

#[test]
fn rejects_packets_and_geometries_outside_gpu_bounds() {
    let bytes = decode_hex(include_str!("../test-data/basic.jxl.hex"));
    let error = LfGlobalPrefix::parse(
        &bytes,
        BitRange {
            offset: 0,
            length: u64::try_from(bytes.len()).unwrap() * 8 + 1,
        },
    )
    .unwrap_err();
    assert!(matches!(error, VarDctPacketError::PacketBoundary { .. }));

    let error = HfMetadataPrefix::parse(&bytes, 1, 8, 0, 1, 3).unwrap_err();
    assert!(matches!(error, VarDctPacketError::GeometryOverflow));

    let green = decode_hex(include_str!(
        "../../jxl_gpu_bitstream/test-data/green_queen_vardct_e3.jxl.hex"
    ));
    let profile = negotiate(&green);
    let VarDctSectionLayout::Sections { lf_groups, .. } = profile.sections() else {
        unreachable!()
    };
    let error = LfGroupPrefix::parse(&green, lf_groups[0], u32::MAX, u32::MAX, 1).unwrap_err();
    assert!(matches!(
        error,
        VarDctPacketError::GpuAddressSpace {
            field: "Modular sample count",
            ..
        }
    ));
}

#[test]
fn preserves_typed_profile_and_metadata_error_causes() {
    let bytes = decode_hex(include_str!("../test-data/basic.jxl.hex"));
    let mut progressive = inventory(&bytes);
    progressive.frames[0].num_passes = 2;
    assert_eq!(
        StandardVarDctProfile::negotiate(&progressive).unwrap_err(),
        VarDctFrontendError::Unsupported {
            feature: UnsupportedVarDctFeature::ProgressivePasses,
        }
    );

    let error = LfGlobalPrefix::parse(
        &[1],
        BitRange {
            offset: 0,
            length: 8,
        },
    )
    .unwrap_err();
    assert!(matches!(error, VarDctPacketError::MetadataBitstream { .. }));
    assert!(std::error::Error::source(&error).is_some());

    let profile = negotiate(&bytes);
    assert_eq!(
        profile.hf_metadata_stream_index(1).unwrap_err(),
        VarDctFrontendError::GroupIndexOutOfRange {
            index: 1,
            group_count: 1,
        }
    );

    let mut skip_smoothing = inventory(&bytes);
    skip_smoothing.frames[0].flags |= 0x80;
    assert!(
        !StandardVarDctProfile::negotiate(&skip_smoothing)
            .unwrap()
            .adaptive_lf_smoothing()
    );
}

#[test]
fn rejects_inconsistent_dynamic_section_topology_with_typed_errors() {
    let bytes = decode_hex(include_str!(
        "../../jxl_gpu_bitstream/test-data/green_queen_vardct_e3.jxl.hex"
    ));

    let mut duplicate = inventory(&bytes);
    let lf_global = *duplicate.frames[0]
        .sections
        .iter()
        .find(|section| section.kind == FrameSectionKind::LowFrequencyGlobal)
        .unwrap();
    duplicate.frames[0].sections.push(lf_global);
    assert_eq!(
        StandardVarDctProfile::negotiate(&duplicate).unwrap_err(),
        VarDctFrontendError::DuplicateSection {
            kind: "LF-global",
            index: 0,
        }
    );

    let mut missing = inventory(&bytes);
    missing.frames[0]
        .sections
        .retain(|section| section.kind != FrameSectionKind::HighFrequencyGlobal);
    assert_eq!(
        StandardVarDctProfile::negotiate(&missing).unwrap_err(),
        VarDctFrontendError::MissingSection {
            kind: "HF-global",
            index: 0,
        }
    );

    let mut out_of_range = inventory(&bytes);
    let section = out_of_range.frames[0]
        .sections
        .iter_mut()
        .find(|section| matches!(section.kind, FrameSectionKind::LowFrequencyGroup { .. }))
        .unwrap();
    section.kind = FrameSectionKind::LowFrequencyGroup { group_index: 1 };
    assert_eq!(
        StandardVarDctProfile::negotiate(&out_of_range).unwrap_err(),
        VarDctFrontendError::GroupIndexOutOfRange {
            index: 1,
            group_count: 1,
        }
    );
}
