use std::sync::Arc;

use jxl_gpu_bitstream::{FrameSectionKind, StreamSlice};
use jxl_test_support::{fixtures::modular_ycbcr, offline::hex};

use super::{parse_dc_global_ir, parse_lf_channel_dequantization, parse_standard_modular_profile};
use crate::GpuCodestream;
use crate::modular_inverse::{ModularInverseJob, plan_modular_inverse};
use crate::modular_transform::ModularTransformIr;

#[test]
fn transformed_ycbcr_topology_matches_native_channels_including_empty_residuals() {
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/modular_ycbcr");
    for case in modular_ycbcr::cases()
        .into_iter()
        .filter(|case| !case.transforms.is_empty())
    {
        let bytes = hex::unhex(
            &std::fs::read_to_string(directory.join(format!("{}.jxl.hex", case.name))).unwrap(),
        );
        let parsed = jxl_gpu_bitstream::parse(&bytes, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        case.validate(&inventory);
        let native = modular_ycbcr::NativeTopology::parse(
            &std::fs::read_to_string(directory.join(format!("{}.topology", case.name))).unwrap(),
        );
        let source = GpuCodestream::from_spans([(
            0,
            StreamSlice::from_shared(Arc::from(parsed.codestream())),
        )])
        .unwrap();
        let frame = &inventory.frames[0];
        let section = frame
            .sections
            .iter()
            .find(|section| {
                matches!(
                    section.kind,
                    FrameSectionKind::Single | FrameSectionKind::LowFrequencyGlobal
                )
            })
            .unwrap();
        let mut reader = source.reader();
        reader.skip_bits(section.bits.offset).unwrap();
        parse_lf_channel_dequantization(&mut reader).unwrap();
        let source_topology =
            crate::modular_geometry::source_topology(&inventory.image_header, frame, 3).unwrap();
        let (_, _, _, _, plan) = parse_dc_global_ir(&mut reader, source_topology).unwrap();
        let actual: Vec<_> = plan
            .topology
            .channels()
            .iter()
            .map(|channel| {
                [
                    i64::from(channel.width),
                    i64::from(channel.height),
                    i64::from(channel.hshift),
                    i64::from(channel.vshift),
                ]
            })
            .collect();
        assert_eq!(
            actual, native.channels,
            "{} native channel topology",
            case.name
        );
        assert_eq!(
            plan.topology.meta_channel_count(),
            native.meta_channels,
            "{} meta channels",
            case.name
        );
        assert_eq!(plan.transforms.len(), native.transform_count);
        assert_eq!(plan.transforms.len(), case.transforms.len());
        for (actual, expected) in plan.transforms.iter().zip(&case.transforms) {
            use modular_ycbcr::Transform;
            match (actual, expected) {
                (
                    ModularTransformIr::Rct(actual),
                    Transform::Rct {
                        begin_channel,
                        rct_type,
                    },
                ) => {
                    assert_eq!(
                        (actual.begin_channel, actual.rct_type),
                        (*begin_channel, *rct_type)
                    );
                }
                (
                    ModularTransformIr::Palette(actual),
                    Transform::Palette {
                        begin_channel,
                        channel_count,
                    },
                ) => {
                    assert_eq!(
                        (actual.begin_channel, actual.channel_count),
                        (*begin_channel, *channel_count)
                    );
                }
                (
                    ModularTransformIr::Squeeze {
                        used_default_parameters,
                        parameters,
                    },
                    Transform::Squeeze(expected),
                ) => {
                    assert_eq!(*used_default_parameters, expected.is_empty());
                    if !expected.is_empty() {
                        let actual: Vec<_> = parameters
                            .iter()
                            .map(|parameter| modular_ycbcr::Squeeze {
                                horizontal: parameter.horizontal,
                                in_place: parameter.in_place,
                                begin_channel: parameter.begin_channel,
                                channel_count: parameter.channel_count,
                            })
                            .collect();
                        assert_eq!(actual, *expected);
                    }
                }
                _ => panic!(
                    "{} unexpected transform: {actual:?} versus {expected:?}",
                    case.name
                ),
            }
        }
        let inverse = plan_modular_inverse(&plan).unwrap();
        assert_eq!(
            inverse
                .jobs()
                .iter()
                .filter(|job| matches!(job, ModularInverseJob::Squeeze { .. }))
                .count(),
            native.squeeze_channels,
            "{} native Squeeze channel count",
            case.name
        );
        parse_standard_modular_profile(&source, &inventory)
            .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
    }
}
