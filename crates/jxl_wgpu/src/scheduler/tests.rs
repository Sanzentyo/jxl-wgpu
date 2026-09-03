use std::mem::{align_of, size_of};

use jxl_gpu_formats::{
    ImageLayout, PixelFormatClass, WgslNumericCapability, classify_pixel_format,
};
use jxl_gpu_protocol::{
    Border2d, Extent2d, MemoryMode, OutputColorEncoding, OutputDesc, OutputId, PlaneRole,
    PrecisionContract, RenderNode, SaveParams, Scale2d,
};

use crate::arena::{ArenaAllocation, ArenaPlan};

use super::*;

fn abi_words<T: Pod>(value: &T) -> &[u32] {
    bytemuck::cast_slice(std::slice::from_ref(value))
}

fn assert_sequential_words<T: Pod>(value: &T) {
    let expected =
        (1..=u32::try_from(size_of::<T>() / size_of::<u32>()).unwrap()).collect::<Vec<_>>();
    assert_eq!(abi_words(value), expected);
}

fn assert_wgsl_fields(shader: &str, name: &str, expected: &[&str]) {
    let module = naga::front::wgsl::parse_str(shader).expect("WGSL parses");
    let ty = module
        .types
        .iter()
        .map(|(_, ty)| ty)
        .find(|ty| ty.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("WGSL struct '{name}' is missing"));
    let naga::TypeInner::Struct { members, .. } = &ty.inner else {
        panic!("WGSL type '{name}' is not a struct");
    };
    let actual = members
        .iter()
        .map(|member| member.name.as_deref().expect("WGSL struct member is named"))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "WGSL field-order drift for {name}");
}

#[test]
fn canonical_classifier_drives_vpi_image_output_support() {
    let mut color_count = 0;
    let mut numeric_count = 0;
    for predefined in jxl_gpu_formats::vpi::VpiPitchLinearFormat::ALL {
        let format = predefined.pixel_format();
        let layout = ImageLayout::packed(Extent2d::new(3, 3), format.clone()).unwrap();
        match classify_pixel_format(&format).unwrap() {
            PixelFormatClass::Color(_) => {
                color_count += 1;
                prepare_image_output(&layout)
                    .unwrap_or_else(|error| panic!("{}: {error}", predefined.name()));
            }
            PixelFormatClass::Numeric(numeric) => {
                numeric_count += 1;
                let Error::Unsupported(message) = prepare_image_output(&layout).unwrap_err() else {
                    panic!("{} did not return typed Unsupported", predefined.name());
                };
                assert!(message.contains("does not assign color semantics"));
                if numeric.wgsl == WgslNumericCapability::UnavailableFloat64 {
                    assert!(message.contains("no native F64 arithmetic"));
                }
            }
        }
    }
    assert_eq!(color_count, 20);
    assert_eq!(numeric_count, 10);
}

#[test]
fn uniform_abi_sizes_are_explicit_and_naturally_aligned() {
    let sizes = [
        ("CopyParams", size_of::<CopyParams>(), 16),
        ("ModularParams", size_of::<ModularParams>(), 32),
        (
            "ChromaUpsampleUniform",
            size_of::<ChromaUpsampleUniform>(),
            32,
        ),
        ("Chroma2dUniform", size_of::<Chroma2dUniform>(), 32),
        ("GaborishUniform", size_of::<GaborishUniform>(), 32),
        ("GaborishRgbUniform", size_of::<GaborishRgbUniform>(), 80),
        ("EpfUniform", size_of::<EpfUniform>(), 80),
        ("UpsampleUniform", size_of::<UpsampleUniform>(), 32),
        ("YcbcrUniform", size_of::<YcbcrUniform>(), 32),
        ("XybUniform", size_of::<XybUniform>(), 128),
        ("TransferUniform", size_of::<TransferUniform>(), 64),
        ("PremultiplyUniform", size_of::<PremultiplyUniform>(), 32),
        ("SaveUniform", size_of::<SaveUniform>(), 32),
        ("ImageOutputUniform", size_of::<ImageOutputUniform>(), 176),
    ];
    for (name, actual, expected) in sizes {
        assert_eq!(actual, expected, "Rust/WGSL ABI size drift for {name}");
        assert_eq!(actual % 16, 0, "uniform {name} is not 16-byte sized");
    }
    for (name, alignment) in [
        ("CopyParams", align_of::<CopyParams>()),
        ("ModularParams", align_of::<ModularParams>()),
        ("ChromaUpsampleUniform", align_of::<ChromaUpsampleUniform>()),
        ("Chroma2dUniform", align_of::<Chroma2dUniform>()),
        ("GaborishUniform", align_of::<GaborishUniform>()),
        ("GaborishRgbUniform", align_of::<GaborishRgbUniform>()),
        ("EpfUniform", align_of::<EpfUniform>()),
        ("UpsampleUniform", align_of::<UpsampleUniform>()),
        ("YcbcrUniform", align_of::<YcbcrUniform>()),
        ("XybUniform", align_of::<XybUniform>()),
        ("TransferUniform", align_of::<TransferUniform>()),
        ("BlendUniform", align_of::<BlendUniform>()),
        ("PremultiplyUniform", align_of::<PremultiplyUniform>()),
        ("ExtendUniform", align_of::<ExtendUniform>()),
        ("SaveUniform", align_of::<SaveUniform>()),
        ("ImageOutputUniform", align_of::<ImageOutputUniform>()),
    ] {
        assert_eq!(
            alignment, 4,
            "Rust/WGSL natural ABI alignment drift for {name}"
        );
    }
}

#[test]
fn uniform_rust_word_order_matches_wgsl_field_order() {
    assert_sequential_words(&CopyParams {
        width: 1,
        height: 2,
        input_stride: 3,
        output_stride: 4,
    });
    assert_wgsl_fields(
        include_str!("../../shaders/copy.wgsl"),
        "Params",
        &["width", "height", "input_stride", "output_stride"],
    );

    assert_sequential_words(&ModularParams {
        width: 1,
        height: 2,
        input_stride: 3,
        output_stride: 4,
        multiplier: f32::from_bits(5),
        bias: f32::from_bits(6),
        _padding: [7, 8],
    });
    assert_wgsl_fields(
        include_str!("../../shaders/modular_to_f32.wgsl"),
        "Params",
        &[
            "width",
            "height",
            "input_stride",
            "output_stride",
            "multiplier",
            "bias",
            "_pad0",
            "_pad1",
        ],
    );

    assert_sequential_words(&ChromaUpsampleUniform {
        input_width: 1,
        input_height: 2,
        output_width: 3,
        output_height: 4,
        input_stride: 5,
        output_stride: 6,
        axis: 7,
        _padding: 8,
    });
    assert_wgsl_fields(
        include_str!("../../shaders/chroma_upsample.wgsl"),
        "Params",
        &[
            "input_width",
            "input_height",
            "output_width",
            "output_height",
            "input_stride",
            "output_stride",
            "axis",
            "_pad0",
        ],
    );

    assert_sequential_words(&Chroma2dUniform {
        input_width: 1,
        input_height: 2,
        output_width: 3,
        output_height: 4,
        input_stride: 5,
        output_stride: 6,
        _padding: [7, 8],
    });
    assert_wgsl_fields(
        include_str!("../../shaders/chroma_2d.wgsl"),
        "Params",
        &[
            "input_width",
            "input_height",
            "output_width",
            "output_height",
            "input_stride",
            "output_stride",
            "_pad0",
            "_pad1",
        ],
    );

    assert_sequential_words(&GaborishUniform {
        width: 1,
        height: 2,
        input_stride: 3,
        output_stride: 4,
        weight0: f32::from_bits(5),
        weight1: f32::from_bits(6),
        weight2: f32::from_bits(7),
        _padding: 8,
    });
    assert_wgsl_fields(
        include_str!("../../shaders/gaborish.wgsl"),
        "Params",
        &[
            "width",
            "height",
            "input_stride",
            "output_stride",
            "weight0",
            "weight1",
            "weight2",
            "_pad0",
        ],
    );

    assert_sequential_words(&GaborishRgbUniform {
        width: 1,
        height: 2,
        input_stride_x: 3,
        input_stride_y: 4,
        input_stride_b: 5,
        output_stride_x: 6,
        output_stride_y: 7,
        output_stride_b: 8,
        weights_x: [
            f32::from_bits(9),
            f32::from_bits(10),
            f32::from_bits(11),
            f32::from_bits(12),
        ],
        weights_y: [
            f32::from_bits(13),
            f32::from_bits(14),
            f32::from_bits(15),
            f32::from_bits(16),
        ],
        weights_b: [
            f32::from_bits(17),
            f32::from_bits(18),
            f32::from_bits(19),
            f32::from_bits(20),
        ],
    });
    assert_wgsl_fields(
        include_str!("../../shaders/gaborish_rgb.wgsl"),
        "Params",
        &[
            "width",
            "height",
            "input_stride_x",
            "input_stride_y",
            "input_stride_b",
            "output_stride_x",
            "output_stride_y",
            "output_stride_b",
            "weight0_x",
            "weight1_x",
            "weight2_x",
            "_pad_x",
            "weight0_y",
            "weight1_y",
            "weight2_y",
            "_pad_y",
            "weight0_b",
            "weight1_b",
            "weight2_b",
            "_pad_b",
        ],
    );

    assert_sequential_words(&EpfUniform {
        width: 1,
        height: 2,
        input_stride_x: 3,
        input_stride_y: 4,
        input_stride_b: 5,
        output_stride_x: 6,
        output_stride_y: 7,
        output_stride_b: 8,
        sigma_width: 9,
        sigma_height: 10,
        sigma_stride: 11,
        sigma_is_plane: 12,
        sigma_scale: f32::from_bits(13),
        border_sad_mul: f32::from_bits(14),
        channel_scale_x: f32::from_bits(15),
        channel_scale_y: f32::from_bits(16),
        channel_scale_b: f32::from_bits(17),
        min_sigma: f32::from_bits(18),
        _padding: [19, 20],
    });
    assert_wgsl_fields(
        include_str!("../../shaders/epf.wgsl"),
        "Params",
        &[
            "width",
            "height",
            "input_stride_x",
            "input_stride_y",
            "input_stride_b",
            "output_stride_x",
            "output_stride_y",
            "output_stride_b",
            "sigma_width",
            "sigma_height",
            "sigma_stride",
            "sigma_is_plane",
            "sigma_scale",
            "border_sad_mul",
            "channel_scale_x",
            "channel_scale_y",
            "channel_scale_b",
            "min_sigma",
            "_pad0",
            "_pad1",
        ],
    );

    assert_sequential_words(&UpsampleUniform {
        input_width: 1,
        input_height: 2,
        output_width: 3,
        output_height: 4,
        input_stride: 5,
        output_stride: 6,
        factor: 7,
        _padding: 8,
    });
    assert_wgsl_fields(
        include_str!("../../shaders/upsample.wgsl"),
        "Params",
        &[
            "input_width",
            "input_height",
            "output_width",
            "output_height",
            "input_stride",
            "output_stride",
            "factor",
            "_pad0",
        ],
    );

    assert_sequential_words(&YcbcrUniform {
        width: 1,
        height: 2,
        cb_stride: 3,
        y_stride: 4,
        cr_stride: 5,
        output_stride: 6,
        component: 7,
        _padding: 8,
    });
    assert_wgsl_fields(
        include_str!("../../shaders/ycbcr_to_rgb.wgsl"),
        "Params",
        &[
            "width",
            "height",
            "cb_stride",
            "y_stride",
            "cr_stride",
            "output_stride",
            "component",
            "_pad0",
        ],
    );

    assert_sequential_words(&XybUniform {
        width: 1,
        height: 2,
        input_stride_x: 3,
        input_stride_y: 4,
        input_stride_b: 5,
        output_stride_r: 6,
        output_stride_g: 7,
        output_stride_b: 8,
        matrix_r: [
            f32::from_bits(9),
            f32::from_bits(10),
            f32::from_bits(11),
            f32::from_bits(12),
        ],
        matrix_g: [
            f32::from_bits(13),
            f32::from_bits(14),
            f32::from_bits(15),
            f32::from_bits(16),
        ],
        matrix_b: [
            f32::from_bits(17),
            f32::from_bits(18),
            f32::from_bits(19),
            f32::from_bits(20),
        ],
        bias_cbrt: [
            f32::from_bits(21),
            f32::from_bits(22),
            f32::from_bits(23),
            f32::from_bits(24),
        ],
        scaled_bias: [
            f32::from_bits(25),
            f32::from_bits(26),
            f32::from_bits(27),
            f32::from_bits(28),
        ],
        intensity_scale: f32::from_bits(29),
        _padding: [30, 31, 32],
    });
    assert_wgsl_fields(
        include_str!("../../shaders/xyb_to_rgb.wgsl"),
        "Params",
        &[
            "width",
            "height",
            "input_stride_x",
            "input_stride_y",
            "input_stride_b",
            "output_stride_r",
            "output_stride_g",
            "output_stride_b",
            "matrix_r",
            "matrix_g",
            "matrix_b",
            "bias_cbrt",
            "scaled_bias",
            "intensity_scale",
            "_pad0",
            "_pad1",
            "_pad2",
        ],
    );

    assert_sequential_words(&TransferUniform {
        width: 1,
        height: 2,
        input_stride_r: 3,
        input_stride_g: 4,
        input_stride_b: 5,
        output_stride_r: 6,
        output_stride_g: 7,
        output_stride_b: 8,
        transfer: 9,
        gamma: f32::from_bits(10),
        intensity_target: f32::from_bits(11),
        min_nits: f32::from_bits(12),
        luminance_rgb: [
            f32::from_bits(13),
            f32::from_bits(14),
            f32::from_bits(15),
            f32::from_bits(16),
        ],
    });
    assert_wgsl_fields(
        include_str!("../../shaders/transfer_function.wgsl"),
        "Params",
        &[
            "width",
            "height",
            "input_stride_r",
            "input_stride_g",
            "input_stride_b",
            "output_stride_r",
            "output_stride_g",
            "output_stride_b",
            "transfer",
            "gamma",
            "intensity_target",
            "min_nits",
            "luminance_rgb",
        ],
    );

    assert_sequential_words(&BlendUniform {
        width: 1,
        height: 2,
        base_stride: 3,
        source_stride: 4,
        output_stride: 5,
        base_alpha_stride: 6,
        source_alpha_stride: 7,
        mode: 8,
        component: 9,
        clamp: 10,
        alpha_associated: 11,
        has_alpha: 12,
    });
    assert_wgsl_fields(
        include_str!("../../shaders/blend.wgsl"),
        "Params",
        &[
            "width",
            "height",
            "base_stride",
            "source_stride",
            "output_stride",
            "base_alpha_stride",
            "source_alpha_stride",
            "mode",
            "component",
            "clamp",
            "alpha_associated",
            "has_alpha",
        ],
    );

    assert_sequential_words(&PremultiplyUniform {
        width: 1,
        height: 2,
        color_stride: 3,
        alpha_stride: 4,
        output_stride: 5,
        _padding: [6, 7, 8],
    });
    assert_wgsl_fields(
        include_str!("../../shaders/premultiply_alpha.wgsl"),
        "Params",
        &[
            "width",
            "height",
            "color_stride",
            "alpha_stride",
            "output_stride",
            "_pad0",
            "_pad1",
            "_pad2",
        ],
    );

    assert_sequential_words(&ExtendUniform {
        width: 1,
        height: 2,
        frame_width: 3,
        frame_height: 4,
        frame_stride: 5,
        reference_stride: 6,
        output_stride: 7,
        origin_x: 8,
        origin_y: 9,
        has_reference: 10,
        _padding: [11, 12],
    });
    assert_wgsl_fields(
        include_str!("../../shaders/extend.wgsl"),
        "Params",
        &[
            "width",
            "height",
            "frame_width",
            "frame_height",
            "frame_stride",
            "reference_stride",
            "output_stride",
            "origin_x",
            "origin_y",
            "has_reference",
            "_pad0",
            "_pad1",
        ],
    );

    assert_sequential_words(&SaveUniform {
        width: 1,
        height: 2,
        source_stride: 3,
        channels: 4,
        channel: 5,
        layout: 6,
        orientation: 7,
        _padding: 8,
    });
    assert_wgsl_fields(
        include_str!("../../shaders/save.wgsl"),
        "Params",
        &[
            "width",
            "height",
            "source_stride",
            "channels",
            "channel",
            "output_layout",
            "orientation",
            "_pad0",
        ],
    );

    assert_sequential_words(&ImageOutputUniform {
        width: 1,
        height: 2,
        source_width: 3,
        source_height: 4,
        r_stride: 5,
        g_stride: 6,
        b_stride: 7,
        kind: 8,
        channels: 9,
        order: 10,
        matrix: 11,
        range: 12,
        siting_x: 13,
        siting_y: 14,
        subsample_x: 15,
        subsample_y: 16,
        bits: 17,
        storage_bits: 18,
        plane0_offset: 19,
        plane0_stride: 20,
        plane1_offset: 21,
        plane1_stride: 22,
        plane2_offset: 23,
        plane2_stride: 24,
        plane3_offset: 25,
        plane3_stride: 26,
        logical_size: 27,
        dispatch_width: 28,
        orientation: 29,
        source_transfer: 30,
        target_transfer: 31,
        _padding: 32,
        primaries_r: [
            f32::from_bits(33),
            f32::from_bits(34),
            f32::from_bits(35),
            f32::from_bits(36),
        ],
        primaries_g: [
            f32::from_bits(37),
            f32::from_bits(38),
            f32::from_bits(39),
            f32::from_bits(40),
        ],
        primaries_b: [
            f32::from_bits(41),
            f32::from_bits(42),
            f32::from_bits(43),
            f32::from_bits(44),
        ],
    });
    assert_wgsl_fields(
        include_str!("../../shaders/rgb_to_image.wgsl"),
        "Params",
        &[
            "width",
            "height",
            "source_width",
            "source_height",
            "r_stride",
            "g_stride",
            "b_stride",
            "kind",
            "channels",
            "order",
            "matrix",
            "range",
            "siting_x",
            "siting_y",
            "subsample_x",
            "subsample_y",
            "bits",
            "storage_bits",
            "plane0_offset",
            "plane0_stride",
            "plane1_offset",
            "plane1_stride",
            "plane2_offset",
            "plane2_stride",
            "plane3_offset",
            "plane3_stride",
            "logical_size",
            "dispatch_width",
            "orientation",
            "source_transfer",
            "target_transfer",
            "_padding",
            "primaries_r",
            "primaries_g",
            "primaries_b",
        ],
    );
}

fn execution(allocations: Vec<ArenaAllocation>, size_bytes: u64) -> ExecutionPlan {
    ExecutionPlan {
        memory_mode: MemoryMode::Resident,
        dispatches: Vec::new(),
        arena: ArenaPlan {
            size_bytes,
            peak_live_bytes: size_bytes,
            peak_scratch_bytes: 0,
            allocations,
        },
        tile_extents: BTreeMap::new(),
        resident_bytes: size_bytes,
        scratch_bytes: 0,
        groups_per_batch: 1,
    }
}

fn allocation(
    plane: u32,
    offset: u64,
    size: u64,
    first_use: usize,
    last_use: usize,
) -> ArenaAllocation {
    ArenaAllocation {
        plane: PlaneId(plane),
        offset,
        size,
        first_use,
        last_use,
    }
}

#[test]
fn resident_slots_reuse_disjoint_lifetimes_without_double_counting() {
    let execution = execution(
        vec![
            allocation(0, 0, 17, 0, 0),
            allocation(1, 0, 9, 1, 1),
            allocation(2, 32, 33, 0, 2),
        ],
        80,
    );

    assert_eq!(
        resident_slot_sizes(&execution, 16).unwrap(),
        BTreeMap::from([(0, 32), (32, 48)])
    );
}

#[test]
fn resident_slots_reject_false_aggregate_budget() {
    let execution = execution(
        vec![allocation(0, 0, 17, 0, 0), allocation(1, 32, 33, 1, 1)],
        64,
    );

    assert!(matches!(
        resident_slot_sizes(&execution, 16),
        Err(Error::Execution(message)) if message.contains("physical slots require 80 bytes")
    ));
}

#[test]
fn resident_slots_reject_simultaneously_live_aliases() {
    let execution = execution(
        vec![allocation(0, 0, 16, 0, 1), allocation(1, 0, 16, 1, 2)],
        16,
    );

    assert!(matches!(
        resident_slot_sizes(&execution, 16),
        Err(Error::Execution(message)) if message.contains("simultaneously live")
    ));
}

#[test]
fn transient_estimate_includes_uniform_packing_and_readback_buffers() {
    let extent = Extent2d::new(3, 3);
    let output = OutputId(0);
    let plan = RenderPlan {
        planes: vec![
            PlaneDesc {
                id: PlaneId(0),
                extent,
                stride: 3,
                sample_type: SampleType::F32,
                role: PlaneRole::Source,
            },
            PlaneDesc {
                id: PlaneId(1),
                extent,
                stride: 3,
                sample_type: SampleType::F32,
                role: PlaneRole::Intermediate,
            },
        ],
        nodes: vec![
            RenderNode {
                name: "copy".into(),
                op: RenderOp::Copy,
                inputs: vec![PlaneId(0)],
                outputs: vec![PlaneId(1)],
                resources: Vec::new(),
                scale: Scale2d::IDENTITY,
                border: Border2d::default(),
                precision: PrecisionContract::Exact,
            },
            RenderNode {
                name: "save".into(),
                op: RenderOp::Save(SaveParams {
                    output,
                    sample_type: SampleType::F32,
                    channels: vec![PlaneId(1)],
                    layout: OutputLayout::Planar,
                    orientation: jxl_gpu_protocol::OutputOrientation::Identity,
                }),
                inputs: vec![PlaneId(1)],
                outputs: Vec::new(),
                resources: Vec::new(),
                scale: Scale2d::IDENTITY,
                border: Border2d::default(),
                precision: PrecisionContract::Exact,
            },
        ],
        outputs: vec![OutputDesc {
            id: output,
            extent,
            sample_type: SampleType::F32,
            channels: 1,
            layout: OutputLayout::Planar,
            color_encoding: OutputColorEncoding::NonColor,
        }],
    };
    let expected = std::mem::size_of::<CopyParams>()
        + std::mem::size_of::<SaveUniform>()
        + 2 * 3 * 3 * std::mem::size_of::<f32>();
    assert_eq!(
        transient_bytes(
            &plan,
            &execution(Vec::new(), 0),
            &BTreeMap::new(),
            &BTreeMap::new(),
            OutputTarget {
                mode: OutputMode::CpuReadback,
                encoding: OutputEncoding::Original,
                direct_readback: false,
            },
        )
        .unwrap(),
        expected as u64
    );
    assert_eq!(
        transient_bytes(
            &plan,
            &execution(Vec::new(), 0),
            &BTreeMap::new(),
            &BTreeMap::new(),
            OutputTarget {
                mode: OutputMode::GpuOnly,
                encoding: OutputEncoding::Original,
                direct_readback: false,
            },
        )
        .unwrap(),
        (expected - 3 * 3 * std::mem::size_of::<f32>()) as u64
    );
    assert_eq!(
        transient_bytes(
            &plan,
            &execution(Vec::new(), 0),
            &BTreeMap::new(),
            &BTreeMap::new(),
            OutputTarget {
                mode: OutputMode::CpuReadback,
                encoding: OutputEncoding::Original,
                direct_readback: true,
            },
        )
        .unwrap(),
        (expected - 3 * 3 * std::mem::size_of::<f32>()) as u64
    );
}

#[test]
fn packed_storage_size_is_checked_against_both_device_limits() {
    let limits = wgpu::Limits {
        max_buffer_size: 1024,
        max_storage_buffer_binding_size: 512,
        ..wgpu::Limits::default()
    };
    validate_storage_buffer_size(&limits, 512, "test output").unwrap();
    assert!(matches!(
        validate_storage_buffer_size(&limits, 516, "test output"),
        Err(Error::ResourceLimit(message))
            if message.contains("516 bytes") && message.contains("limit 512")
    ));

    let limits = wgpu::Limits {
        max_buffer_size: 256,
        max_storage_buffer_binding_size: 512,
        ..wgpu::Limits::default()
    };
    assert!(matches!(
        validate_storage_buffer_size(&limits, 260, "test output"),
        Err(Error::ResourceLimit(message))
            if message.contains("260 bytes") && message.contains("limit 256")
    ));
}

#[test]
fn upsample_nodes_share_weights_storage_plan_and_uniform_transient_accounting() {
    use jxl_gpu_protocol::{
        PlaneDesc, PlaneId, PlaneRole, ResourceData, ResourceId, ResourceUpdate, SampleType,
        StoragePlan, UpsamplingFactor,
    };
    use crate::resident_image_upsample::PreparedUpsamplingMemoryPlan;

    // 1. Verify StoragePlan computation from UpsamplingFactor
    let factor2 = UpsamplingFactor::X2;
    let storage_plan = factor2.weights_storage_plan();
    assert_eq!(storage_plan.bytes, 2 * 2 * 25 * 4); // 400 bytes

    let memory_plan = PreparedUpsamplingMemoryPlan::from(storage_plan);
    assert_eq!(memory_plan.storage_bytes, 400);
    assert_eq!(StoragePlan::from(memory_plan), storage_plan);

    // 2. Build RenderPlan with 3 upsampling nodes sharing single weights ResourceId
    let weights_id = ResourceId(42);
    let mut plan = RenderPlan::default();
    plan.planes.push(PlaneDesc {
        id: PlaneId(1),
        sample_type: SampleType::F32,
        extent: Extent2d::new(8, 8),
        stride: 8,
        role: PlaneRole::Source,
    });
    plan.planes.push(PlaneDesc {
        id: PlaneId(2),
        sample_type: SampleType::F32,
        extent: Extent2d::new(16, 16),
        stride: 16,
        role: PlaneRole::Intermediate,
    });
    plan.planes.push(PlaneDesc {
        id: PlaneId(3),
        sample_type: SampleType::F32,
        extent: Extent2d::new(8, 8),
        stride: 8,
        role: PlaneRole::Source,
    });
    plan.planes.push(PlaneDesc {
        id: PlaneId(4),
        sample_type: SampleType::F32,
        extent: Extent2d::new(16, 16),
        stride: 16,
        role: PlaneRole::Intermediate,
    });

    plan.add_upsample_node("upsample_r", factor2, weights_id, PlaneId(1), PlaneId(2));
    plan.add_upsample_node("upsample_g", factor2, weights_id, PlaneId(3), PlaneId(4));

    // Verify RenderNodes are correctly constructed
    assert_eq!(plan.nodes.len(), 2);
    assert_eq!(plan.nodes[0].border, Border2d::symmetric(2, 2));
    assert_eq!(plan.nodes[0].scale, Scale2d::new(2, 2));
    assert_eq!(plan.nodes[0].resources, vec![weights_id]);
    assert_eq!(plan.nodes[1].resources, vec![weights_id]);

    let mut resources = BTreeMap::new();
    resources.insert(
        weights_id,
        ResourceUpdate {
            id: weights_id,
            revision: 1,
            data: ResourceData::F32(vec![0.04; 100]),
        },
    );

    let execution = execution(Vec::new(), 0);
    let groups = BTreeMap::new();
    let target = OutputTarget {
        mode: OutputMode::CpuReadback,
        encoding: OutputEncoding::Original,
        direct_readback: false,
    };

    let tb = transient_bytes(&plan, &execution, &groups, &resources, target).unwrap();
    // 2 upsample nodes each need 1 UpsampleUniform
    assert_eq!(tb, (size_of::<UpsampleUniform>() * 2) as u64);
}
