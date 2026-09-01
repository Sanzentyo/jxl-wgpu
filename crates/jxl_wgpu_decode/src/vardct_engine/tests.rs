use std::sync::Arc;

use jxl_gpu_bitstream::{
    EdgePreservingFilterInventory, GaborishInventory, RestorationFilterInventory, StreamSlice,
};
use jxl_wgpu::ResidentGaborishWeights;

use crate::GpuCodestream;
use crate::entropy_window::GroupStreamSegment;

use super::restoration::{VarDctEpfHeader, dequant_matrix_multiplier, restoration_config};
use super::types::VarDctDecodeError;
use super::window_plan::{
    AdaptiveStreamLimitDecision, AdaptiveStreamMemory, copy_stream_segment,
    select_budget_adaptive_stream_limit,
};

fn synthetic_stream_memory(
    fixed_bytes: u64,
    packet_window: bool,
    hf_window: bool,
    stream_limit: u64,
) -> AdaptiveStreamMemory {
    let packet_stream_window_bytes = if packet_window { stream_limit } else { 0 };
    let hf_stream_window_bytes = if hf_window { stream_limit } else { 0 };
    AdaptiveStreamMemory {
        total_frame_bytes: fixed_bytes + packet_stream_window_bytes + hf_stream_window_bytes,
        packet_stream_window_bytes,
        hf_stream_window_bytes,
    }
}

#[test]
fn adaptive_stream_limit_uses_the_largest_aligned_cap_that_fits() {
    let decision = select_budget_adaptive_stream_limit(256, 1_120, |stream_limit| {
        Ok(synthetic_stream_memory(1_000, true, true, stream_limit))
    })
    .unwrap();
    assert_eq!(decision, AdaptiveStreamLimitDecision::Selected(60));

    let unchanged = select_budget_adaptive_stream_limit(256, 1_512, |stream_limit| {
        Ok(synthetic_stream_memory(1_000, true, true, stream_limit))
    })
    .unwrap();
    assert_eq!(unchanged, AdaptiveStreamLimitDecision::Selected(256));
}

#[test]
fn adaptive_stream_limit_reports_the_exact_minimum_window_layout() {
    let decision = select_budget_adaptive_stream_limit(256, 1_079, |stream_limit| {
        Ok(synthetic_stream_memory(1_000, true, true, stream_limit))
    })
    .unwrap();
    assert_eq!(
        decision,
        AdaptiveStreamLimitDecision::BudgetTooSmall {
            required_bytes: 1_080,
        }
    );
}

#[test]
fn adaptive_stream_limit_normalizes_the_caller_cap_to_four_bytes() {
    let decision = select_budget_adaptive_stream_limit(255, 2_000, |stream_limit| {
        Ok(synthetic_stream_memory(1_000, true, false, stream_limit))
    })
    .unwrap();
    assert_eq!(decision, AdaptiveStreamLimitDecision::Selected(252));
}

#[test]
fn bounded_stream_upload_crosses_physical_codestream_spans() {
    let bytes: Arc<[u8]> = Arc::from([10, 11, 12, 13, 14, 15, 16]);
    let source = GpuCodestream::from_spans([
        (
            0,
            StreamSlice::from_shared_range(Arc::clone(&bytes), 0..3).unwrap(),
        ),
        (
            3,
            StreamSlice::from_shared_range(Arc::clone(&bytes), 3..5).unwrap(),
        ),
        (5, StreamSlice::from_shared_range(bytes, 5..7).unwrap()),
    ])
    .unwrap();
    let segment = GroupStreamSegment {
        group_index: 0,
        input_start: 2,
        input_end: 7,
        upload_offset: 3,
        window_logical_start: 0,
        window_upload_start: 0,
        available_token_end: 0,
        stream_token_end: 0,
        window_yield_end: 0,
        flags: 0,
    };
    let mut upload = [0u8; 9];
    copy_stream_segment(&source, segment, &mut upload, "test source range").unwrap();
    assert_eq!(upload, [0, 0, 0, 12, 13, 14, 15, 16, 0]);

    assert!(matches!(
        copy_stream_segment(
            &source,
            GroupStreamSegment {
                input_end: 8,
                ..segment
            },
            &mut upload,
            "test source range",
        ),
        Err(VarDctDecodeError::EntropyWindowContract {
            detail: "test source range"
        })
    ));
}

#[test]
fn quant_matrix_scales_cover_all_wire_values_and_reject_out_of_range() {
    let expected = [1.5625, 1.25, 1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768];
    for (scale, expected) in expected.into_iter().enumerate() {
        assert_eq!(
            dequant_matrix_multiplier("X", scale as u32).unwrap(),
            expected
        );
    }
    assert!(matches!(
        dequant_matrix_multiplier("B", 8),
        Err(VarDctDecodeError::InvalidQuantMatrixScale {
            channel: "B",
            scale: 8
        })
    ));
}

#[test]
fn restoration_contract_rejects_invalid_epf_iterations() {
    let error = restoration_config(RestorationFilterInventory::Custom {
        gaborish: GaborishInventory::Disabled,
        epf: EdgePreservingFilterInventory::Enabled {
            iterations: 0,
            sharp_lut: None,
            weights: None,
            sigma: None,
            sigma_for_modular: None,
        },
    })
    .unwrap_err();
    assert!(matches!(
        error,
        VarDctDecodeError::InvalidEpfIterations { iterations: 0 }
    ));
}

#[test]
fn restoration_contract_preserves_disabled_and_standard_defaults() {
    let disabled = restoration_config(RestorationFilterInventory::Custom {
        gaborish: GaborishInventory::Disabled,
        epf: EdgePreservingFilterInventory::Disabled,
    })
    .unwrap();
    assert_eq!(disabled, (None, None));

    let (gaborish, epf) = restoration_config(RestorationFilterInventory::Default).unwrap();
    assert_eq!(gaborish, Some(ResidentGaborishWeights::DEFAULT));
    assert_eq!(
        epf,
        Some(VarDctEpfHeader {
            iterations: 2,
            sharp_lut: [
                0.0,
                1.0 / 7.0,
                2.0 / 7.0,
                3.0 / 7.0,
                4.0 / 7.0,
                5.0 / 7.0,
                6.0 / 7.0,
                1.0,
            ],
            channel_scale: [40.0, 5.0, 3.5],
            quant_mul: 0.46,
            pass0_sigma_scale: 0.9,
            pass2_sigma_scale: 6.5,
            border_sad_mul: 2.0 / 3.0,
        })
    );
}

#[test]
fn restoration_contract_preserves_custom_gaborish_and_epf_values() {
    let half = jxl_gpu_bitstream::FiniteF16::from_bits(0x3800).unwrap();
    let quarter = jxl_gpu_bitstream::FiniteF16::from_bits(0x3400).unwrap();
    let one = jxl_gpu_bitstream::FiniteF16::from_bits(0x3c00).unwrap();
    let two = jxl_gpu_bitstream::FiniteF16::from_bits(0x4000).unwrap();
    let zero = jxl_gpu_bitstream::FiniteF16::from_bits(0).unwrap();
    let weights = [[half, quarter], [quarter, zero], [zero, half]];
    let (gaborish, epf) = restoration_config(RestorationFilterInventory::Custom {
        gaborish: GaborishInventory::Custom { weights },
        epf: EdgePreservingFilterInventory::Enabled {
            iterations: 3,
            sharp_lut: Some([zero, quarter, half, one, zero, quarter, half, one]),
            weights: Some(jxl_gpu_bitstream::EpfWeightsInventory {
                channel_scale: [one, half, quarter],
                pass1_zeroflush: half,
                pass2_zeroflush: quarter,
            }),
            sigma: Some(jxl_gpu_bitstream::EpfSigmaInventory {
                quant_mul: Some(one),
                pass0_sigma_scale: half,
                pass2_sigma_scale: quarter,
                border_sad_mul: two,
            }),
            sigma_for_modular: None,
        },
    })
    .unwrap();
    assert_eq!(
        gaborish,
        Some(ResidentGaborishWeights {
            x: [0.5, 0.25],
            y: [0.25, 0.0],
            b: [0.0, 0.5],
        })
    );
    assert_eq!(
        epf,
        Some(VarDctEpfHeader {
            iterations: 3,
            sharp_lut: [0.0, 0.25, 0.5, 1.0, 0.0, 0.25, 0.5, 1.0],
            channel_scale: [1.0, 0.5, 0.25],
            quant_mul: 1.0,
            pass0_sigma_scale: 0.5,
            pass2_sigma_scale: 0.25,
            border_sad_mul: 2.0,
        })
    );
}
