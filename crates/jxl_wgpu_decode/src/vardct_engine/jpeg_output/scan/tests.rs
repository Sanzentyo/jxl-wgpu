use super::*;
use jxl_gpu_bitstream::jpeg_reconstruction::ScanComponent;

fn scan(start: u8, end: u8, high: u8, low: u8) -> Scan {
    Scan {
        start,
        end,
        high,
        low,
        components: vec![ScanComponent {
            component: 0,
            dc: 0,
            ac: 0,
        }],
        last_pass: 0,
        resets: Vec::new(),
        extra_zeros: Vec::new(),
    }
}

#[test]
fn progression_rejects_ac_before_dc_duplicate_and_missing_refinement_predecessors() {
    let mut state = [[0; 64]; 3];
    assert!(update_progression(&scan(1, 63, 0, 0), &mut state).is_err());
    assert!(update_progression(&scan(0, 0, 1, 0), &mut state).is_err());
    update_progression(&scan(0, 0, 0, 2), &mut state).unwrap();
    assert!(update_progression(&scan(0, 0, 0, 2), &mut state).is_err());
    assert!(update_progression(&scan(0, 0, 1, 0), &mut state).is_err());
    update_progression(&scan(0, 0, 2, 1), &mut state).unwrap();
    update_progression(&scan(0, 0, 1, 0), &mut state).unwrap();
    update_progression(&scan(1, 63, 0, 1), &mut state).unwrap();
    update_progression(&scan(1, 63, 1, 0), &mut state).unwrap();
    assert_eq!(state[0], [65535; 64]);
}

#[test]
fn spectral_component_reset_and_zrl_contracts_precede_task_allocation() {
    let plane = JpegCoefficientPlane {
        coefficient_word_offset: 192,
        quantization_word_offset: 64,
        blocks_per_row: 2,
        block_rows: 2,
        real_blocks: [2, 2],
        sampling: [1, 1],
        channel: 1,
    };
    let geometry = Geometry {
        planes: &[plane],
        components: 1,
        mcus: [2, 2],
    };
    assert_eq!(
        scan_extent(&scan(0, 63, 0, 0), &geometry, false).unwrap(),
        ([2, 2], 4)
    );
    assert!(scan_extent(&scan(0, 63, 0, 0), &geometry, true).is_err());
    for invalid in [
        scan(1, 63, 0, 0),
        scan(0, 1, 0, 0),
        scan(0, 0, 3, 1),
        scan(0, 0, 0, 14),
    ] {
        assert!(scan_extent(&invalid, &geometry, false).is_err());
    }
    let mut duplicate = scan(0, 0, 0, 0);
    duplicate.components.push(duplicate.components[0]);
    assert!(scan_extent(&duplicate, &geometry, true).is_err());
    for resets in [vec![4], vec![1, 1], vec![2, 1]] {
        let mut invalid = scan(0, 0, 0, 0);
        invalid.resets = resets;
        assert!(scan_extent(&invalid, &geometry, true).is_err());
    }
    for extra in [
        vec![(4, 1)],
        vec![(0, 0)],
        vec![(0, 5)],
        vec![(1, 1), (1, 1)],
    ] {
        let mut invalid = scan(1, 63, 0, 0);
        invalid.extra_zeros = extra;
        assert!(scan_extent(&invalid, &geometry, true).is_err());
    }
    let mut refinement = scan(1, 63, 1, 0);
    refinement.extra_zeros.push((0, 1));
    assert!(matches!(
        scan_extent(&refinement, &geometry, true),
        Err(Error::Unsupported(_))
    ));
}
