use jxl_gpu_bitstream::{CodestreamStreamError, ContainerStreamScanner, InventoryError};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{Error, GpuDecoder, GpuOutputRequest};
use std::sync::Arc;

pub(crate) fn reject(backend: &WgpuBackend, data: &[u8], name: &str) {
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let request = GpuOutputRequest::color(jxl_wgpu_decode::vardct_rgb8_format()).unwrap();
    assert!(
        matches!(
            decoder.open(data, request.clone()),
            Err(Error::CodestreamInventory(
                InventoryError::XybIccReference { slot: 1 }
            ))
        ),
        "{name}: contiguous input"
    );
    for chunk_size in [1, 43, data.len()] {
        let mut transport = ContainerStreamScanner::new(Default::default());
        let mut stream = decoder.stream(request.clone()).unwrap();
        let mut rejected = false;
        'chunks: for chunk in data.chunks(chunk_size) {
            for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
                if let Err(error) = stream.push_transport_event(&event) {
                    assert!(
                        matches!(
                            error,
                            Error::CodestreamStream(CodestreamStreamError::Inventory(
                                InventoryError::XybIccReference { slot: 1 }
                            ))
                        ),
                        "{name} chunks={chunk_size}: {error}"
                    );
                    assert!(matches!(
                        stream.push_transport_event(&event),
                        Err(Error::IncrementalInputPoisoned)
                    ));
                    rejected = true;
                    break 'chunks;
                }
            }
        }
        assert!(
            rejected,
            "{name}: header rejection must precede end of input"
        );
        assert!(!stream.is_ready());
        let stats = stream.stats();
        assert_eq!(stats.codestream.frames_started, 0);
        assert_eq!(stats.codestream.section_bytes_emitted, 0);
        assert_eq!(stats.completed_frames, 0);
        assert_eq!(stats.retained_codestream_bytes, 0);
        assert_eq!(stats.retained_spans, 0);
        assert_eq!(stats.input_budget.reserved_bytes, 0);
        assert_eq!(stats.input_budget.reserved_spans, 0);
        assert_eq!(
            backend.transient_memory_budget().snapshot().reserved_bytes,
            0
        );
    }
}

#[test]
fn constructed_inventories_cannot_bypass_reference_colour_validation() {
    use jxl_wgpu_decode::{FrameExecutionPlan, FramePlanError};
    for case in super::corpus::cases().filter(|case| case.xyb) {
        let mut inventory = super::inventory(&case.bytes());
        let mut first = inventory.frames[0].clone();
        first.is_last = false;
        first.save_as_reference = 1;
        first.save_before_color_transform = false;
        inventory.frames.insert(0, first);
        inventory.frames[1].frame_index = 1;
        assert!(matches!(
            FrameExecutionPlan::negotiate(&inventory),
            Err(FramePlanError::InvalidHeader {
                frame_index: 0,
                source: InventoryError::XybIccReference { slot: 1 }
            })
        ));
        inventory.frames[0].save_before_color_transform = true;
        let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
        assert_eq!(plan.nodes[0].save_reference, Some(1));
        assert!(plan.nodes[1].references[1].unwrap().before_color_transform);
    }
}
