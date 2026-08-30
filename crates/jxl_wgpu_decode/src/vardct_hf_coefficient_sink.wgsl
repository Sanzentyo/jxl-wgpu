// This fragment intentionally occupies bind group 1.  The generic entropy implementation owns
// bind group 0, so the decoder can concatenate this source without textual binding replacement.
struct HfOrderDescriptor {
    offset: u32,
    len: u32,
    width: u32,
    height: u32,
};

struct HfCoefficientSinkParams {
    task_metadata_offset_words: u32,
    task_count: u32,
    coefficient_words: u32,
    order_descriptor_count: u32,
};

@group(1) @binding(0) var<storage, read> hf_artifact: array<u32>;
@group(1) @binding(1) var<storage, read> hf_order_descriptors: array<HfOrderDescriptor>;
@group(1) @binding(2) var<storage, read> hf_order_coordinates: array<u32>;
@group(1) @binding(3) var<storage, read_write> hf_coefficients: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> hf_coefficient_status: atomic<u32>;
@group(1) @binding(5) var<uniform> hf_sink_params: HfCoefficientSinkParams;

const HF_SINK_ERROR_TASK: u32 = 1u;
const HF_SINK_ERROR_CHANNEL: u32 = 2u;
const HF_SINK_ERROR_ORDER_DESCRIPTOR: u32 = 3u;
const HF_SINK_ERROR_ORDER_INDEX: u32 = 4u;
const HF_SINK_ERROR_COORDINATE: u32 = 5u;
const HF_SINK_ERROR_COEFFICIENT: u32 = 6u;

fn hf_sink_fail(code: u32) {
    atomicMax(&hf_coefficient_status, code);
}

// Stores one already entropy-decoded signed coefficient. `order_index` is the absolute index in
// the JPEG XL order, including the LF prefix skipped by the HF symbol loop.  The order table may
// contain either the natural order or the entropy-decoded custom permutation.
fn hf_store_quantized_coefficient(
    task_index: u32,
    channel: u32,
    order_index: u32,
    value: i32,
) -> bool {
    if (task_index >= hf_sink_params.task_count) {
        hf_sink_fail(HF_SINK_ERROR_TASK);
        return false;
    }
    if (channel >= 3u) {
        hf_sink_fail(HF_SINK_ERROR_CHANNEL);
        return false;
    }
    let metadata = hf_sink_params.task_metadata_offset_words + task_index * 12u;
    let block_width = hf_artifact[metadata + 4u];
    let block_height = hf_artifact[metadata + 5u];
    let coefficient_offset = hf_artifact[metadata + 8u];
    let coefficient_words = hf_artifact[metadata + 9u];
    let order_id = hf_artifact[metadata + 10u];
    let flags = hf_artifact[metadata + 11u];
    let descriptor_index = order_id * 3u + channel;
    if (descriptor_index >= hf_sink_params.order_descriptor_count) {
        hf_sink_fail(HF_SINK_ERROR_ORDER_DESCRIPTOR);
        return false;
    }
    let descriptor = hf_order_descriptors[descriptor_index];
    let area = block_width * block_height * 64u;
    if (descriptor.len != area || descriptor.width * descriptor.height != area
        || order_index >= descriptor.len) {
        hf_sink_fail(HF_SINK_ERROR_ORDER_INDEX);
        return false;
    }
    let packed_coordinate = hf_order_coordinates[descriptor.offset + order_index];
    var frequency_x = packed_coordinate & 0xffffu;
    var frequency_y = packed_coordinate >> 16u;
    if ((flags & 1u) != 0u) {
        let swap = frequency_x;
        frequency_x = frequency_y;
        frequency_y = swap;
    }
    let width = block_width * 8u;
    let height = block_height * 8u;
    if (frequency_x >= width || frequency_y >= height) {
        hf_sink_fail(HF_SINK_ERROR_COORDINATE);
        return false;
    }
    let packed_index = select(
        frequency_x * height + frequency_y,
        frequency_y * width + frequency_x,
        height < width,
    );
    let destination = coefficient_offset + channel * area + packed_index;
    if (destination >= hf_sink_params.coefficient_words
        || destination >= coefficient_offset + coefficient_words) {
        hf_sink_fail(HF_SINK_ERROR_COEFFICIENT);
        return false;
    }
    atomicAdd(&hf_coefficients[destination], value);
    return true;
}
