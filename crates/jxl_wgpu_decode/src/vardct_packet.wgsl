/*__JXL_MODULAR_ENTROPY_ABI__*/

struct Params {
    entropy: EntropyStreamParams,
    window_logical_start: u32,
    window_upload_start: u32,
    stream_token_end: u32,
    window_yield_end: u32,
    window_flags: u32,
    entropy_state_offset: u32,
    stream_base_bit: u32,
    width: u32,
    height: u32,
    origin_x: u32,
    origin_y: u32,
    sample_count: u32,
    initialize_chroma: u32,
    source_channels: u32,
    source_bits: u32,
    source_mask: u32,
    needs_self_correcting: u32,
    output_kind: u32,
    transfer: u32,
    limited_range: u32,
    channels: u32,
    order: u32,
    bits: u32,
    storage_bits: u32,
    plane0_offset: u32,
    plane0_stride: u32,
    plane1_offset: u32,
    plane1_stride: u32,
    plane2_offset: u32,
    plane2_stride: u32,
    plane3_offset: u32,
    plane3_stride: u32,
    chroma_width: u32,
    chroma_height: u32,
    logical_size: u32,
    numeric_mapping: u32,
    status_index: u32,
    stream_index: u32,
    fixed_leaf_predictor: u32,
    fixed_leaf_offset: u32,
    fixed_leaf_multiplier: u32,
    fixed_leaf_cluster0: u32,
    fixed_leaf_cluster1: u32,
    fixed_leaf_cluster2: u32,
    fixed_leaf_cluster3: u32,
    wp_p1: u32,
    wp_p2: u32,
    wp_p3a: u32,
    wp_p3b: u32,
    wp_p3c: u32,
    wp_p3d: u32,
    wp_p3e: u32,
    wp_w0: u32,
    wp_w1: u32,
    wp_w2: u32,
    wp_w3: u32,
    _reserved: u32,
};

struct PacketControl {
    section_bits: vec4<u32>,
    geometry: vec4<u32>,
    offsets: vec4<u32>,
    capacities: vec4<u32>,
    expected: vec4<u32>,
    quantization: vec4<u32>,
    streams: vec4<u32>,
    scratch: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> codestream: array<u32>;
@group(0) @binding(1) var<storage, read> modular_metadata: array<u32>;
@group(0) @binding(2) var<storage, read_write> reconstructed: array<u32>;
@group(0) @binding(3) var<storage, read_write> raw_metadata: array<u32>;
@group(0) @binding(4) var<storage, read_write> coefficients: array<u32>;
@group(0) @binding(5) var<storage, read_write> status: array<u32>;
@group(0) @binding(6) var<uniform> control: PacketControl;
@group(0) @binding(7) var<storage, read> params_input: array<Params>;

var<private> bit_cursor: u32;
var<private> decode_error: u32;
var<private> current_channel: u32;
var<private> reconstruction_base: u32;
var<private> params: Params;
var<private> target_kind: u32;
var<private> target_offset: u32;
var<private> target_stride: u32;
var<private> consumer_decoded: u32;

const STATUS_OK: u32 = 1u;
const STATUS_IN_PROGRESS: u32 = 14u;
const ERROR_TRUNCATED_BITS: u32 = 2u;
const ERROR_PREFIX: u32 = 3u;
const ERROR_RAW_TOKEN: u32 = 4u;
const ERROR_LZ77_STATE: u32 = 5u;
const ERROR_LZ77_LENGTH: u32 = 6u;
const ERROR_TRAILING_BITS: u32 = 7u;
const ERROR_OUTPUT_BOUNDS: u32 = 8u;
const ERROR_OUTPUT_MAPPING: u32 = 9u;
const ERROR_ANS_STATE: u32 = 10u;
const ERROR_ENTROPY_CLUSTER: u32 = 11u;
const ERROR_MA_TREE: u32 = 12u;
const ERROR_PREDICTOR: u32 = 13u;
const ERROR_LF_HEADER: u32 = 20u;
const ERROR_FIRST_BLOCK: u32 = 21u;
const ERROR_HF_HEADER: u32 = 22u;
const ERROR_STRATEGY: u32 = 24u;
const ERROR_SHARPNESS: u32 = 25u;
const ERROR_HF_GLOBAL: u32 = 27u;
const STATUS_LF_READY: u32 = 30u;
const WINDOW_FIRST: u32 = 1u;
const WINDOW_FINAL: u32 = 2u;
const WINDOW_ENABLED: u32 = 4u;

fn reconstruction_load(index: u32) -> u32 {
    return reconstructed[reconstruction_base + index];
}

fn reconstruction_store(index: u32, value: u32) {
    reconstructed[reconstruction_base + index] = value;
}

fn entropy_window_base() -> u32 {
    return params.sample_count * params.source_channels
        + params.needs_self_correcting * 5u * control.scratch.x;
}

fn bit_mask(count: u32) -> u32 {
    if count == 0u { return 0u; }
    if count == 32u { return 0xffffffffu; }
    return (1u << count) - 1u;
}

fn bounded_window_enabled() -> bool {
    return (params.window_flags & WINDOW_ENABLED) != 0u;
}

fn physical_bit_cursor(logical_cursor: u32) -> u32 {
    if !bounded_window_enabled() {
        return logical_cursor;
    }
    return params.window_upload_start + logical_cursor - params.window_logical_start;
}

fn peek_bits(count: u32) -> u32 {
    if count == 0u { return 0u; }
    let physical = physical_bit_cursor(bit_cursor);
    let word_index = physical >> 5u;
    let word_shift = physical & 31u;
    var value = codestream[word_index] >> word_shift;
    if word_shift + count > 32u {
        value |= codestream[word_index + 1u] << (32u - word_shift);
    }
    return value & bit_mask(count);
}

fn read_bits(count: u32) -> u32 {
    if decode_error != 0u { return 0u; }
    if count > 32u
        || (bounded_window_enabled() && bit_cursor < params.window_logical_start)
        || bit_cursor > params.entropy.token_end
        || count > params.entropy.token_end - bit_cursor {
        decode_error = ERROR_TRUNCATED_BITS;
        return 0u;
    }
    let value = peek_bits(count);
    bit_cursor += count;
    return value;
}

/*__JXL_MODULAR_ENTROPY__*/

fn window_is_first() -> bool {
    return (params.window_flags & WINDOW_FIRST) != 0u;
}

fn window_is_final() -> bool {
    return (params.window_flags & WINDOW_FINAL) != 0u;
}

fn window_should_pause() -> bool {
    return bounded_window_enabled() && !window_is_final()
        && bit_cursor >= params.window_yield_end;
}

fn load_packet_execution_state() {
    let base = params.entropy_state_offset;
    bit_cursor = reconstruction_load(base);
    entropy_ans_state = reconstruction_load(base + 1u);
    entropy_copy_remaining = reconstruction_load(base + 2u);
    entropy_copy_position = reconstruction_load(base + 3u);
    entropy_decoded = reconstruction_load(base + 4u);
    entropy_last_value = reconstruction_load(base + 5u);
    consumer_decoded = reconstruction_load(base + 6u);
    decode_error = reconstruction_load(base + 7u);
    predictor_prev_grad = bitcast<i32>(reconstruction_load(base + 13u));
    if params.needs_self_correcting == 0u {
        return;
    }
    let channel_decoded = consumer_decoded % params.sample_count;
    wp_x = channel_decoded % params.width;
    wp_y = channel_decoded / params.width;
    wp_true_err_w = bitcast<i32>(reconstruction_load(base + 14u));
    wp_true_err_nw = bitcast<i32>(reconstruction_load(base + 15u));
    wp_true_err_n = bitcast<i32>(reconstruction_load(base + 16u));
    wp_true_err_ne = bitcast<i32>(reconstruction_load(base + 17u));
    for (var component = 0u; component < 4u; component += 1u) {
        wp_subpred_nw_ww[component] = reconstruction_load(base + 18u + component);
        wp_subpred_n_w[component] = reconstruction_load(base + 22u + component);
        wp_subpred_ne[component] = reconstruction_load(base + 26u + component);
    }
}

fn save_packet_execution_state(error_code: u32) {
    let base = params.entropy_state_offset;
    reconstruction_store(base, bit_cursor);
    reconstruction_store(base + 1u, entropy_ans_state);
    reconstruction_store(base + 2u, entropy_copy_remaining);
    reconstruction_store(base + 3u, entropy_copy_position);
    reconstruction_store(base + 4u, entropy_decoded);
    reconstruction_store(base + 5u, entropy_last_value);
    reconstruction_store(base + 6u, consumer_decoded);
    reconstruction_store(base + 7u, error_code);
    // Words 8..12 are reserved for the combined LF/HF phase state introduced by the next
    // packet-resume stage. Keeping them in this ABI avoids another layout transition.
    reconstruction_store(base + 13u, bitcast<u32>(predictor_prev_grad));
    if params.needs_self_correcting == 0u {
        return;
    }
    reconstruction_store(base + 14u, bitcast<u32>(wp_true_err_w));
    reconstruction_store(base + 15u, bitcast<u32>(wp_true_err_nw));
    reconstruction_store(base + 16u, bitcast<u32>(wp_true_err_n));
    reconstruction_store(base + 17u, bitcast<u32>(wp_true_err_ne));
    for (var component = 0u; component < 4u; component += 1u) {
        reconstruction_store(base + 18u + component, wp_subpred_nw_ww[component]);
        reconstruction_store(base + 22u + component, wp_subpred_n_w[component]);
        reconstruction_store(base + 26u + component, wp_subpred_ne[component]);
    }
}

fn unpack_signed(value: u32) -> i32 {
    if (value & 1u) == 0u { return i32(value >> 1u); }
    return bitcast<i32>(0u - ((value >> 1u) + 1u));
}

/*__JXL_MODULAR_RECONSTRUCT__*/

fn target_load(index: u32) -> i32 {
    let physical_index = (index / params.width) * target_stride + index % params.width;
    if target_kind == 0u {
        return bitcast<i32>(reconstructed[target_offset + physical_index]);
    }
    return bitcast<i32>(raw_metadata[target_offset + physical_index]);
}

fn target_store(index: u32, value: i32) {
    let physical_index = (index / params.width) * target_stride + index % params.width;
    if target_kind == 0u {
        reconstructed[target_offset + physical_index] = bitcast<u32>(value);
    } else {
        raw_metadata[target_offset + physical_index] = bitcast<u32>(value);
    }
}

fn decode_channel(
    width: u32,
    height: u32,
    stride: u32,
    offset: u32,
    kind: u32,
    start_decoded: u32,
) -> u32 {
    params.width = width;
    params.height = height;
    target_kind = kind;
    target_offset = offset;
    target_stride = stride;
    if start_decoded == 0u {
        predictor_prev_grad = 0i;
        if params.needs_self_correcting != 0u {
            wp_reset();
        }
    }
    var decoded = start_decoded;
    let channel_samples = width * height;
    while decoded < channel_samples && decode_error == 0u && !window_should_pause() {
        let x = decoded % width;
        let y = decoded / width;
        var w = 0i;
        if x != 0u {
            w = target_load(decoded - 1u);
        } else if y != 0u {
            w = target_load(decoded - width);
        }
        var n = w;
        var nw = w;
        if y != 0u {
            n = target_load(decoded - width);
            nw = n;
            if x != 0u { nw = target_load(decoded - width - 1u); }
        }
        var ne = n;
        if y != 0u && x + 1u < width { ne = target_load(decoded - width + 1u); }
        var nee = ne;
        if y != 0u && x + 2u < width { nee = target_load(decoded - width + 2u); }
        var nn = n;
        if y >= 2u { nn = target_load(decoded - 2u * width); }
        var ww = w;
        if x >= 2u { ww = target_load(decoded - 2u); }
        var weighted = WeightedPrediction(0i, 0i, array<i32, 4>(0i, 0i, 0i, 0i));
        if params.needs_self_correcting != 0u {
            weighted = weighted_predict(n, nw, ne, w, nn);
        }
        let leaf = ma_leaf(decoded, x, y, n, w, nw, ne, nn, ww, weighted.max_error);
        if decode_error != 0u { break; }
        let predictor = modular_metadata[leaf + 1u];
        let leaf_offset = modular_metadata[leaf + 2u];
        let cluster = modular_metadata[leaf + 3u];
        let multiplier = modular_metadata[leaf + 4u];
        let difference = unpack_signed(entropy_read_varint(cluster, width));
        let residual = bitcast<i32>(bitcast<u32>(difference) * multiplier + leaf_offset);
        let prediction = predictor_value(predictor, weighted, n, w, nw, ne, nn, ww, nee);
        let sample = bitcast<i32>(bitcast<u32>(residual) + bitcast<u32>(prediction));
        target_store(decoded, sample);
        if params.needs_self_correcting != 0u {
            weighted_record(weighted, sample);
        }
        predictor_prev_grad = select(w - nw + n, 0i, x + 1u == width);
        decoded += 1u;
    }
    return decoded;
}

fn reject(code: u32, value: u32) {
    if decode_error == 0u {
        decode_error = code;
        status[8] = value;
    }
}

fn validate_sharpness(block_count: u32) {
    for (var index = 0u; index < block_count && decode_error == 0u; index += 1u) {
        let sharpness = raw_metadata[control.expected.w + index];
        if sharpness > 7u {
            reject(ERROR_SHARPNESS, sharpness);
        }
    }
}

fn finish_section(expected_end: u32) {
    if decode_error != 0u { return; }
    if bit_cursor > expected_end {
        reject(ERROR_TRAILING_BITS, bit_cursor - expected_end);
        return;
    }
    let remaining = expected_end - bit_cursor;
    if remaining > 7u || peek_bits(remaining) != 0u {
        reject(ERROR_TRAILING_BITS, remaining);
        return;
    }
    bit_cursor = expected_end;
}

fn initialize_packet(start: u32, end: u32, stream_index: u32) {
    params = params_input[0];
    reconstruction_base = 0u;
    decode_error = 0u;
    bit_cursor = start;
    params.entropy.token_start = start;
    params.entropy.token_end = end;
    params.source_mask = 0x7fffffffu;
    params.entropy.lz77_window_mask = params_input[0].entropy.lz77_window_mask;
    params.stream_index = stream_index;
}

fn decode_lf_channels() -> u32 {
    let block_count = control.geometry.z * control.geometry.w;
    params.sample_count = block_count;
    params.source_channels = 3u;
    var decoded = 0u;
    entropy_begin();
    for (current_channel = 0u; current_channel < 3u && decode_error == 0u; current_channel += 1u) {
        decoded += decode_channel(
            control.geometry.z,
            control.geometry.w,
            control.geometry.z,
            current_channel * block_count,
            0u,
            0u,
        );
    }
    entropy_finalize();
    return decoded;
}

fn decode_lf_channels_bounded() -> u32 {
    let block_count = control.geometry.z * control.geometry.w;
    params.sample_count = block_count;
    params.source_channels = 3u;
    params.width = control.geometry.z;
    params.height = control.geometry.w;
    if window_is_first() {
        bit_cursor = params.entropy.token_start;
        consumer_decoded = 0u;
        decode_error = 0u;
        entropy_begin();
    } else {
        load_packet_execution_state();
    }
    current_channel = consumer_decoded / block_count;
    while current_channel < 3u && decode_error == 0u && !window_should_pause() {
        let channel_start = current_channel * block_count;
        let channel_decoded = consumer_decoded - channel_start;
        let decoded = decode_channel(
            control.geometry.z,
            control.geometry.w,
            control.geometry.z,
            current_channel * block_count,
            0u,
            channel_decoded,
        );
        consumer_decoded += decoded - channel_decoded;
        if decode_error != 0u || window_should_pause() {
            break;
        }
        if decoded != block_count {
            decode_error = ERROR_RAW_TOKEN;
            break;
        }
        current_channel += 1u;
    }
    return consumer_decoded;
}

fn finish_bounded_lf(lf_decoded: u32) -> u32 {
    let expected_samples = 3u * control.geometry.z * control.geometry.w;
    var status_code = decode_error;
    if decode_error != 0u {
        if !window_is_final() {
            save_packet_execution_state(decode_error);
        }
    } else if lf_decoded != expected_samples {
        if window_is_final() {
            status_code = ERROR_TRUNCATED_BITS;
        } else {
            save_packet_execution_state(0u);
            status_code = STATUS_IN_PROGRESS;
        }
    } else {
        entropy_finalize();
        status_code = decode_error;
        if !window_is_final() {
            save_packet_execution_state(decode_error);
            status_code = select(STATUS_IN_PROGRESS, decode_error, decode_error != 0u);
        } else if decode_error == 0u {
            status_code = STATUS_LF_READY;
        }
    }
    return status_code;
}

fn write_lf_status(status_code: u32, lf_decoded: u32) {
    status[0] = status_code;
    status[1] = params.stream_base_bit + bit_cursor;
    status[2] = params.stream_base_bit + params.stream_token_end;
    status[3] = lf_decoded;
    status[4] = 0u;
    status[7] = control.capacities.x;
    status[9] = control.quantization.x;
    status[10] = control.quantization.y;
    status[11] = 0u;
    status[12] = control.quantization.z;
}

fn decode_hf_channels(first_blocks: u32) -> u32 {
    let block_count = control.geometry.z * control.geometry.w;
    let correlation_width = (control.geometry.x + 63u) / 64u;
    let correlation_height = (control.geometry.y + 63u) / 64u;
    let correlation_samples = correlation_width * correlation_height;
    let hf_samples = 2u * correlation_samples + 2u * first_blocks + block_count;
    params.sample_count = max(3u * block_count, hf_samples);
    params.source_channels = 1u;
    var decoded = 0u;
    entropy_begin();
    current_channel = 0u;
    decoded += decode_channel(
        correlation_width,
        correlation_height,
        correlation_width,
        control.offsets.x,
        1u,
        0u,
    );
    current_channel = 1u;
    decoded += decode_channel(
        correlation_width,
        correlation_height,
        correlation_width,
        control.offsets.y,
        1u,
        0u,
    );
    current_channel = 2u;
    decoded += decode_channel(
        first_blocks,
        2u,
        control.capacities.w,
        control.offsets.z,
        1u,
        0u,
    );
    current_channel = 3u;
    decoded += decode_channel(
        control.geometry.z,
        control.geometry.w,
        control.geometry.z,
        control.expected.w,
        1u,
        0u,
    );
    entropy_finalize();
    return decoded;
}

fn validate_hf_values(first_blocks: u32) {
    let block_count = control.geometry.z * control.geometry.w;
    if control.expected.y != 0u {
        for (var index = 0u; index < first_blocks && decode_error == 0u; index += 1u) {
            if raw_metadata[control.offsets.z + index] != control.expected.x {
                reject(ERROR_STRATEGY, raw_metadata[control.offsets.z + index]);
            }
        }
    }
    validate_sharpness(block_count);
}

fn finish_hf_packet() {
    if control.streams.z != 0u {
        finish_section(control.section_bits.y);
        bit_cursor = control.section_bits.w;
        params.entropy.token_start = bit_cursor;
        params.entropy.token_end = control.section_bits.w;
    } else {
        let preset_bits = select(
            0u,
            32u - countLeadingZeros(control.streams.w - 1u),
            control.streams.w > 1u,
        );
        let default_matrix = read_bits(1u);
        let preset = read_bits(preset_bits);
        let fixed_hf_tail = read_bits(17u);
        if default_matrix != (control.expected.z & 1u)
            || preset != 0u
            || fixed_hf_tail != (control.expected.z >> 1u) {
            reject(ERROR_HF_GLOBAL, bit_cursor);
        }
        finish_section(params.entropy.token_end);
    }
}

fn clear_coefficients() {
    if decode_error == 0u {
        for (var index = 0u; index < control.capacities.x; index += 1u) {
            coefficients[index] = 0u;
        }
    }
}

@compute @workgroup_size(1, 1, 1)
fn decode_vardct_lf() {
    params = params_input[0];
    reconstruction_base = 0u;
    if bounded_window_enabled() {
        params.source_mask = 0x7fffffffu;
        params.stream_index = control.streams.x;
        let lf_decoded = decode_lf_channels_bounded();
        write_lf_status(finish_bounded_lf(lf_decoded), lf_decoded);
        return;
    }
    initialize_packet(control.section_bits.x, control.section_bits.y, control.streams.x);
    let lf_decoded = decode_lf_channels();
    status[0] = select(STATUS_LF_READY, decode_error, decode_error != 0u);
    status[1] = bit_cursor;
    status[2] = control.section_bits.y;
    status[3] = lf_decoded;
    status[4] = 0u;
    status[7] = control.capacities.x;
    status[9] = control.quantization.x;
    status[10] = control.quantization.y;
    status[11] = 0u;
    status[12] = control.quantization.z;
}

@compute @workgroup_size(1, 1, 1)
fn decode_vardct_hf() {
    initialize_packet(control.section_bits.z, control.section_bits.y, control.streams.y);
    let first_blocks = control.quantization.w;
    if first_blocks == 0u || first_blocks > control.capacities.w {
        reject(ERROR_FIRST_BLOCK, first_blocks);
    }
    let hf_decoded = decode_hf_channels(first_blocks);
    validate_hf_values(first_blocks);
    finish_hf_packet();
    clear_coefficients();
    status[0] = select(STATUS_OK, decode_error, decode_error != 0u);
    status[1] = bit_cursor;
    status[2] = params.entropy.token_end;
    status[4] = hf_decoded;
    status[5] = raw_metadata[control.offsets.z];
    status[6] = raw_metadata[control.offsets.w] + 1u;
    status[7] = control.capacities.x;
    status[8] = select(status[8], 0u, decode_error == 0u);
    status[9] = control.quantization.x;
    status[10] = control.quantization.y;
    status[11] = first_blocks;
}

@compute @workgroup_size(1, 1, 1)
fn decode_vardct_packet() {
    initialize_packet(control.section_bits.x, control.section_bits.y, control.streams.x);
    params.source_channels = 3u;

    let extra_precision = read_bits(2u);
    if read_bits(4u) != 3u {
        reject(ERROR_LF_HEADER, bit_cursor);
    }
    let block_count = control.geometry.z * control.geometry.w;
    let lf_decoded = decode_lf_channels();

    let first_block_bits = control.capacities.z;
    let first_blocks = read_bits(first_block_bits) + 1u;
    if first_blocks > control.capacities.w {
        reject(ERROR_FIRST_BLOCK, first_blocks);
    }
    if read_bits(4u) != 3u {
        reject(ERROR_HF_HEADER, bit_cursor);
    }

    params.stream_index = control.streams.y;
    let hf_decoded = decode_hf_channels(first_blocks);
    validate_hf_values(first_blocks);
    finish_hf_packet();
    clear_coefficients();
    status[0] = select(STATUS_OK, decode_error, decode_error != 0u);
    status[1] = bit_cursor;
    status[2] = params.entropy.token_end;
    status[3] = lf_decoded;
    status[4] = hf_decoded;
    status[5] = raw_metadata[control.offsets.z];
    status[6] = raw_metadata[control.offsets.w] + 1u;
    status[7] = control.capacities.x;
    status[9] = control.quantization.x;
    status[10] = control.quantization.y;
    status[11] = first_blocks;
    status[12] = extra_precision;
}

@compute @workgroup_size(1, 1, 1)
fn validate_vardct_sharpness() {
    decode_error = 0u;
    validate_sharpness(control.geometry.z * control.geometry.w);
    status[0] = select(STATUS_OK, decode_error, decode_error != 0u);
}
