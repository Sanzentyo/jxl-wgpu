override wg_x: u32 = 64u;
override wg_y: u32 = 1u;

/*__JXL_MODULAR_ENTROPY_ABI__*/

struct Params {
    entropy: EntropyStreamParams,
    window_logical_start: u32,
    window_upload_start: u32,
    stream_token_end: u32,
    window_yield_end: u32,
    window_flags: u32,
    entropy_state_offset: u32,
    width: u32,
    height: u32,
    origin_x: u32,
    origin_y: u32,
    sample_count: u32,
    initialize_chroma: u32,
    source_channels: u32,
    channel_layout_offset: u32,
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
    fixed_output_mode: u32,
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
};

struct DispatchControl {
    first_group: u32,
    group_count: u32,
    lane_stride_words: u32,
    _padding: u32,
};

@group(0) @binding(0) var<storage, read> codestream: array<u32>;
@group(0) @binding(1) var<storage, read> modular_metadata: array<u32>;
@group(0) @binding(2) var<storage, read_write> reconstructed: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_words: /*__JXL_OUTPUT_WORDS_TYPE__*/;
@group(0) @binding(4) var<storage, read_write> status: array<u32>;
@group(0) @binding(5) var<storage, read> params_table: array<Params>;
/*__JXL_F64_BINDING__*/
@group(0) @binding(7) var<uniform> dispatch_control: DispatchControl;

var<private> bit_cursor: u32;
var<private> decode_error: u32;
var<private> current_channel: u32;
var<private> consumer_decoded: u32;
var<private> reconstruction_base: u32;
var<private> params: Params;

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

const WINDOW_FIRST: u32 = 1u;
const WINDOW_FINAL: u32 = 2u;

const FIXED_OUTPUT_DIRECT_NORMALIZED_GRAY8: u32 = 1u;
const FIXED_OUTPUT_COMPACT_NORMALIZED_GRAY8: u32 = 2u;

fn reconstruction_load(index: u32) -> u32 {
    return reconstructed[reconstruction_base + index];
}

fn reconstruction_store(index: u32, value: u32) {
    reconstructed[reconstruction_base + index] = value;
}

fn entropy_window_base() -> u32 {
    return params.sample_count * params.source_channels
        + params.needs_self_correcting * 5u * params.width;
}

fn bit_mask(count: u32) -> u32 {
    if count == 0u {
        return 0u;
    }
    if count == 32u {
        return 0xffffffffu;
    }
    return (1u << count) - 1u;
}

fn peek_bits(count: u32) -> u32 {
    if count == 0u {
        return 0u;
    }
    if bit_cursor < params.window_logical_start {
        decode_error = ERROR_TRUNCATED_BITS;
        return 0u;
    }
    let relative_cursor = bit_cursor - params.window_logical_start;
    if relative_cursor > 0xffffffffu - params.window_upload_start {
        decode_error = ERROR_TRUNCATED_BITS;
        return 0u;
    }
    let physical_cursor = params.window_upload_start + relative_cursor;
    let word_index = physical_cursor >> 5u;
    let word_shift = physical_cursor & 31u;
    var value = codestream[word_index] >> word_shift;
    if word_shift + count > 32u {
        value = value | (codestream[word_index + 1u] << (32u - word_shift));
    }
    return value & bit_mask(count);
}

fn read_bits(count: u32) -> u32 {
    if decode_error != 0u {
        return 0u;
    }
    if count > 32u || bit_cursor < params.window_logical_start
        || bit_cursor > params.entropy.token_end
        || count > params.entropy.token_end - bit_cursor {
        decode_error = ERROR_TRUNCATED_BITS;
        return 0u;
    }
    let value = peek_bits(count);
    bit_cursor = bit_cursor + count;
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
    return !window_is_final() && bit_cursor >= params.window_yield_end;
}

fn load_entropy_execution_state() {
    let base = params.entropy_state_offset;
    bit_cursor = reconstruction_load(base);
    entropy_ans_state = reconstruction_load(base + 1u);
    entropy_copy_remaining = reconstruction_load(base + 2u);
    entropy_copy_position = reconstruction_load(base + 3u);
    entropy_decoded = reconstruction_load(base + 4u);
    entropy_last_value = reconstruction_load(base + 5u);
    consumer_decoded = reconstruction_load(base + 6u);
    decode_error = reconstruction_load(base + 7u);
    load_modular_execution_state();
}

fn save_entropy_execution_state(error_code: u32) {
    let base = params.entropy_state_offset;
    reconstruction_store(base, bit_cursor);
    reconstruction_store(base + 1u, entropy_ans_state);
    reconstruction_store(base + 2u, entropy_copy_remaining);
    reconstruction_store(base + 3u, entropy_copy_position);
    reconstruction_store(base + 4u, entropy_decoded);
    reconstruction_store(base + 5u, entropy_last_value);
    reconstruction_store(base + 6u, consumer_decoded);
    reconstruction_store(base + 7u, error_code);
    save_modular_execution_state();
}

fn unpack_signed(value: u32) -> i32 {
    if (value & 1u) == 0u {
        return i32(value >> 1u);
    }
    let magnitude = (value >> 1u) + 1u;
    return bitcast<i32>(0u - magnitude);
}

fn write_byte(offset: u32, value: u32) {
    if offset >= params.logical_size {
        decode_error = ERROR_OUTPUT_BOUNDS;
        return;
    }
    let word_index = offset >> 2u;
    let shift = (offset & 3u) << 3u;
    let mask = 0xffu << shift;
    /*__JXL_WRITE_BYTE_WORD__*/
}

fn write_word(offset: u32, value: u32) {
    if (offset & 3u) != 0u || offset > params.logical_size {
        decode_error = ERROR_OUTPUT_BOUNDS;
        return;
    }
    if params.logical_size - offset < 4u {
        decode_error = ERROR_OUTPUT_BOUNDS;
        return;
    }
    /*__JXL_WRITE_FULL_WORD__*/
}

fn write_stored_code(offset: u32, code: u32) {
    let stored = code << (params.storage_bits - params.bits);
    if params.storage_bits == 8u {
        write_byte(offset, stored);
    } else if params.storage_bits == 16u {
        write_byte(offset, stored);
        write_byte(offset + 1u, stored >> 8u);
    } else {
        decode_error = ERROR_OUTPUT_BOUNDS;
    }
}

fn write_native_code(offset: u32, code: i32) {
    if code < 0i || u32(code) > params.source_mask {
        decode_error = ERROR_OUTPUT_MAPPING;
        return;
    }
    let value = u32(code);
    if params.storage_bits == 8u {
        write_byte(offset, value);
    } else if params.storage_bits == 16u {
        write_byte(offset, value);
        write_byte(offset + 1u, value >> 8u);
    } else {
        decode_error = ERROR_OUTPUT_MAPPING;
    }
}

fn write_native_pixel(x: u32, y: u32, index: u32) {
    if params.output_kind != 9u || params.channels != params.source_channels
        || params.bits != params.source_bits {
        decode_error = ERROR_OUTPUT_MAPPING;
        return;
    }
    let bytes_per_component = params.storage_bits / 8u;
    let pixel_offset = params.plane0_offset
        + y * params.plane0_stride
        + x * params.channels * bytes_per_component;
    if params.source_channels == 1u {
        write_native_code(pixel_offset, bitcast<i32>(reconstruction_load(index)));
        return;
    }

    let y_value = bitcast<i32>(reconstruction_load(index));
    let co = bitcast<i32>(reconstruction_load(params.sample_count + index));
    let cg = bitcast<i32>(reconstruction_load(2u * params.sample_count + index));
    let temporary = y_value - (cg >> 1u);
    let green = cg + temporary;
    let blue = temporary - (co >> 1u);
    let red = co + blue;
    write_native_code(pixel_offset, red);
    write_native_code(pixel_offset + bytes_per_component, green);
    write_native_code(pixel_offset + 2u * bytes_per_component, blue);
    if params.source_channels == 4u {
        let alpha = bitcast<i32>(reconstruction_load(3u * params.sample_count + index));
        write_native_code(pixel_offset + 3u * bytes_per_component, alpha);
    }
}

fn normalized_unsigned(sample: u32, bits: u32) -> u32 {
    if bits == 8u {
        return sample;
    }
    if bits == 16u {
        return sample * 257u;
    }
    if bits == 32u {
        return sample * 16843009u;
    }
    decode_error = ERROR_OUTPUT_MAPPING;
    return 0u;
}

fn normalized_signed_nonnegative(sample: u32, bits: u32) -> u32 {
    // MAX = quotient * 255 + 127 for each supported signed destination. Splitting the
    // multiplication this way avoids u32 overflow for the S32 full-range endpoint.
    let remainder = (sample * 127u + 127u) / 255u;
    if bits == 8u {
        return remainder;
    }
    if bits == 16u {
        return sample * 128u + remainder;
    }
    if bits == 32u {
        return sample * 8421504u + remainder;
    }
    decode_error = ERROR_OUTPUT_MAPPING;
    return 0u;
}

fn normalized_gray8_f32_bits(sample: u32) -> u32 {
    if sample == 0u {
        return 0u;
    }
    if sample == 255u {
        return 0x3f800000u;
    }
    var leading_bit = 0u;
    var remaining = sample;
    while remaining > 1u {
        remaining = remaining >> 1u;
        leading_bit += 1u;
    }
    // For sample in [1, 254], floor(log2(sample / 255)) is leading_bit - 8. Form the
    // 24-bit significand with integer division, rounded to nearest; the odd denominator means
    // there can be no exact halfway case. This avoids backend-dependent relaxed f32 division.
    let numerator = sample << (31u - leading_bit);
    let significand = (numerator + 127u) / 255u;
    let biased_exponent = leading_bit + 119u;
    return (biased_exponent << 23u) | (significand & 0x007fffffu);
}

fn widen_normalized_f32_to_f64_words(bits: u32) -> vec2<u32> {
    // Normalized Gray8 values are either +0 or finite, positive, normal f32 values. Widening a
    // normal f32 into binary64 is exact: rebias the exponent and move the 23 fraction bits to the
    // most-significant end of binary64's 52-bit fraction. The returned order is little-endian.
    if bits == 0u {
        return vec2<u32>(0u, 0u);
    }
    let exponent = (bits >> 23u) & 0xffu;
    let fraction = bits & 0x007fffffu;
    let low = (fraction & 0x7u) << 29u;
    let high = ((exponent + 896u) << 20u) | (fraction >> 3u);
    return vec2<u32>(low, high);
}

fn write_numeric_sample(x: u32, y: u32, sample: u32) {
    if params.numeric_mapping == 0u {
        decode_error = ERROR_OUTPUT_MAPPING;
        return;
    }
    let bytes_per_component = params.bits / 8u;
    let pixel_offset = params.plane0_offset
        + y * params.plane0_stride
        + x * params.channels * bytes_per_component;
    for (var component = 0u; component < params.channels; component += 1u) {
        let offset = pixel_offset + component * bytes_per_component;
        if params.output_kind == 0u {
            let value = normalized_unsigned(sample, params.bits);
            if params.bits == 8u {
                write_byte(offset, value);
            } else if params.bits == 16u {
                write_byte(offset, value);
                write_byte(offset + 1u, value >> 8u);
            } else if params.bits == 32u {
                write_word(offset, value);
            } else {
                decode_error = ERROR_OUTPUT_MAPPING;
            }
        } else if params.output_kind == 7u {
            let value = normalized_signed_nonnegative(sample, params.bits);
            if params.bits == 8u {
                write_byte(offset, value);
            } else if params.bits == 16u {
                write_byte(offset, value);
                write_byte(offset + 1u, value >> 8u);
            } else if params.bits == 32u {
                write_word(offset, value);
            } else {
                decode_error = ERROR_OUTPUT_MAPPING;
            }
        } else if params.output_kind == 8u {
            let normalized_bits = normalized_gray8_f32_bits(sample);
            if params.bits == 32u {
                write_word(offset, normalized_bits);
            } else if params.bits == 64u {
                /*__JXL_F64_OUTPUT__*/
            } else {
                decode_error = ERROR_OUTPUT_MAPPING;
            }
        } else {
            decode_error = ERROR_OUTPUT_MAPPING;
        }
    }
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn target_nonlinear(value: u32) -> f32 {
    let encoded = f32(value) / 255.0;
    if params.transfer == 0u {
        return encoded;
    }
    let linear = srgb_to_linear(encoded);
    if params.transfer == 2u {
        return linear;
    }
    if linear < 0.018 {
        return 4.5 * linear;
    }
    return 1.099 * pow(linear, 0.45) - 0.099;
}

fn color_code(value: u32) -> u32 {
    let nonlinear = clamp(target_nonlinear(value), 0.0, 1.0);
    let maximum = f32((1u << params.bits) - 1u);
    var code = maximum * nonlinear;
    if params.limited_range != 0u {
        let scale = f32(1u << (params.bits - 8u));
        code = scale * (16.0 + 219.0 * nonlinear);
    }
    return u32(clamp(round(code), 0.0, maximum));
}

fn neutral_chroma_code() -> u32 {
    return 1u << (params.bits - 1u);
}

fn rgb_code(value: u32) -> u32 {
    return u32(clamp(round(255.0 * target_nonlinear(value)), 0.0, 255.0));
}

fn stored_rgb_code(position: u32, gray: u32) -> u32 {
    var canonical = position;
    if (params.order == 1u || params.order == 3u) && position < 3u {
        canonical = 2u - position;
    }
    if canonical == 3u {
        return 255u;
    }
    return gray;
}

fn write_output_sample(x: u32, y: u32, sample: u32) {
    if params.output_kind == 0u || params.output_kind == 7u || params.output_kind == 8u {
        write_numeric_sample(x, y, sample);
        return;
    }
    if params.output_kind == 1u || params.output_kind == 2u || params.output_kind == 3u {
        let bytes_per_sample = params.storage_bits / 8u;
        let offset = params.plane0_offset + y * params.plane0_stride + x * bytes_per_sample;
        write_stored_code(offset, color_code(sample));
        return;
    }
    if params.output_kind == 5u {
        let gray = rgb_code(sample);
        let offset = params.plane0_offset + y * params.plane0_stride + x * params.channels;
        if params.channels == 4u {
            let packed = stored_rgb_code(0u, gray)
                | (stored_rgb_code(1u, gray) << 8u)
                | (stored_rgb_code(2u, gray) << 16u)
                | (stored_rgb_code(3u, gray) << 24u);
            write_word(offset, packed);
        } else {
            for (var position = 0u; position < params.channels; position += 1u) {
                write_byte(offset + position, stored_rgb_code(position, gray));
            }
        }
        return;
    }
    if params.output_kind == 6u {
        let gray = rgb_code(sample);
        write_byte(params.plane0_offset + y * params.plane0_stride + x, stored_rgb_code(0u, gray));
        write_byte(params.plane1_offset + y * params.plane1_stride + x, stored_rgb_code(1u, gray));
        write_byte(params.plane2_offset + y * params.plane2_stride + x, stored_rgb_code(2u, gray));
        if params.channels == 4u {
            write_byte(params.plane3_offset + y * params.plane3_stride + x, stored_rgb_code(3u, gray));
        }
    }
}

fn finalize_output() {
    if params.source_channels != 1u || params.output_kind == 9u {
        if params.output_kind != 9u {
            decode_error = ERROR_OUTPUT_MAPPING;
            return;
        }
        for (var index = 0u; index < params.sample_count; index += 1u) {
            let x = params.origin_x + index % params.width;
            let y = params.origin_y + index / params.width;
            write_native_pixel(x, y, index);
        }
        return;
    }

    if params.output_kind != 4u {
        for (var index = 0u; index < params.sample_count; index += 1u) {
            let x = params.origin_x + index % params.width;
            let y = params.origin_y + index / params.width;
            write_output_sample(x, y, reconstruction_load(index));
        }
    }
    if params.output_kind == 2u && params.initialize_chroma != 0u {
        let bytes_per_sample = params.storage_bits / 8u;
        let neutral = neutral_chroma_code();
        for (var y = 0u; y < params.chroma_height; y = y + 1u) {
            for (var x = 0u; x < params.chroma_width; x = x + 1u) {
                let offset = params.plane1_offset + y * params.plane1_stride + x * 2u * bytes_per_sample;
                write_stored_code(offset, neutral);
                write_stored_code(offset + bytes_per_sample, neutral);
            }
        }
    } else if params.output_kind == 3u && params.initialize_chroma != 0u {
        let bytes_per_sample = params.storage_bits / 8u;
        let neutral = neutral_chroma_code();
        for (var y = 0u; y < params.chroma_height; y = y + 1u) {
            for (var x = 0u; x < params.chroma_width; x = x + 1u) {
                write_stored_code(
                    params.plane1_offset + y * params.plane1_stride + x * bytes_per_sample,
                    neutral,
                );
                write_stored_code(
                    params.plane2_offset + y * params.plane2_stride + x * bytes_per_sample,
                    neutral,
                );
            }
        }
    } else if params.output_kind == 4u {
        let neutral = neutral_chroma_code();
        let pair_count = (params.width + 1u) / 2u;
        for (var y = 0u; y < params.height; y += 1u) {
            for (var pair = 0u; pair < pair_count; pair += 1u) {
                let x0 = pair * 2u;
                let x1 = min(x0 + 1u, params.width - 1u);
                let y0 = color_code(reconstruction_load(y * params.width + x0));
                let y1 = color_code(reconstruction_load(y * params.width + x1));
                var packed = y0 | (neutral << 8u) | (y1 << 16u) | (neutral << 24u);
                if params.order == 1u {
                    packed = neutral | (y0 << 8u) | (neutral << 16u) | (y1 << 24u);
                }
                let output_y = params.origin_y + y;
                let output_pair = params.origin_x / 2u + pair;
                write_word(
                    params.plane0_offset + output_y * params.plane0_stride + output_pair * 4u,
                    packed,
                );
            }
        }
    }
}

/*__JXL_MODULAR_RESUME__*/
/*__JXL_MODULAR_RECONSTRUCT__*/

@compute @workgroup_size(wg_x, wg_y, 1)
fn decode(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let lane_index = global_invocation_id.x;
    if lane_index >= dispatch_control.group_count {
        return;
    }
    let group_index = dispatch_control.first_group + lane_index;
    params = params_table[group_index];
    reconstruction_base = lane_index * dispatch_control.lane_stride_words;
    decode_error = 0u;
    if window_is_first() {
        bit_cursor = params.entropy.token_start;
        consumer_decoded = 0u;
        entropy_begin();
    } else {
        load_entropy_execution_state();
    }
    current_channel = consumer_decoded / params.sample_count;

    while current_channel < params.source_channels && decode_error == 0u
        && !window_should_pause() {
        let channel_start = current_channel * params.sample_count;
        let decoded = decode_adaptive_channel(
            consumer_decoded - channel_start,
            !window_is_final(),
            params.window_yield_end,
        );
        consumer_decoded += decoded;
        if decode_error != 0u || window_should_pause() {
            break;
        }
        if consumer_decoded != (current_channel + 1u) * params.sample_count {
            decode_error = ERROR_RAW_TOKEN;
            break;
        }
        current_channel += 1u;
    }
    let expected_samples = params.sample_count * params.source_channels;
    var status_code = decode_error;
    if decode_error != 0u {
        if !window_is_final() {
            save_entropy_execution_state(decode_error);
        }
    } else if consumer_decoded != expected_samples {
        if window_is_final() {
            status_code = ERROR_TRUNCATED_BITS;
        } else {
            save_entropy_execution_state(0u);
            status_code = STATUS_IN_PROGRESS;
        }
    } else if !window_is_final() {
        save_entropy_execution_state(0u);
        status_code = STATUS_IN_PROGRESS;
    } else {
        entropy_finish_exact();
        status_code = decode_error;
        if decode_error == 0u && params.fixed_output_mode == 0u {
            finalize_output();
            status_code = decode_error;
        }
    }
    let status_base = params.status_index * 4u;
    if status_code == 0u {
        status_code = STATUS_OK;
    }
    status[status_base] = status_code;
    status[status_base + 1u] = consumer_decoded;
    status[status_base + 2u] = bit_cursor;
    status[status_base + 3u] = params.stream_token_end;
}
