/*__JXL_MODULAR_ENTROPY_ABI__*/
/*__JXL_VARDCT_BLOCK_CONTEXT__*/

struct Params {
    entropy: EntropyStreamParams,
    block_origin_x: u32,
    block_origin_y: u32,
    block_width: u32,
    block_height: u32,
    blocks_per_row: u32,
    block_task_map_offset_words: u32,
    num_hf_presets: u32,
    num_block_clusters: u32,
    context_map_offset_words: u32,
    lf_plane_stride_words: u32,
    lz77_window_base_words: u32,
    coeff_shift: u32,
    _reserved: u32,
    block_context: HfBlockContextTables,
};

@group(0) @binding(0) var<storage, read> codestream: array<u32>;
// The packed common entropy descriptor starts at word zero. HF coefficient-context and block-
// context maps follow at the offsets carried by Params, avoiding two additional storage bindings.
@group(0) @binding(1) var<storage, read> modular_metadata: array<u32>;
@group(0) @binding(2) var<storage, read_write> reconstruction: array<u32>;
@group(0) @binding(3) var<storage, read> params_input: array<Params>;
@group(0) @binding(4) var<storage, read_write> statuses: array<u32>;

var<private> bit_cursor: u32;
var<private> decode_error: u32;
var<private> params: Params;

const STATUS_OK: u32 = 1u;
const ERROR_TRUNCATED_BITS: u32 = 2u;
const ERROR_PREFIX: u32 = 3u;
const ERROR_LZ77_STATE: u32 = 5u;
const ERROR_TRAILING_BITS: u32 = 7u;
const ERROR_ANS_STATE: u32 = 10u;
const ERROR_ENTROPY_CLUSTER: u32 = 11u;
const ERROR_HF_PRESET: u32 = 20u;
const ERROR_GROUP_GEOMETRY: u32 = 21u;
const ERROR_BLOCK_TASK: u32 = 22u;
const ERROR_TASK_SHAPE: u32 = 23u;
const ERROR_NONZERO_COUNT: u32 = 24u;
const ERROR_CONTEXT_MAP: u32 = 25u;
const ERROR_COEFFICIENT_SINK: u32 = 32u;

fn reconstruction_load(index: u32) -> u32 {
    return reconstruction[index];
}

fn reconstruction_store(index: u32, value: u32) {
    reconstruction[index] = value;
}

fn entropy_window_base() -> u32 {
    return params.lz77_window_base_words;
}

fn bit_mask(count: u32) -> u32 {
    if count == 0u { return 0u; }
    if count == 32u { return 0xffffffffu; }
    return (1u << count) - 1u;
}

fn peek_bits(count: u32) -> u32 {
    if count == 0u { return 0u; }
    let word_index = bit_cursor >> 5u;
    let word_shift = bit_cursor & 31u;
    var value = codestream[word_index] >> word_shift;
    if word_shift + count > 32u {
        value |= codestream[word_index + 1u] << (32u - word_shift);
    }
    return value & bit_mask(count);
}

fn read_bits(count: u32) -> u32 {
    if count > 32u || bit_cursor > params.entropy.token_end
        || count > params.entropy.token_end - bit_cursor {
        decode_error = ERROR_TRUNCATED_BITS;
        return 0u;
    }
    let value = peek_bits(count);
    bit_cursor += count;
    return value;
}

/*__JXL_MODULAR_ENTROPY__*/

/*__JXL_HF_COEFFICIENT_SINK__*/

const COEFF_FREQ_CONTEXT: array<u32, 63> = array<u32, 63>(
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u,
    15u, 16u, 16u, 17u, 17u, 18u, 18u, 19u, 19u, 20u, 20u, 21u, 21u, 22u, 22u,
    23u, 23u, 23u, 23u, 24u, 24u, 24u, 24u, 25u, 25u, 25u, 25u, 26u, 26u, 26u,
    26u, 27u, 27u, 27u, 27u, 28u, 28u, 28u, 28u, 29u, 29u, 29u, 29u, 30u, 30u,
    30u, 30u,
);

const COEFF_NUM_NONZERO_CONTEXT: array<u32, 63> = array<u32, 63>(
    0u, 31u, 62u, 62u, 93u, 93u, 93u, 93u, 123u, 123u, 123u, 123u, 152u,
    152u, 152u, 152u, 152u, 152u, 152u, 152u, 180u, 180u, 180u, 180u, 180u,
    180u, 180u, 180u, 180u, 180u, 180u, 180u, 206u, 206u, 206u, 206u, 206u,
    206u, 206u, 206u, 206u, 206u, 206u, 206u, 206u, 206u, 206u, 206u, 206u,
    206u, 206u, 206u, 206u, 206u, 206u, 206u, 206u, 206u, 206u, 206u, 206u,
    206u, 206u,
);

fn fail(code: u32) {
    if decode_error == 0u {
        decode_error = code;
    }
}

fn context_cluster(context_index: u32) -> u32 {
    let context_count = 495u * params.num_hf_presets * params.num_block_clusters;
    if context_index >= context_count {
        fail(ERROR_CONTEXT_MAP);
        return 0u;
    }
    return modular_metadata[params.context_map_offset_words + context_index];
}

fn unpack_signed(value: u32) -> i32 {
    if (value & 1u) == 0u {
        return i32(value >> 1u);
    }
    return bitcast<i32>(0u - ((value >> 1u) + 1u));
}

fn channel_index(order_channel: u32) -> u32 {
    if order_channel == 0u { return 1u; }
    if order_channel == 1u { return 0u; }
    return 2u;
}

@compute @workgroup_size(1, 1, 1)
fn decode_hf_coefficients(@builtin(workgroup_id) workgroup: vec3<u32>) {
    let lane = workgroup.x;
    params = params_input[lane];
    bit_cursor = params.entropy.token_start;
    decode_error = 0u;
    hf_coefficient_error = 0u;
    var nonzero_coefficients = 0u;
    var selected_preset = 0u;
    let status_base = lane * 8u;

    if params.block_width == 0u || params.block_height == 0u
        || params.block_width > 32u {
        fail(ERROR_GROUP_GEOMETRY);
    }
    let preset_bits = select(
        0u,
        32u - countLeadingZeros(params.num_hf_presets - 1u),
        params.num_hf_presets > 1u,
    );
    if decode_error == 0u {
        selected_preset = read_bits(preset_bits);
        if selected_preset >= params.num_hf_presets {
            fail(ERROR_HF_PRESET);
        }
    }
    entropy_begin();

    var nonzero_grid: array<u32, 96>;
    for (var index = 0u; index < 96u; index += 1u) {
        nonzero_grid[index] = 0u;
    }
    let preset_context_base = selected_preset * 495u * params.num_block_clusters;
    for (var y = 0u; y < params.block_height && decode_error == 0u; y += 1u) {
        for (var x = 0u; x < params.block_width && decode_error == 0u; x += 1u) {
            let raster = (params.block_origin_y + y) * params.blocks_per_row
                + params.block_origin_x + x;
            let task_plus_one = hf_artifact[params.block_task_map_offset_words + raster];
            if task_plus_one == 0u {
                fail(ERROR_BLOCK_TASK);
                continue;
            }
            let task_index = task_plus_one - 1u;
            let task_metadata = hf_sink_params.task_metadata_offset_words + task_index * 12u;
            if hf_artifact[task_metadata] != 0u || hf_artifact[task_metadata + 2u] != params.block_origin_x + x
                || hf_artifact[task_metadata + 3u] != params.block_origin_y + y
                || hf_artifact[task_metadata + 4u] != 1u || hf_artifact[task_metadata + 5u] != 1u
                || hf_artifact[task_metadata + 7u] != lane {
                fail(ERROR_TASK_SHAPE);
                continue;
            }

            for (var order_channel = 0u; order_channel < 3u && decode_error == 0u;
                order_channel += 1u) {
                let channel = channel_index(order_channel);
                let qf = hf_artifact[task_metadata + 6u];
                let order_id = hf_artifact[task_metadata + 10u];
                let lf = vec3<i32>(
                    bitcast<i32>(reconstruction[params.lf_plane_stride_words + raster]),
                    bitcast<i32>(reconstruction[raster]),
                    bitcast<i32>(reconstruction[2u * params.lf_plane_stride_words + raster]),
                );
                let block_context = hf_block_context(
                    params.block_context,
                    order_channel,
                    order_id,
                    qf,
                    lf,
                );
                if block_context >= params.num_block_clusters {
                    fail(ERROR_CONTEXT_MAP);
                    continue;
                }
                let grid_index = channel * 32u + x;
                var predicted = 32u;
                if y == 0u {
                    if x != 0u { predicted = nonzero_grid[grid_index - 1u]; }
                } else if x == 0u {
                    predicted = nonzero_grid[grid_index];
                } else {
                    predicted = (nonzero_grid[grid_index] + nonzero_grid[grid_index - 1u] + 1u) >> 1u;
                }
                let predicted_context = select(predicted, 4u + predicted / 2u, predicted >= 8u);
                let nonzero_context = preset_context_base + block_context
                    + predicted_context * params.num_block_clusters;
                var remaining_nonzero = entropy_read_varint(
                    context_cluster(nonzero_context), 0u
                );
                if remaining_nonzero > 63u {
                    fail(ERROR_NONZERO_COUNT);
                    continue;
                }
                nonzero_grid[grid_index] = remaining_nonzero;
                if remaining_nonzero == 0u { continue; }

                var previous_nonzero = select(0u, 1u, remaining_nonzero <= 4u);
                let coefficient_context_base = preset_context_base + block_context * 458u
                    + 37u * params.num_block_clusters;
                for (var order_index = 1u; order_index < 64u && decode_error == 0u;
                    order_index += 1u) {
                    let remaining_context = remaining_nonzero - 1u;
                    let coefficient_context = (
                        COEFF_NUM_NONZERO_CONTEXT[remaining_context]
                        + COEFF_FREQ_CONTEXT[order_index - 1u]
                    ) * 2u + previous_nonzero;
                    let packed = entropy_read_varint(
                        context_cluster(coefficient_context_base + coefficient_context), 0u
                    );
                    if packed == 0u { previous_nonzero = 0u; continue; }
                    let coefficient = unpack_signed(packed) << params.coeff_shift;
                    if !hf_store_quantized_coefficient(
                        task_index, channel, order_index, coefficient
                    ) {
                        fail(ERROR_COEFFICIENT_SINK + hf_coefficient_error);
                        continue;
                    }
                    previous_nonzero = 1u;
                    remaining_nonzero -= 1u;
                    nonzero_coefficients += 1u;
                    if remaining_nonzero == 0u { break; }
                }
            }
        }
    }
    entropy_finish_exact();
    statuses[status_base] = select(decode_error, STATUS_OK, decode_error == 0u);
    statuses[status_base + 1u] = bit_cursor;
    statuses[status_base + 2u] = params.entropy.token_end;
    statuses[status_base + 3u] = entropy_decoded;
    statuses[status_base + 4u] = selected_preset;
    statuses[status_base + 5u] = lane;
    statuses[status_base + 6u] = nonzero_coefficients;
    statuses[status_base + 7u] = hf_coefficient_error;
}
