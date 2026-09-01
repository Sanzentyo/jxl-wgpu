/*__JXL_MODULAR_ENTROPY_ABI__*/
/*__JXL_VARDCT_BLOCK_CONTEXT__*/

struct Params {
    entropy: EntropyStreamParams,
    window_logical_start: u32,
    window_upload_start: u32,
    stream_token_end: u32,
    window_yield_end: u32,
    window_flags: u32,
    execution_state_base_words: u32,
    status_index: u32,
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
    global_group_index: u32,
    block_context: HfBlockContextTables,
    _reserved: u32,
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
const STATUS_IN_PROGRESS: u32 = 14u;
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

const WINDOW_FIRST: u32 = 1u;
const WINDOW_FINAL: u32 = 2u;
const PHASE_NONZERO_COUNT: u32 = 0u;
const PHASE_COEFFICIENTS: u32 = 1u;
const PHASE_DONE: u32 = 2u;

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
        value |= codestream[word_index + 1u] << (32u - word_shift);
    }
    return value & bit_mask(count);
}

fn read_bits(count: u32) -> u32 {
    if decode_error != 0u { return 0u; }
    if count > 32u || bit_cursor < params.window_logical_start
        || bit_cursor > params.entropy.token_end
        || count > params.entropy.token_end - bit_cursor {
        decode_error = ERROR_TRUNCATED_BITS;
        return 0u;
    }
    let value = peek_bits(count);
    bit_cursor += count;
    return value;
}

fn modular_metadata_base() -> u32 {
    return 0u;
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

fn window_is_first() -> bool {
    return (params.window_flags & WINDOW_FIRST) != 0u;
}

fn window_is_final() -> bool {
    return (params.window_flags & WINDOW_FINAL) != 0u;
}

fn window_should_pause() -> bool {
    return !window_is_final() && bit_cursor >= params.window_yield_end;
}

fn advance_channel(position: ptr<function, vec3<u32>>) {
    (*position).z += 1u;
    if (*position).z == 3u {
        (*position).z = 0u;
        (*position).x += 1u;
        if (*position).x == params.block_width {
            (*position).x = 0u;
            (*position).y += 1u;
        }
    }
}

fn load_execution_state(
    position: ptr<function, vec3<u32>>,
    coefficient_progress: ptr<function, vec4<u32>>,
    selected_preset: ptr<function, u32>,
    nonzero_coefficients: ptr<function, u32>,
    phase: ptr<function, u32>,
    nonzero_grid: ptr<function, array<u32, 96>>,
) {
    let base = params.execution_state_base_words;
    bit_cursor = reconstruction_load(base);
    entropy_ans_state = reconstruction_load(base + 1u);
    entropy_copy_remaining = reconstruction_load(base + 2u);
    entropy_copy_position = reconstruction_load(base + 3u);
    entropy_decoded = reconstruction_load(base + 4u);
    entropy_last_value = reconstruction_load(base + 5u);
    *phase = reconstruction_load(base + 6u);
    decode_error = reconstruction_load(base + 7u);
    *selected_preset = reconstruction_load(base + 8u);
    (*position).y = reconstruction_load(base + 9u);
    (*position).x = reconstruction_load(base + 10u);
    (*position).z = reconstruction_load(base + 11u);
    (*coefficient_progress).x = reconstruction_load(base + 12u);
    (*coefficient_progress).y = reconstruction_load(base + 13u);
    (*coefficient_progress).z = reconstruction_load(base + 14u);
    *nonzero_coefficients = reconstruction_load(base + 15u);
    hf_coefficient_error = reconstruction_load(base + 16u);
    for (var index = 0u; index < 96u; index += 1u) {
        (*nonzero_grid)[index] = reconstruction_load(base + 18u + index);
    }
}

fn save_execution_state(
    position: vec3<u32>,
    coefficient_progress: vec4<u32>,
    selected_preset: u32,
    nonzero_coefficients: u32,
    phase: u32,
    nonzero_grid: ptr<function, array<u32, 96>>,
) {
    let base = params.execution_state_base_words;
    reconstruction_store(base, bit_cursor);
    reconstruction_store(base + 1u, entropy_ans_state);
    reconstruction_store(base + 2u, entropy_copy_remaining);
    reconstruction_store(base + 3u, entropy_copy_position);
    reconstruction_store(base + 4u, entropy_decoded);
    reconstruction_store(base + 5u, entropy_last_value);
    reconstruction_store(base + 6u, phase);
    reconstruction_store(base + 7u, decode_error);
    reconstruction_store(base + 8u, selected_preset);
    reconstruction_store(base + 9u, position.y);
    reconstruction_store(base + 10u, position.x);
    reconstruction_store(base + 11u, position.z);
    reconstruction_store(base + 12u, coefficient_progress.x);
    reconstruction_store(base + 13u, coefficient_progress.y);
    reconstruction_store(base + 14u, coefficient_progress.z);
    reconstruction_store(base + 15u, nonzero_coefficients);
    reconstruction_store(base + 16u, hf_coefficient_error);
    reconstruction_store(base + 17u, 0u);
    for (var index = 0u; index < 96u; index += 1u) {
        reconstruction_store(base + 18u + index, (*nonzero_grid)[index]);
    }
    reconstruction_store(base + 114u, 0u);
    reconstruction_store(base + 115u, 0u);
}

@compute @workgroup_size(1, 1, 1)
fn decode_hf_coefficients(@builtin(workgroup_id) workgroup: vec3<u32>) {
    let lane = workgroup.x;
    params = params_input[lane];
    decode_error = 0u;
    hf_coefficient_error = 0u;
    var selected_preset = 0u;
    var nonzero_coefficients = 0u;
    var position = vec3<u32>(0u);
    // x=order index, y=remaining nonzero, z=previous-nonzero. The fourth word is reserved.
    var coefficient_progress = vec4<u32>(0u);
    var phase = PHASE_NONZERO_COUNT;
    var nonzero_grid: array<u32, 96>;

    if window_is_first() {
        bit_cursor = params.entropy.token_start;
        for (var index = 0u; index < 96u; index += 1u) {
            nonzero_grid[index] = 0u;
        }
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
    } else {
        load_execution_state(
            &position,
            &coefficient_progress,
            &selected_preset,
            &nonzero_coefficients,
            &phase,
            &nonzero_grid,
        );
    }

    let preset_context_base = selected_preset * 495u * params.num_block_clusters;
    loop {
        if decode_error != 0u || phase == PHASE_DONE || window_should_pause() {
            break;
        }
        let x = position.x;
        let y = position.y;
        let order_channel = position.z;
        if y >= params.block_height {
            phase = PHASE_DONE;
            break;
        }
        let raster = (params.block_origin_y + y) * params.blocks_per_row
            + params.block_origin_x + x;
        let task_plus_one = hf_artifact[params.block_task_map_offset_words + raster];
        if task_plus_one == 0u {
            if order_channel != 0u || phase != PHASE_NONZERO_COUNT {
                fail(ERROR_BLOCK_TASK);
                continue;
            }
            position.x += 1u;
            if position.x == params.block_width {
                position.x = 0u;
                position.y += 1u;
            }
            continue;
        }
        let task_index = task_plus_one - 1u;
        let task_metadata = hf_sink_params.task_metadata_offset_words + task_index * 12u;
        if hf_artifact[task_metadata + 2u] != params.block_origin_x + x
            || hf_artifact[task_metadata + 3u] != params.block_origin_y + y
            || hf_artifact[task_metadata + 4u] == 0u
            || hf_artifact[task_metadata + 5u] == 0u
            || hf_artifact[task_metadata + 7u] != params.status_index {
            fail(ERROR_TASK_SHAPE);
            continue;
        }
        let task_block_width = hf_artifact[task_metadata + 4u];
        let task_block_height = hf_artifact[task_metadata + 5u];
        let num_blocks = task_block_width * task_block_height;
        let num_blocks_log = countTrailingZeros(num_blocks);
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

        if phase == PHASE_NONZERO_COUNT {
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
            let remaining_nonzero = entropy_read_varint(context_cluster(nonzero_context), 0u);
            if decode_error != 0u { continue; }
            if remaining_nonzero > 63u * num_blocks {
                fail(ERROR_NONZERO_COUNT);
                continue;
            }
            let normalized_nonzero = (remaining_nonzero + num_blocks - 1u) >> num_blocks_log;
            for (var dx = 0u; dx < task_block_width; dx += 1u) {
                nonzero_grid[grid_index + dx] = normalized_nonzero;
            }
            if remaining_nonzero == 0u {
                advance_channel(&position);
                continue;
            }
            coefficient_progress.x = num_blocks;
            coefficient_progress.y = remaining_nonzero;
            coefficient_progress.z = select(0u, 1u, remaining_nonzero <= num_blocks * 4u);
            phase = PHASE_COEFFICIENTS;
            if window_should_pause() { break; }
        }

        let coefficient_context_base = preset_context_base + block_context * 458u
            + 37u * params.num_block_clusters;
        loop {
            if coefficient_progress.x >= 64u * num_blocks
                || coefficient_progress.y == 0u || decode_error != 0u {
                break;
            }
            let remaining_context = (coefficient_progress.y - 1u) >> num_blocks_log;
            let frequency_context = (coefficient_progress.x - num_blocks) >> num_blocks_log;
            let coefficient_context = (
                COEFF_NUM_NONZERO_CONTEXT[remaining_context]
                + COEFF_FREQ_CONTEXT[frequency_context]
            ) * 2u + coefficient_progress.z;
            let packed = entropy_read_varint(
                context_cluster(coefficient_context_base + coefficient_context), 0u
            );
            let order_index = coefficient_progress.x;
            coefficient_progress.x += 1u;
            if decode_error != 0u { break; }
            if packed == 0u {
                coefficient_progress.z = 0u;
            } else {
                let coefficient = unpack_signed(packed) << params.coeff_shift;
                if !hf_store_quantized_coefficient(
                    task_index, channel, order_index, coefficient
                ) {
                    fail(ERROR_COEFFICIENT_SINK + hf_coefficient_error);
                    break;
                }
                coefficient_progress.z = 1u;
                coefficient_progress.y -= 1u;
                nonzero_coefficients += 1u;
            }
            if window_should_pause() { break; }
        }
        if decode_error != 0u || window_should_pause() { break; }
        phase = PHASE_NONZERO_COUNT;
        coefficient_progress = vec4<u32>(0u);
        advance_channel(&position);
    }

    var status_code = decode_error;
    if decode_error == 0u && phase == PHASE_DONE {
        if window_is_final() {
            entropy_finish_exact();
            status_code = decode_error;
        } else {
            status_code = STATUS_IN_PROGRESS;
        }
    } else if decode_error == 0u {
        if window_is_final() {
            status_code = ERROR_TRUNCATED_BITS;
        } else {
            status_code = STATUS_IN_PROGRESS;
        }
    }
    if !window_is_final() {
        save_execution_state(
            position,
            coefficient_progress,
            selected_preset,
            nonzero_coefficients,
            phase,
            &nonzero_grid,
        );
    }
    if status_code == 0u { status_code = STATUS_OK; }
    let status_base = params.status_index * 8u;
    statuses[status_base] = status_code;
    statuses[status_base + 1u] = bit_cursor;
    statuses[status_base + 2u] = params.stream_token_end;
    statuses[status_base + 3u] = entropy_decoded;
    statuses[status_base + 4u] = selected_preset;
    statuses[status_base + 5u] = params.global_group_index;
    statuses[status_base + 6u] = nonzero_coefficients;
    statuses[status_base + 7u] = hf_coefficient_error;
}
