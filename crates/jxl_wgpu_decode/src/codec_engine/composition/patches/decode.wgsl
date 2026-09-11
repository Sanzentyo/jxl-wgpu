/*__JXL_MODULAR_ENTROPY_ABI__*/

struct Params {
    entropy: EntropyStreamParams,
    capacity: u32,
    window: vec4<u32>, // logical start, upload start, available end, yield end
    image: vec4<u32>, // width, height, extra count, max reference rectangles
    limits: vec4<u32>, // max positions, command words, context offset, reset
    references: array<vec4<u32>, 4>, // width, height, present, before transform
};
@group(0) @binding(0) var<storage, read> codestream: array<u32>;
@group(0) @binding(1) var<storage, read> modular_metadata: array<u32>;
@group(0) @binding(2) var<storage, read_write> state: array<u32>;
@group(0) @binding(3) var<storage, read_write> commands: array<u32>;
@group(0) @binding(4) var<storage, read> params: Params;

var<private> bit_cursor: u32;
var<private> decode_error: u32;
const ERROR_TRUNCATED_BITS: u32 = 2u;
const ERROR_PREFIX: u32 = 3u;
const ERROR_LZ77_STATE: u32 = 5u;
const ERROR_TRAILING_BITS: u32 = 7u;
const ERROR_ANS_STATE: u32 = 10u;
const ERROR_ENTROPY_CLUSTER: u32 = 11u;
const ERROR_PATCH: u32 = 12u;
const ERROR_REFERENCE: u32 = 13u;
const ERROR_POSITION: u32 = 14u;
const ERROR_BLEND: u32 = 15u;
const FINISHED: u32 = 0xffffffffu;

fn modular_metadata_base() -> u32 { return 0u; }
fn entropy_window_base() -> u32 { return 32u; }
fn reconstruction_load(index: u32) -> u32 { return state[index]; }
fn reconstruction_store(index: u32, value: u32) { state[index] = value; }
fn bit_mask(count: u32) -> u32 {
    if count == 32u { return 0xffffffffu; }
    return (1u << count) - 1u;
}
fn peek_bits(count: u32) -> u32 {
    if count == 0u { return 0u; }
    let position = bit_cursor - params.window.x + params.window.y;
    let shift = position & 31u;
    var value = codestream[position >> 5u] >> shift;
    if shift + count > 32u {
        value |= codestream[(position >> 5u) + 1u] << (32u - shift);
    }
    return value & bit_mask(count);
}
fn read_bits(count: u32) -> u32 {
    if decode_error != 0u { return 0u; }
    if count > 32u || bit_cursor < params.window.x || bit_cursor > params.window.z
        || count > params.window.z - bit_cursor {
        decode_error = ERROR_TRUNCATED_BITS;
        return 0u;
    }
    let value = peek_bits(count);
    bit_cursor += count;
    return value;
}

/*__JXL_MODULAR_ENTROPY__*/

fn patch_value(context: u32) -> u32 {
    return entropy_read_varint(modular_metadata[params.limits.z + context], 0u);
}
fn store_command(field: u32, value: u32) {
    if params.capacity != 0u {
        if state[3] >= params.capacity { decode_error = ERROR_PATCH; return; }
        commands[state[3] * params.limits.y + field] = value;
    }
}
fn check_rect(x: u32, y: u32, width: u32, height: u32) -> bool {
    return x <= width && y <= height && state[14] <= width - x && state[15] <= height - y;
}
fn next_position() {
    state[19] = 0u;
    state[2] = 10u;
    if !check_rect(state[17], state[18], params.image.x, params.image.y) {
        decode_error = ERROR_POSITION;
    }
    store_command(0u, state[11]);
    store_command(1u, state[12]);
    store_command(2u, state[13]);
    store_command(3u, state[14]);
    store_command(4u, state[15]);
    store_command(5u, state[17]);
    store_command(6u, state[18]);
}
fn next_blend() {
    let offset = 8u + state[19] * 3u;
    store_command(offset, state[20]);
    store_command(offset + 1u, state[21]);
    var flags = state[22];
    if params.image.z != 0u && state[20] >= 4u {
        flags |= modular_metadata[params.limits.z + 10u + state[21]] << 1u;
    }
    store_command(offset + 2u, flags);
    state[19] += 1u;
    state[2] = 10u;
    if state[19] > params.image.z {
        state[3] += 1u;
        state[16] -= 1u;
        state[2] = 8u;
        if state[16] == 0u {
            state[10] -= 1u;
            state[2] = 1u;
            if state[10] == 0u { state[2] = FINISHED; }
        }
    }
}
fn delta(previous: u32, packed: u32) -> u32 {
    let magnitude = (packed >> 1u) + (packed & 1u);
    if (packed & 1u) != 0u {
        if magnitude > previous { decode_error = ERROR_POSITION; return 0u; }
        return previous - magnitude;
    }
    if magnitude > 0xffffffffu - previous { decode_error = ERROR_POSITION; return 0u; }
    return previous + magnitude;
}

@compute @workgroup_size(1)
fn main() {
    decode_error = state[0];
    bit_cursor = state[1];
    if params.limits.w != 0u {
        for (var i = 0u; i < 32u; i += 1u) { state[i] = 0u; }
        decode_error = 0u;
        bit_cursor = 0u;
        entropy_begin();
    } else {
        entropy_ans_state = state[4];
        entropy_copy_remaining = state[5];
        entropy_copy_position = state[6];
        entropy_decoded = state[7];
        entropy_last_value = state[8];
    }
    // Each continuation has bounded work even for zero-bit unary distributions.
    for (var steps = 0u; steps < 4096u && decode_error == 0u && state[2] != FINISHED; steps += 1u) {
        if bit_cursor >= params.window.w && params.window.z < params.entropy.token_end { break; }
        switch state[2] {
            case 0u: {
                state[10] = patch_value(0u);
                if state[10] > params.image.w { decode_error = ERROR_PATCH; }
                state[2] = select(1u, FINISHED, state[10] == 0u);
            }
            case 1u: {
                state[11] = patch_value(1u);
                if state[11] >= 4u { decode_error = ERROR_REFERENCE; }
                else if params.references[state[11]].z == 0u || params.references[state[11]].w == 0u {
                    decode_error = ERROR_REFERENCE;
                }
                state[2] = 2u;
            }
            case 2u: { state[12] = patch_value(3u); state[2] = 3u; }
            case 3u: { state[13] = patch_value(3u); state[2] = 4u; }
            case 4u: {
                let value = patch_value(2u);
                if value == 0xffffffffu { decode_error = ERROR_POSITION; }
                state[14] = value + 1u; state[2] = 5u;
            }
            case 5u: {
                let value = patch_value(2u);
                if value == 0xffffffffu { decode_error = ERROR_POSITION; }
                state[15] = value + 1u;
                let geometry = params.references[state[11]];
                if !check_rect(state[12], state[13], geometry.x, geometry.y) { decode_error = ERROR_POSITION; }
                state[2] = 6u;
            }
            case 6u: {
                let value = patch_value(7u);
                if value >= params.limits.x || state[3] > params.limits.x - (value + 1u) {
                    decode_error = ERROR_PATCH;
                }
                state[16] = value + 1u; state[2] = 7u;
            }
            case 7u: { state[17] = patch_value(4u); state[2] = 9u; }
            case 8u: { state[17] = delta(state[17], patch_value(6u)); state[2] = 13u; }
            case 9u: { state[18] = patch_value(4u); next_position(); }
            case 13u: { state[18] = delta(state[18], patch_value(6u)); next_position(); }
            case 10u: {
                state[20] = patch_value(5u); state[21] = 0u; state[22] = 0u;
                if state[20] >= 8u { decode_error = ERROR_BLEND; }
                if state[20] >= 4u && params.image.z > 1u { state[2] = 11u; }
                else if state[20] >= 3u { state[2] = 12u; }
                else { next_blend(); }
            }
            case 11u: {
                state[21] = patch_value(8u);
                if state[21] >= params.image.z { decode_error = ERROR_BLEND; }
                state[2] = 12u;
            }
            case 12u: { state[22] = u32(patch_value(9u) != 0u); next_blend(); }
            default: { decode_error = ERROR_PATCH; }
        }
    }
    if state[2] == FINISHED {
        entropy_finalize();
        if entropy_copy_remaining != 0u { decode_error = ERROR_LZ77_STATE; }
    }
    state[0] = decode_error;
    state[1] = bit_cursor;
    state[4] = entropy_ans_state;
    state[5] = entropy_copy_remaining;
    state[6] = entropy_copy_position;
    state[7] = entropy_decoded;
    state[8] = entropy_last_value;
}
