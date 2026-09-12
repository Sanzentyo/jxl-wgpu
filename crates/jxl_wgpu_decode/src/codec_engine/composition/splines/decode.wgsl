/*__JXL_MODULAR_ENTROPY_ABI__*/

struct Params {
    entropy: EntropyStreamParams,
    capacity: u32,
    window: vec4<u32>,
    limits: vec4<u32>, // control points, context offset, reset, header stride
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
const ERROR_COUNT: u32 = 12u;
const ERROR_POSITION: u32 = 13u;
const ERROR_DELTA: u32 = 14u;
const ERROR_COINCIDENT: u32 = 15u;
const POSITION_LIMIT: i32 = 8388608;
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

fn value(context: u32) -> u32 {
    return entropy_read_varint(modular_metadata[params.limits.y + context], 0u);
}
fn unpack_signed(packed: u32) -> i32 {
    return bitcast<i32>((packed >> 1u) ^ (0u - (packed & 1u)));
}
fn store_word(offset: u32, word: u32) {
    if params.capacity == 0u || decode_error != 0u { return; }
    if offset >= params.capacity { decode_error = ERROR_COUNT; return; }
    commands[offset] = word;
}
fn header() -> u32 { return 4u + state[10] * params.limits.w; }
fn position(coordinate: i32) -> i32 {
    if coordinate <= -POSITION_LIMIT || coordinate >= POSITION_LIMIT {
        decode_error = ERROR_POSITION;
        return 0;
    }
    return coordinate;
}
fn starting_position(previous: i32, packed: u32) -> i32 {
    if state[10] == 0u {
        if packed >= u32(POSITION_LIMIT) { decode_error = ERROR_POSITION; return 0; }
        return i32(packed);
    }
    let change = unpack_signed(packed);
    // Each previous coordinate is bounded; reject before signed addition can wrap.
    if change <= -2 * POSITION_LIMIT || change >= 2 * POSITION_LIMIT {
        decode_error = ERROR_POSITION;
        return 0;
    }
    return position(previous + change);
}
fn delta(previous: i32, packed: u32) -> i32 {
    let change = unpack_signed(packed);
    if change <= -1073741824 || change >= 1073741824 {
        decode_error = ERROR_DELTA;
        return 0;
    }
    return position(previous + change);
}

@compute @workgroup_size(1)
fn main() {
    decode_error = state[0];
    bit_cursor = state[1];
    if params.limits.z != 0u {
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
    // One token per step, including unary distributions and LZ77 copies. The same cursor
    // is replayed only after the exact header/point storage has been admitted.
    for (var steps = 0u; steps < 4096u && decode_error == 0u && state[2] != FINISHED; steps += 1u) {
        if bit_cursor >= params.window.w && params.window.z < params.entropy.token_end { break; }
        switch state[2] {
            case 0u: {
                let count = value(2u);
                if count >= params.limits.x { decode_error = ERROR_COUNT; }
                state[9] = count + 1u;
                state[14] = count + 1u;
                state[3] = 4u + state[9] * params.limits.w;
                store_word(0u, state[9]);
                store_word(2u, state[3]);
                state[2] = 1u;
            }
            case 1u: {
                state[11] = bitcast<u32>(starting_position(bitcast<i32>(state[11]), value(1u)));
                state[2] = 2u;
            }
            case 2u: {
                state[12] = bitcast<u32>(starting_position(bitcast<i32>(state[12]), value(1u)));
                store_word(header() + 2u, state[11]);
                store_word(header() + 3u, state[12]);
                state[10] += 1u;
                state[2] = select(1u, 3u, state[10] == state[9]);
            }
            case 3u: {
                store_word(1u, bitcast<u32>(unpack_signed(value(0u))));
                state[10] = 0u;
                state[2] = 4u;
            }
            case 4u: {
                let count = value(3u);
                if count > params.limits.x - state[14] { decode_error = ERROR_COUNT; }
                state[16] = state[3];
                state[3] += count * 2u;
                state[14] += count;
                state[15] = count;
                state[17] = 0u;
                state[18] = 0u;
                state[21] = 0u;
                store_word(header(), state[16]);
                store_word(header() + 1u, count + 1u);
                if params.capacity != 0u && decode_error == 0u {
                    state[19] = commands[header() + 2u];
                    state[20] = commands[header() + 3u];
                }
                state[2] = select(5u, 7u, count == 0u);
            }
            case 5u: {
                state[17] = bitcast<u32>(delta(bitcast<i32>(state[17]), value(4u)));
                state[2] = 6u;
            }
            case 6u: {
                state[18] = bitcast<u32>(delta(bitcast<i32>(state[18]), value(4u)));
                if state[17] == 0u && state[18] == 0u { decode_error = ERROR_COINCIDENT; }
                if params.capacity != 0u && decode_error == 0u {
                    state[19] = bitcast<u32>(position(bitcast<i32>(state[19]) + bitcast<i32>(state[17])));
                    state[20] = bitcast<u32>(position(bitcast<i32>(state[20]) + bitcast<i32>(state[18])));
                    store_word(state[16], state[19]);
                    store_word(state[16] + 1u, state[20]);
                }
                state[16] += 2u;
                state[15] -= 1u;
                state[2] = select(5u, 7u, state[15] == 0u);
            }
            case 7u: {
                store_word(header() + 4u + state[21], bitcast<u32>(unpack_signed(value(5u))));
                state[21] += 1u;
                if state[21] == 128u {
                    state[10] += 1u;
                    state[2] = select(4u, FINISHED, state[10] == state[9]);
                }
            }
            default: { decode_error = ERROR_COUNT; }
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
