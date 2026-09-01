/*__JXL_MODULAR_ENTROPY_ABI__*/

struct ProbeParams {
    entropy: EntropyStreamParams,
    symbol_count: u32,
    distance_multiplier: u32,
    _reserved0: u32,
    _reserved1: u32,
    _reserved2: u32,
};

@group(0) @binding(0) var<storage, read> codestream: array<u32>;
@group(0) @binding(1) var<storage, read> modular_metadata: array<u32>;
@group(0) @binding(2) var<storage, read> contexts: array<u32>;
@group(0) @binding(3) var<storage, read> params_input: array<ProbeParams>;
@group(0) @binding(4) var<storage, read_write> reconstruction: array<u32>;
@group(0) @binding(5) var<storage, read_write> status: array<u32>;

var<private> bit_cursor: u32;
var<private> decode_error: u32;
var<private> params: ProbeParams;

const ERROR_TRUNCATED_BITS: u32 = 2u;
const ERROR_PREFIX: u32 = 3u;
const ERROR_LZ77_STATE: u32 = 5u;
const ERROR_TRAILING_BITS: u32 = 7u;
const ERROR_ANS_STATE: u32 = 10u;
const ERROR_ENTROPY_CLUSTER: u32 = 11u;

fn reconstruction_load(index: u32) -> u32 {
    return reconstruction[index];
}

fn reconstruction_store(index: u32, value: u32) {
    reconstruction[index] = value;
}

fn entropy_window_base() -> u32 {
    return params.symbol_count;
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
    if decode_error != 0u { return 0u; }
    if count > 32u || bit_cursor > params.entropy.token_end
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

@compute @workgroup_size(1, 1, 1)
fn probe_entropy() {
    params = params_input[0];
    bit_cursor = params.entropy.token_start;
    decode_error = 0u;
    entropy_begin();
    for (var index = 0u; index < params.symbol_count && decode_error == 0u; index += 1u) {
        reconstruction[index] = entropy_read_varint(
            contexts[index],
            params.distance_multiplier,
        );
    }
    entropy_finish_exact();
    status[0] = decode_error;
    status[1] = bit_cursor;
    status[2] = entropy_ans_state;
    status[3] = entropy_decoded;
}
