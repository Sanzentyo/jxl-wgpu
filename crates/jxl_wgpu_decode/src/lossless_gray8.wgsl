struct Params {
    token_start: u32,
    token_end: u32,
    width: u32,
    height: u32,
    sample_count: u32,
    output_mode: u32,
    transfer: u32,
    limited_range: u32,
    plane0_offset: u32,
    plane0_stride: u32,
    plane1_offset: u32,
    plane1_stride: u32,
    plane2_offset: u32,
    plane2_stride: u32,
    chroma_width: u32,
    chroma_height: u32,
};

@group(0) @binding(0) var<storage, read> codestream: array<u32>;
@group(0) @binding(1) var<storage, read> prefix_lookup: array<u32>;
@group(0) @binding(2) var<storage, read_write> reconstructed: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_words: array<u32>;
@group(0) @binding(4) var<storage, read_write> status: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

var<private> bit_cursor: u32;
var<private> decode_error: u32;

const STATUS_OK: u32 = 1u;
const ERROR_TRUNCATED_BITS: u32 = 2u;
const ERROR_PREFIX: u32 = 3u;
const ERROR_RAW_TOKEN: u32 = 4u;
const ERROR_LZ77_STATE: u32 = 5u;
const ERROR_LZ77_LENGTH: u32 = 6u;
const ERROR_TRAILING_BITS: u32 = 7u;

fn bit_mask(count: u32) -> u32 {
    if count == 0u {
        return 0u;
    }
    return (1u << count) - 1u;
}

fn peek_bits(count: u32) -> u32 {
    if count == 0u {
        return 0u;
    }
    let word_index = bit_cursor >> 5u;
    let word_shift = bit_cursor & 31u;
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
    if count > 31u || bit_cursor > params.token_end || count > params.token_end - bit_cursor {
        decode_error = ERROR_TRUNCATED_BITS;
        return 0u;
    }
    let value = peek_bits(count);
    bit_cursor = bit_cursor + count;
    return value;
}

fn read_prefix_symbol() -> u32 {
    if decode_error != 0u || bit_cursor >= params.token_end {
        decode_error = ERROR_TRUNCATED_BITS;
        return 0xffffffffu;
    }
    let available = min(15u, params.token_end - bit_cursor);
    let lookup_index = peek_bits(available);
    let entry = prefix_lookup[lookup_index];
    let bit_len = entry & 0xffu;
    if bit_len == 0u || bit_len > available {
        decode_error = ERROR_PREFIX;
        return 0xffffffffu;
    }
    bit_cursor = bit_cursor + bit_len;
    return entry >> 8u;
}

fn decode_raw_hybrid(token: u32) -> u32 {
    if token == 0u {
        return 0u;
    }
    if token >= 19u {
        decode_error = ERROR_RAW_TOKEN;
        return 0u;
    }
    let extra_count = token - 1u;
    let value = (1u << extra_count) + read_bits(extra_count);
    if value > 510u {
        decode_error = ERROR_RAW_TOKEN;
        return 0u;
    }
    return value;
}

fn decode_lz77_hybrid(token: u32) -> u32 {
    if token < 16u {
        return token;
    }
    if token >= 33u {
        decode_error = ERROR_LZ77_LENGTH;
        return 0u;
    }
    let extra_count = token - 12u;
    return (1u << extra_count) + read_bits(extra_count);
}

fn unpack_signed(value: u32) -> i32 {
    if (value & 1u) == 0u {
        return i32(value >> 1u);
    }
    return -i32((value + 1u) >> 1u);
}

fn write_byte(offset: u32, value: u32) {
    let word_index = offset >> 2u;
    let shift = (offset & 3u) << 3u;
    let mask = 0xffu << shift;
    output_words[word_index] = (output_words[word_index] & ~mask) | ((value & 0xffu) << shift);
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

fn luma_code(value: u32) -> u32 {
    let nonlinear = clamp(target_nonlinear(value), 0.0, 1.0);
    if params.limited_range != 0u {
        return u32(round(16.0 + 219.0 * nonlinear));
    }
    return u32(round(255.0 * nonlinear));
}

fn emit_token(index: u32, packed: u32) {
    let x = index % params.width;
    let y = index / params.width;
    var left = 0i;
    var top = 0i;
    var top_left = 0i;
    if x > 0u {
        left = i32(reconstructed[index - 1u]);
    } else if y > 0u {
        left = i32(reconstructed[index - params.width]);
    }
    if y == 0u {
        top = left;
        top_left = left;
    } else {
        top = i32(reconstructed[index - params.width]);
        if x == 0u {
            top_left = top;
        } else {
            top_left = i32(reconstructed[index - params.width - 1u]);
        }
    }
    let gradient = left + top - top_left;
    let prediction = clamp(gradient, min(left, top), max(left, top));
    let sample = u32(prediction + unpack_signed(packed)) & 255u;
    reconstructed[index] = sample;

    let output_offset = params.plane0_offset + y * params.plane0_stride + x;
    if params.output_mode == 0u {
        write_byte(output_offset, sample);
    } else {
        write_byte(output_offset, luma_code(sample));
    }
}

fn fill_chroma() {
    if params.output_mode == 2u {
        for (var y = 0u; y < params.chroma_height; y = y + 1u) {
            for (var x = 0u; x < params.chroma_width; x = x + 1u) {
                let offset = params.plane1_offset + y * params.plane1_stride + x * 2u;
                write_byte(offset, 128u);
                write_byte(offset + 1u, 128u);
            }
        }
    } else if params.output_mode == 3u {
        for (var y = 0u; y < params.chroma_height; y = y + 1u) {
            for (var x = 0u; x < params.chroma_width; x = x + 1u) {
                write_byte(params.plane1_offset + y * params.plane1_stride + x, 128u);
                write_byte(params.plane2_offset + y * params.plane2_stride + x, 128u);
            }
        }
    }
}

@compute @workgroup_size(1)
fn decode() {
    bit_cursor = params.token_start;
    decode_error = 0u;
    var decoded = 0u;
    // The profile's RLE code may begin the stream. Its implicit distance-one history is zero.
    var last_token = 0u;

    while decoded < params.sample_count && decode_error == 0u {
        let symbol = read_prefix_symbol();
        if symbol == 0u {
            // With LZ77 enabled raw zero is an escape candidate. The encoder emits its LZ77
            // length code immediately afterwards; otherwise zero remains an ordinary literal.
            let after_zero = bit_cursor;
            var following = 0xffffffffu;
            if bit_cursor < params.token_end {
                following = read_prefix_symbol();
            }
            if following >= 224u && following < 257u {
                let run_value = decode_lz77_hybrid(following - 224u);
                let run_count = run_value + 8u;
                if decode_error != 0u || run_count > params.sample_count - decoded {
                    decode_error = ERROR_LZ77_LENGTH;
                    break;
                }
                for (var copied = 0u; copied < run_count; copied = copied + 1u) {
                    emit_token(decoded, 0u);
                    decoded = decoded + 1u;
                }
                last_token = 0u;
            } else {
                bit_cursor = after_zero;
                decode_error = 0u;
                emit_token(decoded, 0u);
                decoded = decoded + 1u;
                last_token = 0u;
            }
        } else if symbol < 19u {
            let packed = decode_raw_hybrid(symbol);
            if decode_error == 0u {
                emit_token(decoded, packed);
                decoded = decoded + 1u;
                last_token = packed;
            }
        } else if symbol >= 224u && symbol < 257u {
            if last_token != 0u {
                decode_error = ERROR_LZ77_STATE;
                break;
            }
            // An LZ77 symbol must be consumed together with the preceding raw-zero escape.
            decode_error = ERROR_LZ77_STATE;
            break;
        } else {
            decode_error = ERROR_PREFIX;
        }
    }

    if decode_error == 0u && bit_cursor != params.token_end {
        decode_error = ERROR_TRAILING_BITS;
    }
    if decode_error == 0u {
        fill_chroma();
        status[0] = STATUS_OK;
    } else {
        status[0] = decode_error;
    }
    status[1] = decoded;
    status[2] = bit_cursor;
    status[3] = params.token_end;
}
