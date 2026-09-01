//! Reusable packed JPEG XL entropy descriptor decoder.
//!
//! Required caller ABI: `modular_metadata`; `params.entropy`; private `bit_cursor` and
//! `decode_error`; `read_bits`, `peek_bits`, `reconstruction_load`, `reconstruction_store`, and
//! `entropy_window_base`; plus the shared `ERROR_*` constants. Call `entropy_begin`,
//! `entropy_read_varint`, then `entropy_finish_exact` when the entropy stream owns the complete
//! token range. Consumers followed by non-entropy fields call `entropy_finalize` instead and
//! validate their enclosing section after reading those fields.

const META_NODE_COUNT: u32 = 0u;
const META_MAX_DEPTH: u32 = 1u;
const META_CODER: u32 = 2u;
const META_CLUSTER_COUNT: u32 = 3u;
const META_CONFIG_OFFSET: u32 = 4u;
const META_TREE_OFFSET: u32 = 5u;
const META_TABLE_OFFSET: u32 = 6u;
const META_TABLE_STRIDE: u32 = 7u;
const META_ANS_LOG_BUCKET: u32 = 8u;
const META_LZ_ENABLED: u32 = 9u;
const META_LZ_MIN_SYMBOL: u32 = 10u;
const META_LZ_MIN_LENGTH: u32 = 11u;
const META_LZ_LENGTH_SPLIT: u32 = 12u;
const META_LZ_LENGTH_MSB: u32 = 13u;
const META_LZ_LENGTH_LSB: u32 = 14u;
const META_DISTANCE_CLUSTER: u32 = 15u;

fn entropy_metadata(index: u32) -> u32 {
    return modular_metadata[modular_metadata_base() + index];
}

var<private> entropy_ans_state: u32;
var<private> entropy_copy_remaining: u32;
var<private> entropy_copy_position: u32;
var<private> entropy_decoded: u32;
var<private> entropy_last_value: u32;

// Pipeline specialization keeps descriptor branches out of the existing direct-channel kernels.
override descriptor_channels: u32 = 0u;

// The lossless consumer configures this private view from its transformed-channel header.  The
// shared entropy fragment is also composed into VarDCT and entropy-probe shaders, so the state is
// deliberately populated through an explicit call rather than by reading consumer-specific
// Params fields here.
var<private> modular_layout_channel_count: u32;
var<private> modular_layout_sample_count: u32;
var<private> modular_layout_max_width: u32;
var<private> modular_layout_arena_words: u32;
var<private> modular_layout_descriptor_offset: u32;
var<private> modular_layout_legacy_sample_count: u32;
var<private> modular_layout_legacy_width: u32;
var<private> modular_layout_legacy_height: u32;
var<private> modular_current_width: u32;
var<private> modular_current_height: u32;
var<private> modular_current_decoded_start: u32;
var<private> modular_current_decoded_end: u32;

fn modular_layout_configure(
    layout_offset: u32,
    legacy_channel_count: u32,
    legacy_sample_count: u32,
    legacy_width: u32,
    legacy_height: u32,
) {
    modular_layout_legacy_sample_count = legacy_sample_count;
    modular_layout_legacy_width = legacy_width;
    modular_layout_legacy_height = legacy_height;
    if layout_offset == 0u {
        modular_layout_channel_count = legacy_channel_count;
        modular_layout_sample_count = legacy_sample_count * legacy_channel_count;
        modular_layout_max_width = legacy_width;
        modular_layout_arena_words = modular_layout_sample_count;
        modular_layout_descriptor_offset = 0u;
    } else {
        modular_layout_channel_count = modular_metadata[layout_offset];
        modular_layout_sample_count = modular_metadata[layout_offset + 1u];
        modular_layout_max_width = modular_metadata[layout_offset + 2u];
        modular_layout_arena_words = modular_metadata[layout_offset + 3u];
        modular_layout_descriptor_offset = modular_metadata[layout_offset + 4u];
    }
    modular_current_width = legacy_width;
    modular_current_height = legacy_height;
    modular_current_decoded_start = 0u;
    modular_current_decoded_end = legacy_sample_count;
}

fn modular_descriptor_mode() -> bool {
    return descriptor_channels != 0u;
}

fn modular_entropy_channel_count() -> u32 {
    return modular_layout_channel_count;
}

fn modular_entropy_sample_count() -> u32 {
    return modular_layout_sample_count;
}

fn modular_entropy_max_width(fallback: u32) -> u32 {
    if modular_descriptor_mode() {
        return modular_layout_max_width;
    }
    return fallback;
}

fn modular_arena_words(fallback: u32) -> u32 {
    if modular_descriptor_mode() {
        return modular_layout_arena_words;
    }
    return fallback;
}

fn modular_channel_descriptor(channel: u32) -> u32 {
    return modular_layout_descriptor_offset + channel * 8u;
}

fn modular_channel_decoded_start(channel: u32) -> u32 {
    if !modular_descriptor_mode() {
        return channel * modular_layout_legacy_sample_count;
    }
    return modular_metadata[modular_channel_descriptor(channel) + 4u];
}

fn modular_channel_decoded_end(channel: u32) -> u32 {
    if !modular_descriptor_mode() {
        return (channel + 1u) * modular_layout_legacy_sample_count;
    }
    return modular_metadata[modular_channel_descriptor(channel) + 5u];
}

fn modular_channel_for_decoded(decoded: u32) -> u32 {
    var channel = 0u;
    while channel < modular_layout_channel_count {
        if decoded < modular_channel_decoded_end(channel) {
            return channel;
        }
        channel += 1u;
    }
    return modular_layout_channel_count;
}

fn modular_select_channel(channel: u32) {
    if modular_descriptor_mode() {
        let descriptor = modular_channel_descriptor(channel);
        modular_current_width = modular_metadata[descriptor + 2u];
        modular_current_height = modular_metadata[descriptor + 3u];
        modular_current_decoded_start = modular_metadata[descriptor + 4u];
        modular_current_decoded_end = modular_metadata[descriptor + 5u];
    } else {
        modular_current_width = modular_layout_legacy_width;
        modular_current_height = modular_layout_legacy_height;
        modular_current_decoded_start = channel * modular_layout_legacy_sample_count;
        modular_current_decoded_end = (channel + 1u) * modular_layout_legacy_sample_count;
    }
}

fn modular_current_channel_width(fallback: u32) -> u32 {
    if modular_descriptor_mode() {
        return modular_current_width;
    }
    return fallback;
}

fn modular_current_channel_height(fallback: u32) -> u32 {
    if modular_descriptor_mode() {
        return modular_current_height;
    }
    return fallback;
}

fn modular_current_channel_decoded_start() -> u32 {
    return modular_current_decoded_start;
}

fn modular_current_channel_decoded_end() -> u32 {
    return modular_current_decoded_end;
}

fn modular_channel_reference_offset(channel: u32) -> u32 {
    return modular_metadata[modular_channel_descriptor(channel) + 6u];
}

fn modular_channel_reference_count(channel: u32) -> u32 {
    return modular_metadata[modular_channel_descriptor(channel) + 7u];
}

fn modular_descriptor_sample_load(channel: u32, x: u32, y: u32) -> u32 {
    let descriptor = modular_channel_descriptor(channel);
    return reconstruction_load(
        modular_metadata[descriptor]
            + y * modular_metadata[descriptor + 1u]
            + x,
    );
}

fn modular_descriptor_sample_store(channel: u32, x: u32, y: u32, value: u32) {
    let descriptor = modular_channel_descriptor(channel);
    reconstruction_store(
        modular_metadata[descriptor]
            + y * modular_metadata[descriptor + 1u]
            + x,
        value,
    );
}

fn entropy_config_offset(cluster: u32) -> u32 {
    return entropy_metadata(META_CONFIG_OFFSET) + cluster * 4u;
}

fn entropy_begin() {
    entropy_ans_state = 0u;
    entropy_copy_remaining = 0u;
    entropy_copy_position = 0u;
    entropy_decoded = 0u;
    entropy_last_value = 0u;
    if entropy_metadata(META_CODER) == 1u {
        entropy_ans_state = read_bits(32u);
    }
}

fn entropy_finalize() {
    if decode_error == 0u && entropy_metadata(META_CODER) == 1u
        && entropy_ans_state != 0x00130000u {
        decode_error = ERROR_ANS_STATE;
    }
}

fn entropy_finish_exact() {
    entropy_finalize();
    if decode_error != 0u { return; }
    if bit_cursor > params.entropy.token_end {
        decode_error = ERROR_TRUNCATED_BITS;
        return;
    }
    let remaining = params.entropy.token_end - bit_cursor;
    if remaining > 7u || peek_bits(remaining) != 0u {
        decode_error = ERROR_TRAILING_BITS;
        return;
    }
    bit_cursor = params.entropy.token_end;
}

fn entropy_read_prefix_symbol(cluster: u32) -> u32 {
    let config = entropy_config_offset(cluster);
    let single_plus_one = modular_metadata[config + 3u];
    if single_plus_one != 0u {
        return single_plus_one - 1u;
    }
    if bit_cursor >= params.entropy.token_end {
        decode_error = ERROR_TRUNCATED_BITS;
        return 0xffffffffu;
    }
    let available = min(15u, params.entropy.token_end - bit_cursor);
    let lookup_index = peek_bits(available);
    let table_offset = entropy_metadata(META_TABLE_OFFSET)
        + cluster * entropy_metadata(META_TABLE_STRIDE);
    let entry = modular_metadata[table_offset + lookup_index];
    let bit_length = entry & 0xffu;
    if bit_length == 0u || bit_length > available {
        decode_error = ERROR_PREFIX;
        return 0xffffffffu;
    }
    bit_cursor += bit_length;
    return entry >> 8u;
}

fn entropy_read_ans_symbol(cluster: u32) -> u32 {
    let log_bucket_size = entropy_metadata(META_ANS_LOG_BUCKET);
    let index = entropy_ans_state & 0xfffu;
    let bucket_index = index >> log_bucket_size;
    let position = index & ((1u << log_bucket_size) - 1u);
    let table_offset = entropy_metadata(META_TABLE_OFFSET)
        + cluster * entropy_metadata(META_TABLE_STRIDE)
        + bucket_index * 2u;
    let first = modular_metadata[table_offset];
    let second = modular_metadata[table_offset + 1u];
    let alias_symbol = first & 0xffu;
    let alias_cutoff = (first >> 8u) & 0xffu;
    var distribution = first >> 16u;
    let map_alias = position >= alias_cutoff;
    var symbol = bucket_index;
    var offset = position;
    if map_alias {
        symbol = alias_symbol;
        offset += second & 0xffffu;
        distribution = distribution ^ (second >> 16u);
    }
    if distribution == 0u {
        decode_error = ERROR_ANS_STATE;
        return 0xffffffffu;
    }
    let next = (entropy_ans_state >> 12u) * distribution + offset;
    if next < 65536u {
        entropy_ans_state = (next << 16u) | read_bits(16u);
    } else {
        entropy_ans_state = next;
    }
    return symbol;
}

fn entropy_read_symbol(cluster: u32) -> u32 {
    if cluster >= entropy_metadata(META_CLUSTER_COUNT) {
        decode_error = ERROR_ENTROPY_CLUSTER;
        return 0xffffffffu;
    }
    if entropy_metadata(META_CODER) == 0u {
        return entropy_read_prefix_symbol(cluster);
    }
    if entropy_metadata(META_CODER) == 1u {
        return entropy_read_ans_symbol(cluster);
    }
    decode_error = ERROR_ENTROPY_CLUSTER;
    return 0xffffffffu;
}

fn entropy_read_hybrid(token: u32, split_exponent: u32, msb_count: u32, lsb_count: u32) -> u32 {
    let split = 1u << split_exponent;
    if token < split {
        return token;
    }
    let embedded = msb_count + lsb_count;
    let bit_count = (split_exponent - embedded + ((token - split) >> embedded)) & 31u;
    let extra = read_bits(bit_count);
    let low = token & ((1u << lsb_count) - 1u);
    let shifted = token >> lsb_count;
    let high = (shifted & ((1u << msb_count) - 1u)) | (1u << msb_count);
    return (((high << bit_count) | extra) << lsb_count) | low;
}

fn entropy_read_clustered(cluster: u32) -> u32 {
    let token = entropy_read_symbol(cluster);
    let config = entropy_config_offset(cluster);
    return entropy_read_hybrid(
        token,
        modular_metadata[config],
        modular_metadata[config + 1u],
        modular_metadata[config + 2u],
    );
}

fn entropy_special_distance(index: u32) -> vec2<i32> {
    const distances: array<vec2<i32>, 120> = array<vec2<i32>, 120>(
        vec2<i32>(0, 1), vec2<i32>(1, 0), vec2<i32>(1, 1), vec2<i32>(-1, 1),
        vec2<i32>(0, 2), vec2<i32>(2, 0), vec2<i32>(1, 2), vec2<i32>(-1, 2),
        vec2<i32>(2, 1), vec2<i32>(-2, 1), vec2<i32>(2, 2), vec2<i32>(-2, 2),
        vec2<i32>(0, 3), vec2<i32>(3, 0), vec2<i32>(1, 3), vec2<i32>(-1, 3),
        vec2<i32>(3, 1), vec2<i32>(-3, 1), vec2<i32>(2, 3), vec2<i32>(-2, 3),
        vec2<i32>(3, 2), vec2<i32>(-3, 2), vec2<i32>(0, 4), vec2<i32>(4, 0),
        vec2<i32>(1, 4), vec2<i32>(-1, 4), vec2<i32>(4, 1), vec2<i32>(-4, 1),
        vec2<i32>(3, 3), vec2<i32>(-3, 3), vec2<i32>(2, 4), vec2<i32>(-2, 4),
        vec2<i32>(4, 2), vec2<i32>(-4, 2), vec2<i32>(0, 5), vec2<i32>(3, 4),
        vec2<i32>(-3, 4), vec2<i32>(4, 3), vec2<i32>(-4, 3), vec2<i32>(5, 0),
        vec2<i32>(1, 5), vec2<i32>(-1, 5), vec2<i32>(5, 1), vec2<i32>(-5, 1),
        vec2<i32>(2, 5), vec2<i32>(-2, 5), vec2<i32>(5, 2), vec2<i32>(-5, 2),
        vec2<i32>(4, 4), vec2<i32>(-4, 4), vec2<i32>(3, 5), vec2<i32>(-3, 5),
        vec2<i32>(5, 3), vec2<i32>(-5, 3), vec2<i32>(0, 6), vec2<i32>(6, 0),
        vec2<i32>(1, 6), vec2<i32>(-1, 6), vec2<i32>(6, 1), vec2<i32>(-6, 1),
        vec2<i32>(2, 6), vec2<i32>(-2, 6), vec2<i32>(6, 2), vec2<i32>(-6, 2),
        vec2<i32>(4, 5), vec2<i32>(-4, 5), vec2<i32>(5, 4), vec2<i32>(-5, 4),
        vec2<i32>(3, 6), vec2<i32>(-3, 6), vec2<i32>(6, 3), vec2<i32>(-6, 3),
        vec2<i32>(0, 7), vec2<i32>(7, 0), vec2<i32>(1, 7), vec2<i32>(-1, 7),
        vec2<i32>(5, 5), vec2<i32>(-5, 5), vec2<i32>(7, 1), vec2<i32>(-7, 1),
        vec2<i32>(4, 6), vec2<i32>(-4, 6), vec2<i32>(6, 4), vec2<i32>(-6, 4),
        vec2<i32>(2, 7), vec2<i32>(-2, 7), vec2<i32>(7, 2), vec2<i32>(-7, 2),
        vec2<i32>(3, 7), vec2<i32>(-3, 7), vec2<i32>(7, 3), vec2<i32>(-7, 3),
        vec2<i32>(5, 6), vec2<i32>(-5, 6), vec2<i32>(6, 5), vec2<i32>(-6, 5),
        vec2<i32>(8, 0), vec2<i32>(4, 7), vec2<i32>(-4, 7), vec2<i32>(7, 4),
        vec2<i32>(-7, 4), vec2<i32>(8, 1), vec2<i32>(8, 2), vec2<i32>(6, 6),
        vec2<i32>(-6, 6), vec2<i32>(8, 3), vec2<i32>(5, 7), vec2<i32>(-5, 7),
        vec2<i32>(7, 5), vec2<i32>(-7, 5), vec2<i32>(8, 4), vec2<i32>(6, 7),
        vec2<i32>(-6, 7), vec2<i32>(7, 6), vec2<i32>(-7, 6), vec2<i32>(8, 5),
        vec2<i32>(7, 7), vec2<i32>(-7, 7), vec2<i32>(8, 6), vec2<i32>(8, 7),
    );
    return distances[index];
}

fn entropy_copy_value() -> u32 {
    var value = entropy_last_value;
    if params.entropy.lz77_window_mask != 0u {
        value = reconstruction_load(
            entropy_window_base() + (entropy_copy_position & params.entropy.lz77_window_mask)
        );
    }
    entropy_copy_position += 1u;
    entropy_copy_remaining -= 1u;
    return value;
}

fn entropy_record_value(value: u32) {
    if params.entropy.lz77_window_mask == 0u {
        entropy_last_value = value;
    } else {
        reconstruction_store(
            entropy_window_base() + (entropy_decoded & params.entropy.lz77_window_mask),
            value,
        );
    }
    entropy_decoded += 1u;
}

fn entropy_read_varint(cluster: u32, distance_multiplier: u32) -> u32 {
    var value = 0u;
    var effective_distance_multiplier = distance_multiplier;
    if modular_descriptor_mode() {
        effective_distance_multiplier = modular_layout_max_width;
    }
    if entropy_metadata(META_LZ_ENABLED) == 0u {
        return entropy_read_clustered(cluster);
    }
    if entropy_copy_remaining != 0u {
        value = entropy_copy_value();
        entropy_record_value(value);
        return value;
    }
    let token = entropy_read_symbol(cluster);
    let minimum_symbol = entropy_metadata(META_LZ_MIN_SYMBOL);
    if token < minimum_symbol {
        let config = entropy_config_offset(cluster);
        value = entropy_read_hybrid(
            token,
            modular_metadata[config],
            modular_metadata[config + 1u],
            modular_metadata[config + 2u],
        );
        entropy_record_value(value);
        return value;
    }
    if entropy_decoded == 0u {
        decode_error = ERROR_LZ77_STATE;
        return 0u;
    }
    let run_value = entropy_read_hybrid(
        token - minimum_symbol,
        entropy_metadata(META_LZ_LENGTH_SPLIT),
        entropy_metadata(META_LZ_LENGTH_MSB),
        entropy_metadata(META_LZ_LENGTH_LSB),
    );
    entropy_copy_remaining = run_value + entropy_metadata(META_LZ_MIN_LENGTH);
    let distance_cluster = entropy_metadata(META_DISTANCE_CLUSTER);
    let distance_token = entropy_read_symbol(distance_cluster);
    let distance_config = entropy_config_offset(distance_cluster);
    var distance = entropy_read_hybrid(
        distance_token,
        modular_metadata[distance_config],
        modular_metadata[distance_config + 1u],
        modular_metadata[distance_config + 2u],
    );
    if effective_distance_multiplier != 0u {
        if distance < 120u {
            let special = entropy_special_distance(distance);
            distance = u32(max(
                special.x + i32(effective_distance_multiplier) * special.y - 1i,
                0i,
            ));
        } else {
            distance -= 120u;
        }
    }
    distance = min(min(distance, 0xfffffu) + 1u, entropy_decoded);
    entropy_copy_position = entropy_decoded - distance;
    value = entropy_copy_value();
    entropy_record_value(value);
    return value;
}
