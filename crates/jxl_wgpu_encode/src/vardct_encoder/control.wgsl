// Shared checked artifact ABI and deterministic entropy/control writers.

// Exactly 828 bytes. All artifact offsets and lengths are expressed in u32
// words and are independently checked by the host before dispatch.
struct Params {
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    strategy: u32,
    global_scale: u32,
    quant_lf: u32,
    hf_multiplier: u32,
    raw_prefix: array<PrefixEntry, 33>,
    strategy_offset: u32,
    dc_offset: u32,
    token_offset: u32,
    extra_offset: u32,
    fragment_offset: u32,
    fragment_word_capacity: u32,
    artifact_words: u32,
    topology: u32,
    fragment_descriptor_offset: u32,
    fragment_descriptor_len: u32,
    lf_groups_x: u32,
    lf_groups_y: u32,
    lf_quantization: array<f32, 3>,
    lf_correlation: array<f32, 2>,
    hf_prefix: array<PrefixEntry, 33>,
    hf_correlation: array<f32, 2>,
    hf_quantization: array<f32, 3>,
    ac_descriptor_offset: u32,
    ac_descriptor_len: u32,
    ac_fragment_offset: u32,
    ac_words_per_block: u32,
    ac_fragment_words: u32,
    workgroups_x: u32,
    ac_pass_count: u32,
    ac_pass_words: u32,
    progressive: array<u32, 11>,
    saliency_offset: u32,
    saliency_groups: u32,
    color_normalization: u32,
    source_sample_mask: u32,
    source_exponent_bits: u32,
    source_validation_offset: u32,
    source_validation_groups: u32,
    source_big_endian: u32,
    sources: array<Source, 3>,
}

@group(0) @binding(0)
var<storage, read> source_words: array<u32>;
@group(0) @binding(12) var<storage, read> source_words_1: array<u32>;
@group(0) @binding(13) var<storage, read> source_words_2: array<u32>;
@group(0) @binding(14) var<storage, read> source_words_3: array<u32>;

@group(0) @binding(1)
var<storage, read> params: Params;

@group(0) @binding(2)
var<storage, read_write> artifact_words: array<u32>;

override wg_x: u32 = 64u;

const ARTIFACT_READY: u32 = 0x56444354u;
const HEADER_HISTOGRAM_OFFSET: u32 = 22u;

fn append_fragment_bits(value: u32, count: u32, start: u32) -> u32 {
    let capacity_bits = params.fragment_word_capacity * 32u;
    for (var index = 0u; index < count; index += 1u) {
        let bit_offset = start + index;
        if bit_offset < capacity_bits {
            let word_index = params.fragment_offset + (bit_offset >> 5u);
            let bit = (value >> index) & 1u;
            artifact_words[word_index] |= bit << (bit_offset & 31u);
        }
    }
    return start + count;
}

fn encode_dc_token(slot: u32, signed_value: i32, start: u32) -> u32 {
    let value = zigzag_signed(signed_value);
    var token = 0u;
    var extra_bit_count = 0u;
    var extra = 0u;
    if value != 0u {
        extra_bit_count = 31u - countLeadingZeros(value);
        token = extra_bit_count + 1u;
        extra = value - (1u << extra_bit_count);
    }
    artifact_words[params.token_offset + slot] = token;
    artifact_words[params.extra_offset + slot] = extra;
    if token < 33u {
        artifact_words[HEADER_HISTOGRAM_OFFSET + token] += 1u;
        let prefix = params.raw_prefix[token];
        let after_prefix = append_fragment_bits(prefix.bits, prefix.bit_len, start);
        return append_fragment_bits(extra, extra_bit_count, after_prefix);
    }
    return params.fragment_word_capacity * 32u + 1u;
}

fn append_ac_bits(base: u32, capacity: u32, value: u32, count: u32, start: u32) -> u32 {
    for (var index = 0u; index < count; index += 1u) {
        let bit_offset = start + index;
        if bit_offset < capacity * 32u {
            artifact_words[base + (bit_offset >> 5u)] |=
                ((value >> index) & 1u) << (bit_offset & 31u);
        }
    }
    return start + count;
}

fn encode_ac_unsigned(base: u32, capacity: u32, value: u32, start: u32) -> u32 {
    var token = 0u;
    var extra_count = 0u;
    var extra = 0u;
    if value != 0u {
        extra_count = 31u - countLeadingZeros(value);
        token = extra_count + 1u;
        extra = value - (1u << extra_count);
    }
    if token >= 33u {
        return capacity * 32u + 1u;
    }
    let prefix = params.hf_prefix[token];
    let after_prefix = append_ac_bits(base, capacity, prefix.bits, prefix.bit_len, start);
    return append_ac_bits(base, capacity, extra, extra_count, after_prefix);
}

@compute @workgroup_size(1)
fn serialize_control() {
    var error = 0u;
    for (var index = 0u; index < params.ac_descriptor_len; index += 1u) {
        error |= artifact_words[params.ac_descriptor_offset + index]
            & (LF_QUANTIZATION_OVERFLOW | HF_QUANTIZATION_OVERFLOW | NON_FINITE_SOURCE);
    }
    for (var group = 0u; group < params.source_validation_groups; group += 1u) {
        let status = artifact_words[params.source_validation_offset + group];
        if status != SOURCE_VALIDATED && status != (SOURCE_VALIDATED | NON_FINITE_SOURCE) {
            error |= SOURCE_VALIDATION_INCOMPLETE;
        }
        error |= status & NON_FINITE_SOURCE;
    }
    if error != 0u {
        artifact_words[0] = error;
        return;
    }
    let block_count = params.blocks_x * params.blocks_y;
    let sample_count = block_count * 3u;
    var bit_offset = 0u;
    let lf_group_count = params.lf_groups_x * params.lf_groups_y;

    for (var block = 0u; params.topology != 2u && block < block_count; block += 1u) {
        let is_first = block == 0u || params.topology == 1u;
        artifact_words[params.strategy_offset + block] =
            params.strategy | select(0u, 1u << 8u, is_first);
    }
    for (var group = 0u; group < lf_group_count; group += 1u) {
        let group_x = group % params.lf_groups_x;
        let group_y = group / params.lf_groups_x;
        let origin_x = group_x * 256u;
        let origin_y = group_y * 256u;
        let group_width = min(256u, params.blocks_x - origin_x);
        let group_height = min(256u, params.blocks_y - origin_y);
        let group_bit_offset = bit_offset;
        for (var channel = 0u; channel < 3u; channel += 1u) {
            let base = channel * block_count;
            for (var local_y = 0u; local_y < group_height; local_y += 1u) {
                for (var local_x = 0u; local_x < group_width; local_x += 1u) {
                    let block =
                        (origin_y + local_y) * params.blocks_x + origin_x + local_x;
                    var left = 0;
                    if local_x > 0u {
                        left = bitcast<i32>(
                            artifact_words[params.dc_offset + base + block - 1u],
                        );
                    } else if local_y > 0u {
                        left = bitcast<i32>(
                            artifact_words[
                                params.dc_offset + base + block - params.blocks_x
                            ],
                        );
                    }
                    var top = left;
                    if local_y > 0u {
                        top = bitcast<i32>(
                            artifact_words[
                                params.dc_offset + base + block - params.blocks_x
                            ],
                        );
                    }
                    var top_left = left;
                    if local_x > 0u && local_y > 0u {
                        top_left = bitcast<i32>(
                            artifact_words[
                                params.dc_offset + base + block - params.blocks_x - 1u
                            ],
                        );
                    }
                    let actual = bitcast<i32>(
                        artifact_words[params.dc_offset + base + block],
                    );
                    bit_offset = encode_dc_token(
                        base + block,
                        actual - clamped_gradient(top, left, top_left),
                        bit_offset,
                    );
                }
            }
        }
        artifact_words[params.fragment_descriptor_offset + 2u * group] = group_bit_offset;
        artifact_words[params.fragment_descriptor_offset + 2u * group + 1u] =
            bit_offset - group_bit_offset;
    }

    // Header ABI. Status is written last so a mapped ready record cannot
    // expose partially initialized live counts or layout metadata.
    artifact_words[1] = block_count;
    artifact_words[2] = sample_count;
    artifact_words[3] = params.strategy;
    artifact_words[4] = u32(params.ac_descriptor_len != 0u);
    artifact_words[5] = params.strategy_offset;
    artifact_words[6] = block_count;
    artifact_words[7] = params.dc_offset;
    artifact_words[8] = sample_count;
    artifact_words[9] = params.token_offset;
    artifact_words[10] = sample_count;
    artifact_words[11] = params.extra_offset;
    artifact_words[12] = sample_count;
    artifact_words[13] = params.fragment_offset;
    artifact_words[14] = params.fragment_word_capacity;
    artifact_words[15] = bit_offset;
    artifact_words[16] = params.artifact_words;
    artifact_words[17] = params.width;
    artifact_words[18] = params.height;
    artifact_words[19] = params.blocks_x;
    artifact_words[20] = params.blocks_y;
    artifact_words[21] = params.topology;
    artifact_words[55] = params.fragment_descriptor_offset;
    artifact_words[56] = params.fragment_descriptor_len;
    artifact_words[57] = params.lf_groups_x;
    artifact_words[58] = params.lf_groups_y;
    artifact_words[59] = lf_group_count;
    artifact_words[60] = params.ac_descriptor_offset;
    artifact_words[61] = params.ac_descriptor_len;
    artifact_words[62] = params.ac_fragment_offset;
    artifact_words[63] = params.ac_words_per_block;
    artifact_words[64] = params.ac_fragment_words;
    artifact_words[65] = params.ac_pass_count;
    artifact_words[66] = params.saliency_offset;
    artifact_words[67] = params.saliency_groups;
    artifact_words[0] = ARTIFACT_READY;
}

// Split in canonical frequency coordinates, independent of serialized coefficient order.
// Signed division rounds toward zero, including i32::MIN. Earlier contributions
// are subtracted only where their spectral rectangle actually included this value.
fn progressive_value(value: i32, index: u32, width: u32, height: u32, pass_index: u32) -> i32 {
    if params.ac_pass_count == 1u { return value; }
    let columns = max(width, height);
    let rows = min(width, height);
    let x = index % columns;
    let y = index / columns;
    let current = params.progressive[pass_index];
    let square = current & 255u;
    if x >= columns / 8u * square || y >= rows / 8u * square { return 0i; }
    var residual = value;
    for (var previous = 0u; previous < pass_index; previous += 1u) {
        let spec = params.progressive[previous];
        let side = spec & 255u;
        if x < columns / 8u * side && y < rows / 8u * side {
            let step = i32(1u << (spec >> 8u));
            residual -= (residual / step) * step;
        }
    }
    return residual / i32(1u << (current >> 8u));
}
