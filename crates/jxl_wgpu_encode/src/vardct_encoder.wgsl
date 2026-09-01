// One VarDCT strategy, up to a 32x32-pixel footprint. The shader owns every
// data-dependent operation: input normalization, transfer function, XYB,
// forward transform, per-8x8 DC quantization/prediction, strategy-map
// generation, AC quantization, signed tokenization, histogramming, and prefix bit packing.

struct PrefixEntry {
    bits: u32,
    bit_len: u32,
}

// Exactly 512 bytes. Keeping the control ABI on two common storage-buffer
// alignment quantum makes the record safe to suballocate in later batching.
struct Params {
    row_stride: u32,
    byte_offset: u32,
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    strategy: u32,
    global_scale: u32,
    quant_lf: u32,
    dc_prefix: array<PrefixEntry, 19>,
    hf_prefix: array<PrefixEntry, 19>,
    lf_quantization: array<f32, 3>,
    lf_correlation: array<f32, 2>,
    hf_correlation: array<f32, 2>,
    hf_quantization: array<f32, 3>,
    padding: array<u32, 33>,
}

// Exactly 26.25 KiB. The first 2304 bytes are 256-byte suballocation aligned
// control/entropy sections. Fixed maxima cover at most 16 DC blocks, one complete
// DCT8 pass-group fragment, and 1024 diagnostic coefficients per channel.
struct Artifact {
    strategy_map: array<u32, 16>,
    quantized_dc_yxb: array<i32, 48>,
    dc_raw_tokens: array<u32, 48>,
    dc_extra_bits: array<u32, 48>,
    dc_fragment_words: array<u32, 64>,
    dc_fragment_bit_len: u32,
    dc_sample_count: u32,
    block_count: u32,
    strategy: u32,
    raw_histogram: array<u32, 19>,
    dc_padding: array<u32, 9>,
    ac_fragment_words: array<u32, 256>,
    ac_fragment_bit_len: u32,
    ac_token_count: u32,
    ac_histogram: array<u32, 19>,
    ac_padding: array<u32, 43>,
    forward_xyb_bits: array<u32, 3072>,
    quantized_xyb: array<i32, 3072>,
}

@group(0) @binding(0)
var<storage, read> source_words: array<u32>;

@group(0) @binding(1)
var<storage, read> params: Params;

@group(0) @binding(2)
var<storage, read_write> artifact: Artifact;

// array<vec3<f32>> has a 16-byte stride in WGSL, so this consumes exactly the
// portable 16 KiB workgroup-storage floor at the 32x32 maximum.
var<workgroup> block_xyb: array<vec3<f32>, 1024>;

override wg_x: u32 = 256u;

const PI: f32 = 3.14159265358979323846;
const SQRT_TWO: f32 = 1.41421356237309504880;
const OPSIN_BIAS: f32 = 0.0037930732552754493;
const NEG_OPSIN_BIAS_CBRT: f32 = -0.15595420054924863;
const MAX_BLOCKS: u32 = 16u;
const MAX_COEFFICIENTS: u32 = 1024u;
const MAX_FRAGMENT_BITS: u32 = 2048u;
const MAX_AC_FRAGMENT_BITS: u32 = 8192u;
const MAX_HF_QUANTIZED_MAGNITUDE: i32 = 131071;
const DCT8_NATURAL_ORDER: array<u32, 64> = array<u32, 64>(
    0u, 1u, 8u, 16u, 9u, 2u, 3u, 10u, 17u, 24u, 32u, 25u, 18u, 11u, 4u, 5u,
    12u, 19u, 26u, 33u, 40u, 48u, 41u, 34u, 27u, 20u, 13u, 6u, 7u, 14u, 21u, 28u,
    35u, 42u, 49u, 56u, 57u, 50u, 43u, 36u, 29u, 22u, 15u, 23u, 30u, 37u, 44u, 51u,
    58u, 59u, 52u, 45u, 38u, 31u, 39u, 46u, 53u, 60u, 61u, 54u, 47u, 55u, 62u, 63u,
);

fn load_u8(byte_address: u32) -> u32 {
    let word = source_words[byte_address >> 2u];
    return (word >> ((byte_address & 3u) * 8u)) & 255u;
}

fn srgb_to_linear(encoded: f32) -> f32 {
    if encoded <= 0.04045 {
        return encoded / 12.92;
    }
    return pow((encoded + 0.055) / 1.055, 2.4);
}

fn linear_rgb_to_xyb(rgb: vec3<f32>) -> vec3<f32> {
    let mixed = max(
        vec3<f32>(
            0.3000000000 * rgb.x + 0.6220000000 * rgb.y + 0.0780000000 * rgb.z,
            0.2300000000 * rgb.x + 0.6920000000 * rgb.y + 0.0780000000 * rgb.z,
            0.2434226892 * rgb.x + 0.2047674442 * rgb.y + 0.5518098665 * rgb.z,
        ) + vec3<f32>(OPSIN_BIAS),
        vec3<f32>(0.0),
    );
    let absorbance = vec3<f32>(
        pow(mixed.x, 1.0 / 3.0) + NEG_OPSIN_BIAS_CBRT,
        pow(mixed.y, 1.0 / 3.0) + NEG_OPSIN_BIAS_CBRT,
        pow(mixed.z, 1.0 / 3.0) + NEG_OPSIN_BIAS_CBRT,
    );
    return vec3<f32>(
        0.5 * (absorbance.x - absorbance.y),
        0.5 * (absorbance.x + absorbance.y),
        absorbance.z,
    );
}

fn dct_basis(frequency: u32, position: u32, size: u32) -> f32 {
    if frequency == 0u {
        return 1.0;
    }
    return SQRT_TWO * cos(
        f32((2u * position + 1u) * frequency) * PI / (2.0 * f32(size)),
    );
}

fn dct8_weight_parameter(channel: u32, index: u32) -> f32 {
    if channel == 0u {
        return array<f32, 6>(3150.0, 0.0, -0.4, -0.4, -0.4, -2.0)[index];
    }
    if channel == 1u {
        return array<f32, 6>(560.0, 0.0, -0.3, -0.3, -0.3, -0.3)[index];
    }
    return array<f32, 6>(512.0, -2.0, -1.0, 0.0, -1.0, -2.0)[index];
}

fn dct8_quant_weight(channel: u32, frequency_x: u32, frequency_y: u32) -> f32 {
    var bands: array<f32, 6>;
    bands[0] = dct8_weight_parameter(channel, 0u);
    for (var index = 1u; index < 6u; index += 1u) {
        let parameter = dct8_weight_parameter(channel, index);
        let multiplier = select(1.0 / (1.0 - parameter), 1.0 + parameter, parameter > 0.0);
        bands[index] = bands[index - 1u] * multiplier;
    }
    let dx = f32(frequency_x) / 7.0;
    let dy = f32(frequency_y) / 7.0;
    let scaled_position = sqrt(dx * dx + dy * dy) * 5.0 / (SQRT_TWO + 1e-6);
    let band_index = min(u32(scaled_position), 4u);
    let fraction = scaled_position - f32(band_index);
    let lower = bands[band_index];
    let upper = bands[band_index + 1u];
    return lower * pow(upper / lower, fraction);
}

fn quantize_dct8_ac(coefficient: vec3<f32>, frequency_x: u32, frequency_y: u32) -> vec3<i32> {
    let decorrelated = vec3<f32>(
        fma(-coefficient.y, params.hf_correlation[0], coefficient.x),
        coefficient.y,
        fma(-coefficient.y, params.hf_correlation[1], coefficient.z),
    );
    let scale = f32(params.global_scale) * 6.0 / 65536.0;
    var quantized = vec3<i32>(0);
    for (var channel = 0u; channel < 3u; channel += 1u) {
        let value = decorrelated[channel]
            * scale
            * params.hf_quantization[channel]
            * dct8_quant_weight(channel, frequency_x, frequency_y);
        quantized[channel] = clamp(
            i32(round(value)),
            -MAX_HF_QUANTIZED_MAGNITUDE,
            MAX_HF_QUANTIZED_MAGNITUDE,
        );
    }
    return quantized;
}

fn zigzag_signed(value: i32) -> u32 {
    if value < 0 {
        return u32(-value) * 2u - 1u;
    }
    return u32(value) * 2u;
}

fn clamped_gradient(top: i32, left: i32, top_left: i32) -> i32 {
    return clamp(top + left - top_left, min(top, left), max(top, left));
}

fn append_fragment_bits(value: u32, count: u32, start: u32) -> u32 {
    for (var index = 0u; index < count; index += 1u) {
        let bit_offset = start + index;
        if bit_offset < MAX_FRAGMENT_BITS {
            let bit = (value >> index) & 1u;
            artifact.dc_fragment_words[bit_offset >> 5u] |= bit << (bit_offset & 31u);
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
    artifact.dc_raw_tokens[slot] = token;
    artifact.dc_extra_bits[slot] = extra;
    if token < 19u {
        artifact.raw_histogram[token] += 1u;
        let prefix = params.dc_prefix[token];
        let after_prefix = append_fragment_bits(prefix.bits, prefix.bit_len, start);
        return append_fragment_bits(extra, extra_bit_count, after_prefix);
    }
    // Preserve an invalid length for host-side validation without performing
    // an out-of-bounds prefix-table access.
    return MAX_FRAGMENT_BITS + 1u;
}

fn append_ac_fragment_bits(value: u32, count: u32, start: u32) -> u32 {
    for (var index = 0u; index < count; index += 1u) {
        let bit_offset = start + index;
        if bit_offset < MAX_AC_FRAGMENT_BITS {
            let bit = (value >> index) & 1u;
            artifact.ac_fragment_words[bit_offset >> 5u] |= bit << (bit_offset & 31u);
        }
    }
    return start + count;
}

fn encode_ac_unsigned(value: u32, start: u32) -> u32 {
    var token = 0u;
    var extra_bit_count = 0u;
    var extra = 0u;
    if value != 0u {
        extra_bit_count = 31u - countLeadingZeros(value);
        token = extra_bit_count + 1u;
        extra = value - (1u << extra_bit_count);
    }
    if token < 19u {
        artifact.ac_histogram[token] += 1u;
        let prefix = params.hf_prefix[token];
        let after_prefix = append_ac_fragment_bits(prefix.bits, prefix.bit_len, start);
        return append_ac_fragment_bits(extra, extra_bit_count, after_prefix);
    }
    return MAX_AC_FRAGMENT_BITS + 1u;
}

fn encode_ac_signed(value: i32, start: u32) -> u32 {
    return encode_ac_unsigned(zigzag_signed(value), start);
}

fn quantized_block_dc(channel: u32, block_index: u32) -> i32 {
    let block_x = block_index % params.blocks_x;
    let block_y = block_index / params.blocks_x;
    var sum = vec3<f32>(0.0);
    for (var y = 0u; y < 8u; y += 1u) {
        for (var x = 0u; x < 8u; x += 1u) {
            let pixel = (block_y * 8u + y) * params.width + block_x * 8u + x;
            sum += block_xyb[pixel];
        }
    }
    let mean = sum / 64.0;
    let dc_scale = f32(params.global_scale * params.quant_lf);
    if channel == 0u {
        return i32(round(mean.y * dc_scale * params.lf_quantization[1]));
    }
    if channel == 1u {
        let decorrelated_x = fma(-mean.y, params.lf_correlation[0], mean.x);
        return i32(round(decorrelated_x * dc_scale * params.lf_quantization[0]));
    }
    let decorrelated_b = fma(-mean.y, params.lf_correlation[1], mean.z);
    return i32(round(decorrelated_b * dc_scale * params.lf_quantization[2]));
}

@compute @workgroup_size(wg_x, 1, 1)
fn encode(@builtin(local_invocation_index) local_index: u32) {
    let pixel_count = params.width * params.height;
    for (var pixel = local_index; pixel < pixel_count; pixel += wg_x) {
        let pixel_x = pixel % params.width;
        let pixel_y = pixel / params.width;
        let pixel_address = params.byte_offset + pixel_y * params.row_stride + pixel_x * 3u;
        let encoded = vec3<f32>(
            f32(load_u8(pixel_address)) / 255.0,
            f32(load_u8(pixel_address + 1u)) / 255.0,
            f32(load_u8(pixel_address + 2u)) / 255.0,
        );
        let linear = vec3<f32>(
            srgb_to_linear(encoded.x),
            srgb_to_linear(encoded.y),
            srgb_to_linear(encoded.z),
        );
        block_xyb[pixel] = linear_rgb_to_xyb(linear);
    }
    workgroupBarrier();

    // Full diagnostic forward transform. The first production spectral path quantizes a
    // standard 8x8 DCT directly into the persistent coefficient/fragment ABI. Other bounded
    // strategies retain their zero-HF profile until their matrix/layout lowering is enabled.
    for (var coefficient_index = local_index;
         coefficient_index < pixel_count;
         coefficient_index += wg_x) {
        let frequency_x = coefficient_index % params.width;
        let frequency_y = coefficient_index / params.width;
        var coefficient = vec3<f32>(0.0);
        for (var pixel = 0u; pixel < pixel_count; pixel += 1u) {
            let x = pixel % params.width;
            let y = pixel / params.width;
            let basis = dct_basis(frequency_x, x, params.width)
                * dct_basis(frequency_y, y, params.height) / f32(pixel_count);
            coefficient += block_xyb[pixel] * basis;
        }
        for (var channel = 0u; channel < 3u; channel += 1u) {
            let offset = channel * MAX_COEFFICIENTS + coefficient_index;
            artifact.forward_xyb_bits[offset] = bitcast<u32>(coefficient[channel]);
        }
        if params.strategy == 0u && coefficient_index != 0u {
            let quantized = quantize_dct8_ac(coefficient, frequency_x, frequency_y);
            // JPEG XL's ComputeScaledDCT<8, 8> wire layout is transposed: the physical
            // coefficient row is the horizontal frequency and the column is the vertical
            // frequency. Keep forward_xyb_bits in mathematical row-major order for diagnostics,
            // but place quantized coefficients in the normative entropy/decoder layout.
            let wire_index = frequency_x * 8u + frequency_y;
            for (var channel = 0u; channel < 3u; channel += 1u) {
                artifact.quantized_xyb[channel * MAX_COEFFICIENTS + wire_index]
                    = quantized[channel];
            }
        }
    }
    storageBarrier();
    workgroupBarrier();

    if local_index == 0u {
        let block_count = params.blocks_x * params.blocks_y;
        artifact.block_count = block_count;
        artifact.dc_sample_count = block_count * 3u;
        artifact.strategy = params.strategy;

        // Bits 0..7 hold the standard codestream strategy ID; bit 8 is the
        // first-block marker. A single transform covers this entire map.
        for (var block = 0u; block < block_count; block += 1u) {
            artifact.strategy_map[block] = params.strategy | select(0u, 1u << 8u, block == 0u);
        }
        for (var index = 0u; index < 64u; index += 1u) {
            artifact.dc_fragment_words[index] = 0u;
        }
        for (var index = 0u; index < 19u; index += 1u) {
            artifact.raw_histogram[index] = 0u;
            artifact.ac_histogram[index] = 0u;
        }
        for (var index = 0u; index < 256u; index += 1u) {
            artifact.ac_fragment_words[index] = 0u;
        }

        var bit_offset = 0u;
        // Standard DC channel order is Y, X, B. Values use one fixed 16-slot
        // row-major plane per channel; only block_count entries are live.
        for (var channel = 0u; channel < 3u; channel += 1u) {
            let channel_base = channel * MAX_BLOCKS;
            for (var block = 0u; block < block_count; block += 1u) {
                let block_x = block % params.blocks_x;
                let block_y = block / params.blocks_x;
                var left = 0;
                if block_x > 0u {
                    left = artifact.quantized_dc_yxb[channel_base + block - 1u];
                } else if block_y > 0u {
                    left = artifact.quantized_dc_yxb[channel_base + block - params.blocks_x];
                }
                var top = left;
                if block_y > 0u {
                    top = artifact.quantized_dc_yxb[channel_base + block - params.blocks_x];
                }
                var top_left = left;
                if block_x > 0u && block_y > 0u {
                    top_left = artifact.quantized_dc_yxb[channel_base + block - params.blocks_x - 1u];
                }
                let actual = quantized_block_dc(channel, block);
                artifact.quantized_dc_yxb[channel_base + block] = actual;
                var xyb_channel = 2u;
                if channel == 0u {
                    xyb_channel = 1u;
                } else if channel == 1u {
                    xyb_channel = 0u;
                }
                artifact.quantized_xyb[xyb_channel * MAX_COEFFICIENTS + block] = actual;
                let residual = actual - clamped_gradient(top, left, top_left);
                bit_offset = encode_dc_token(channel_base + block, residual, bit_offset);
            }
        }
        artifact.dc_fragment_bit_len = bit_offset;

        if params.strategy == 0u {
            var nonzero_counts = array<u32, 3>(0u, 0u, 0u);
            var total_nonzero = 0u;
            for (var channel = 0u; channel < 3u; channel += 1u) {
                var xyb_channel = 2u;
                if channel == 0u {
                    xyb_channel = 1u;
                } else if channel == 1u {
                    xyb_channel = 0u;
                }
                for (var order_index = 1u; order_index < 64u; order_index += 1u) {
                    let coefficient = artifact.quantized_xyb[
                        xyb_channel * MAX_COEFFICIENTS + DCT8_NATURAL_ORDER[order_index]
                    ];
                    if coefficient != 0 {
                        nonzero_counts[channel] += 1u;
                        total_nonzero += 1u;
                    }
                }
            }

            if total_nonzero != 0u {
                var ac_bit_offset = 0u;
                var ac_token_count = 0u;
                for (var channel = 0u; channel < 3u; channel += 1u) {
                    ac_bit_offset = encode_ac_unsigned(nonzero_counts[channel], ac_bit_offset);
                    ac_token_count += 1u;
                    if nonzero_counts[channel] == 0u {
                        continue;
                    }
                    var xyb_channel = 2u;
                    if channel == 0u {
                        xyb_channel = 1u;
                    } else if channel == 1u {
                        xyb_channel = 0u;
                    }
                    var remaining = nonzero_counts[channel];
                    for (var order_index = 1u; order_index < 64u; order_index += 1u) {
                        let coefficient = artifact.quantized_xyb[
                            xyb_channel * MAX_COEFFICIENTS + DCT8_NATURAL_ORDER[order_index]
                        ];
                        ac_bit_offset = encode_ac_signed(coefficient, ac_bit_offset);
                        ac_token_count += 1u;
                        if coefficient != 0 {
                            remaining -= 1u;
                            if remaining == 0u {
                                break;
                            }
                        }
                    }
                }
                artifact.ac_fragment_bit_len = ac_bit_offset;
                artifact.ac_token_count = ac_token_count;
            }
        }
    }
}
