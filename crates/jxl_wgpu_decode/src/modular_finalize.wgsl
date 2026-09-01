//! Packs inverse-transformed Modular source planes from a resident arena.

override wg_x: u32 = 64u;

struct Params {
    // width, height, source channel count, source bit depth
    extent: vec4<u32>,
    // output x/y origin, status record index, reserved
    region: vec4<u32>,
    source_offsets: vec4<u32>,
    source_strides: vec4<u32>,
    // output kind, transfer, limited range, component count
    output: vec4<u32>,
    // channel order, component bits, storage bits, numeric mapping
    format: vec4<u32>,
    // plane 0 offset/stride, plane 1 offset/stride (bytes)
    plane01: vec4<u32>,
    // plane 2 offset/stride, plane 3 offset/stride (bytes)
    plane23: vec4<u32>,
    // logical output bytes, chroma width, chroma height, reserved
    bounds: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> arena: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
/*__JXL_F64_BINDING__*/

const STATUS_OK: u32 = 1u;
const ERROR_OUTPUT_MAPPING: u32 = 9u;

fn reject_output_mapping() {
    let status_word = params.region.z * 4u;
    if atomicLoad(&status[status_word]) == STATUS_OK {
        atomicStore(&status[status_word], ERROR_OUTPUT_MAPPING);
    }
}

fn write_byte(offset: u32, value: u32) {
    if offset >= params.bounds.x {
        return;
    }
    let word_index = offset >> 2u;
    let shift = (offset & 3u) * 8u;
    let mask = 0xffu << shift;
    var previous = atomicLoad(&output_words[word_index]);
    loop {
        let replacement = (previous & ~mask) | ((value & 0xffu) << shift);
        let exchanged = atomicCompareExchangeWeak(
            &output_words[word_index],
            previous,
            replacement,
        );
        if exchanged.exchanged {
            break;
        }
        previous = exchanged.old_value;
    }
}

fn write_word(offset: u32, value: u32) {
    if (offset & 3u) != 0u || offset > params.bounds.x || params.bounds.x - offset < 4u {
        return;
    }
    atomicStore(&output_words[offset >> 2u], value);
}

fn source_mask() -> u32 {
    return (1u << params.extent.w) - 1u;
}

fn source_sample(channel: u32, x: u32, y: u32) -> u32 {
    let raw = bitcast<i32>(arena[
        params.source_offsets[channel]
            + y * params.source_strides[channel]
            + x
    ]);
    if raw < 0i || u32(raw) > source_mask() {
        reject_output_mapping();
        return 0u;
    }
    return u32(raw);
}

fn write_stored_code(offset: u32, value: u32) {
    if params.format.z == 8u {
        write_byte(offset, value);
    } else {
        write_byte(offset, value);
        write_byte(offset + 1u, value >> 8u);
    }
}

fn write_native_pixel(source_x: u32, source_y: u32, x: u32, y: u32) {
    let bytes_per_component = params.format.z / 8u;
    let pixel_offset = params.plane01.x
        + y * params.plane01.y
        + x * params.extent.z * bytes_per_component;
    for (var channel = 0u; channel < params.extent.z; channel += 1u) {
        write_stored_code(
            pixel_offset + channel * bytes_per_component,
            source_sample(channel, source_x, source_y),
        );
    }
}

fn normalized_unsigned(sample: u32, bits: u32) -> u32 {
    if bits == 8u {
        return sample;
    }
    if bits == 16u {
        return sample * 257u;
    }
    return sample * 16843009u;
}

fn normalized_signed_nonnegative(sample: u32, bits: u32) -> u32 {
    let remainder = (sample * 127u + 127u) / 255u;
    if bits == 8u {
        return remainder;
    }
    if bits == 16u {
        return sample * 128u + remainder;
    }
    return sample * 8421504u + remainder;
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
        remaining >>= 1u;
        leading_bit += 1u;
    }
    let numerator = sample << (31u - leading_bit);
    let significand = (numerator + 127u) / 255u;
    return ((leading_bit + 119u) << 23u) | (significand & 0x007fffffu);
}

fn widen_normalized_f32_to_f64_words(bits: u32) -> vec2<u32> {
    if bits == 0u {
        return vec2<u32>(0u, 0u);
    }
    let exponent = (bits >> 23u) & 0xffu;
    let fraction = bits & 0x007fffffu;
    return vec2<u32>(
        (fraction & 0x7u) << 29u,
        ((exponent + 896u) << 20u) | (fraction >> 3u),
    );
}

fn write_numeric_sample(x: u32, y: u32, sample: u32) {
    let bytes_per_component = params.format.y / 8u;
    let pixel_offset = params.plane01.x
        + y * params.plane01.y
        + x * params.output.w * bytes_per_component;
    for (var component = 0u; component < params.output.w; component += 1u) {
        let offset = pixel_offset + component * bytes_per_component;
        if params.output.x == 0u {
            let value = normalized_unsigned(sample, params.format.y);
            if params.format.y == 8u {
                write_byte(offset, value);
            } else if params.format.y == 16u {
                write_byte(offset, value);
                write_byte(offset + 1u, value >> 8u);
            } else {
                write_word(offset, value);
            }
        } else if params.output.x == 7u {
            let value = normalized_signed_nonnegative(sample, params.format.y);
            if params.format.y == 8u {
                write_byte(offset, value);
            } else if params.format.y == 16u {
                write_byte(offset, value);
                write_byte(offset + 1u, value >> 8u);
            } else {
                write_word(offset, value);
            }
        } else {
            let normalized_bits = normalized_gray8_f32_bits(sample);
            if params.format.y == 32u {
                write_word(offset, normalized_bits);
            } else {
                /*__JXL_F64_OUTPUT__*/
            }
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
    if params.output.y == 0u {
        return encoded;
    }
    let linear = srgb_to_linear(encoded);
    if params.output.y == 2u {
        return linear;
    }
    if linear < 0.018 {
        return 4.5 * linear;
    }
    return 1.099 * pow(linear, 0.45) - 0.099;
}

fn color_code(value: u32) -> u32 {
    let nonlinear = clamp(target_nonlinear(value), 0.0, 1.0);
    let maximum = f32((1u << params.format.y) - 1u);
    var code = maximum * nonlinear;
    if params.output.z != 0u {
        let scale = f32(1u << (params.format.y - 8u));
        code = scale * (16.0 + 219.0 * nonlinear);
    }
    return u32(clamp(round(code), 0.0, maximum));
}

fn neutral_chroma_code() -> u32 {
    return 1u << (params.format.y - 1u);
}

fn stored_rgb_code(position: u32, gray: u32) -> u32 {
    var canonical = position;
    if (params.format.x == 1u || params.format.x == 3u) && position < 3u {
        canonical = 2u - position;
    }
    if canonical == 3u {
        return 255u;
    }
    return gray;
}

fn write_gray_pixel(x: u32, y: u32, sample: u32) {
    if params.output.x == 0u || params.output.x == 7u || params.output.x == 8u {
        write_numeric_sample(x, y, sample);
    } else if params.output.x == 1u || params.output.x == 2u || params.output.x == 3u {
        let bytes_per_sample = params.format.z / 8u;
        write_stored_code(
            params.plane01.x + y * params.plane01.y + x * bytes_per_sample,
            color_code(sample),
        );
    } else if params.output.x == 5u {
        let gray = u32(clamp(round(255.0 * target_nonlinear(sample)), 0.0, 255.0));
        let offset = params.plane01.x + y * params.plane01.y + x * params.output.w;
        for (var position = 0u; position < params.output.w; position += 1u) {
            write_byte(offset + position, stored_rgb_code(position, gray));
        }
    } else if params.output.x == 6u {
        let gray = u32(clamp(round(255.0 * target_nonlinear(sample)), 0.0, 255.0));
        write_byte(params.plane01.x + y * params.plane01.y + x, stored_rgb_code(0u, gray));
        write_byte(params.plane01.z + y * params.plane01.w + x, stored_rgb_code(1u, gray));
        write_byte(params.plane23.x + y * params.plane23.y + x, stored_rgb_code(2u, gray));
        if params.output.w == 4u {
            write_byte(params.plane23.z + y * params.plane23.w + x, 255u);
        }
    }
}

fn write_chroma(x: u32, y: u32) {
    if x >= params.bounds.y || y >= params.bounds.z {
        return;
    }
    let bytes_per_sample = params.format.z / 8u;
    let neutral = neutral_chroma_code();
    if params.output.x == 2u {
        let offset = params.plane01.z
            + y * params.plane01.w
            + x * 2u * bytes_per_sample;
        write_stored_code(offset, neutral);
        write_stored_code(offset + bytes_per_sample, neutral);
    } else if params.output.x == 3u {
        write_stored_code(
            params.plane01.z + y * params.plane01.w + x * bytes_per_sample,
            neutral,
        );
        write_stored_code(
            params.plane23.x + y * params.plane23.y + x * bytes_per_sample,
            neutral,
        );
    }
}

fn write_packed_422(source_x: u32, source_y: u32, x: u32, y: u32) {
    if (x & 1u) != 0u {
        return;
    }
    let source_x1 = min(source_x + 1u, params.extent.x - 1u);
    let y0 = color_code(source_sample(0u, source_x, source_y));
    let y1 = color_code(source_sample(0u, source_x1, source_y));
    let neutral = neutral_chroma_code();
    var packed = y0 | (neutral << 8u) | (y1 << 16u) | (neutral << 24u);
    if params.format.x == 1u {
        packed = neutral | (y0 << 8u) | (neutral << 16u) | (y1 << 24u);
    }
    write_word(params.plane01.x + y * params.plane01.y + (x >> 1u) * 4u, packed);
}

@compute @workgroup_size(wg_x, 1, 1)
fn finalize(@builtin(global_invocation_id) id: vec3<u32>) {
    if atomicLoad(&status[params.region.z * 4u]) != STATUS_OK {
        return;
    }
    if id.x >= params.extent.x || id.y >= params.extent.y {
        return;
    }
    let source_x = id.x;
    let source_y = id.y;
    let x = params.region.x + source_x;
    let y = params.region.y + source_y;
    if params.output.x == 9u {
        write_native_pixel(source_x, source_y, x, y);
        return;
    }
    if params.output.x == 4u {
        write_packed_422(source_x, source_y, x, y);
        return;
    }
    let sample = source_sample(0u, source_x, source_y);
    write_gray_pixel(x, y, sample);
    write_chroma(x, y);
}
