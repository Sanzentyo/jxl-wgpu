override wg_x: u32 = 256u;
override wg_y: u32 = 1u;

struct Params {
    width: u32,
    height: u32,
    source_width: u32,
    source_height: u32,
    r_stride: u32,
    g_stride: u32,
    b_stride: u32,
    kind: u32,
    channels: u32,
    order: u32,
    matrix: u32,
    range: u32,
    siting_x: u32,
    siting_y: u32,
    subsample_x: u32,
    subsample_y: u32,
    bits: u32,
    storage_bits: u32,
    plane0_offset: u32,
    plane0_stride: u32,
    plane1_offset: u32,
    plane1_stride: u32,
    plane2_offset: u32,
    plane2_stride: u32,
    plane3_offset: u32,
    plane3_stride: u32,
    logical_size: u32,
    dispatch_width: u32,
    orientation: u32,
    source_transfer: u32,
    target_transfer: u32,
    identity_color_transform: u32,
    primaries_r: vec4<f32>,
    primaries_g: vec4<f32>,
    primaries_b: vec4<f32>,
    alpha: vec4<u32>, // association conversion, reserved
    transfer_parameters: vec4<f32>, // source gamma, target gamma, black floor, display nits (zero = generic)
    source_luminance: vec4<f32>,
    target_luminance: vec4<f32>,
    tone_mapping: ToneMappingParams,
    gamut_mapping: GamutMappingParams,
};

@group(0) @binding(0) var<storage, read> source_r: array<u32>;
@group(0) @binding(1) var<storage, read> source_g: array<u32>;
@group(0) @binding(2) var<storage, read> source_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

fn source_coordinate(destination: vec2<u32>) -> vec2<u32> {
    return image_source_coordinate(
        destination, vec2<u32>(params.source_width, params.source_height), params.orientation,
    );
}

fn source_rgb_at(x: u32, y: u32) -> vec3<f32> {
    return bitcast<vec3<f32>>(source_rgb_words_at(x, y));
}

fn source_alpha_at(x: u32, y: u32) -> f32 {
    return bitcast<f32>(source_alpha_word_at(x, y));
}

fn target_linear_rgb_at(x: u32, y: u32) -> vec3<f32> {
    let source = source_rgb_at(x, y);
    let source_linear = display_to_linear(
        source, params.source_transfer, params.transfer_parameters.x,
        params.transfer_parameters.w, params.source_luminance,
    );
    var converted_linear = vec3<f32>(
        dot(params.primaries_r.xyz, source_linear),
        dot(params.primaries_g.xyz, source_linear),
        dot(params.primaries_b.xyz, source_linear),
    );
    let threshold = params.transfer_parameters.z;
    if threshold >= 0.0 { converted_linear = select(converted_linear, vec3<f32>(0.0), converted_linear <= vec3<f32>(threshold)); }
    let mapped = tone_map_light(converted_linear, params.target_luminance.xyz,
        vec3<f32>(1.0), params.tone_mapping);
    if params.tone_mapping.range.w == 5.0 || (params.tone_mapping.range.z > 0.0
        && dot(converted_linear, params.target_luminance.xyz) < params.tone_mapping.range.z) {
        return mapped;
    }
    return gamut_map_rgb(mapped, params.gamut_mapping);
}

fn target_intensity() -> f32 {
    if params.tone_mapping.range.w != 0.0 { return params.tone_mapping.range.y; }
    return params.transfer_parameters.w;
}

fn target_rgb_at(x: u32, y: u32) -> vec3<f32> {
    if identity_output_color() { return source_rgb_at(x, y); }
    var linear = target_linear_rgb_at(x, y);
    if params.order == 5u {
        // Gray is target-linear luminance after tone/gamut mapping, not encoded RGB luma.
        linear = vec3<f32>(dot(linear, params.gamut_mapping.luminance.xyz));
    }
    return display_from_linear(
        linear, params.target_transfer, params.transfer_parameters.y,
        target_intensity(), params.target_luminance,
    );
}

fn identity_output_color() -> bool {
    return params.order != 5u && params.identity_color_transform != 0u
        && (params.gamut_mapping.luminance.w < 0.0 || params.tone_mapping.range.w == 5.0);
}

fn rgb_at(x: u32, y: u32) -> vec3<f32> {
    let color = target_rgb_at(x, y);
    if params.alpha.x == 0u { return color; }
    return color * output_alpha_multiplier_at(x, y);
}

fn output_alpha_multiplier_at(x: u32, y: u32) -> f32 {
    // Packed 4:2:2 duplicates the final luma at odd widths. Its alpha must use that same
    // clamped output coordinate before orientation, including on a one-pixel axis.
    return image_alpha_multiplier(source_alpha_at(min(x, params.width - 1u), min(y, params.height - 1u)), params.alpha.x);
}

fn coefficients() -> vec2<f32> {
    switch params.matrix {
        case 0u: { return vec2<f32>(0.299, 0.114); }
        case 1u: { return vec2<f32>(0.2126, 0.0722); }
        default: { return vec2<f32>(0.2627, 0.0593); }
    }
}

fn rgb_to_yuv(rgb: vec3<f32>) -> vec3<f32> {
    let coefficient = coefficients();
    let kr = coefficient.x;
    let kb = coefficient.y;
    let kg = 1.0 - kr - kb;
    let y = kr * rgb.r + kg * rgb.g + kb * rgb.b;
    return vec3<f32>(
        y,
        (rgb.b - y) / (2.0 * (1.0 - kb)) + 0.5,
        (rgb.r - y) / (2.0 * (1.0 - kr)) + 0.5,
    );
}

fn yuv_at(x: u32, y: u32) -> vec3<f32> {
    if params.matrix != 3u {
        return rgb_to_yuv(rgb_at(x, y));
    }
    var linear = target_linear_rgb_at(x, y);
    var encoded = display_from_linear(
        linear, params.target_transfer, params.transfer_parameters.y,
        target_intensity(), params.target_luminance,
    );
    if params.alpha.x != 0u {
        encoded *= output_alpha_multiplier_at(x, y);
        var forward_luminance = params.target_luminance;
        forward_luminance.w = 1.0 / (1.0 + forward_luminance.w) - 1.0;
        linear = display_to_linear(
            encoded, params.target_transfer, params.transfer_parameters.y,
            target_intensity(), forward_luminance,
        );
    }
    let coefficient = coefficients();
    let kr = coefficient.x;
    let kb = coefficient.y;
    let kg = 1.0 - kr - kb;
    let y_encoded = display_from_linear(
        vec3<f32>(kr * linear.r + kg * linear.g + kb * linear.b),
        params.target_transfer, params.transfer_parameters.y,
        target_intensity(), params.target_luminance,
    ).x;
    let cb_divisor = select(1.9404, 1.5816, encoded.b > y_encoded);
    let cr_divisor = select(1.7184, 0.9936, encoded.r > y_encoded);
    return vec3<f32>(
        y_encoded,
        (encoded.b - y_encoded) / cb_divisor + 0.5,
        (encoded.r - y_encoded) / cr_divisor + 0.5,
    );
}

fn chroma_at(cx: u32, cy: u32) -> vec2<f32> {
    let origin_x = cx * params.subsample_x;
    let origin_y = cy * params.subsample_y;
    let centered_x = params.siting_x == 0u && params.subsample_x > 1u;
    let centered_y = params.siting_y == 0u && params.subsample_y > 1u;
    if !centered_x && !centered_y {
        return yuv_at(origin_x, origin_y).yz;
    }
    let count_x = select(1u, params.subsample_x, centered_x);
    let count_y = select(1u, params.subsample_y, centered_y);
    var sum = vec2<f32>(0.0);
    var count = 0u;
    for (var dy = 0u; dy < count_y; dy += 1u) {
        for (var dx = 0u; dx < count_x; dx += 1u) {
            let x = origin_x + dx;
            let y = origin_y + dy;
            if x < params.width && y < params.height {
                sum += yuv_at(x, y).yz;
                count += 1u;
            }
        }
    }
    return sum / f32(count);
}

fn quantize8(value: f32, component: u32) -> u32 {
    var code: f32;
    if params.range == 0u {
        code = 255.0 * value;
    } else if component == 0u {
        code = 16.0 + 219.0 * value;
    } else {
        code = 128.0 + 224.0 * (value - 0.5);
    }
    return u32(clamp(floor(code + 0.5), 0.0, 255.0));
}

fn quantize16(value: f32) -> u32 {
    var code = 65535.0 * value;
    if params.range == 1u {
        code = 4096.0 + 56064.0 * value;
    }
    return u32(clamp(floor(code + 0.5), 0.0, 65535.0));
}

fn quantize_code(value: f32, component: u32) -> u32 {
    let maximum = f32((1u << params.bits) - 1u);
    var code = maximum * value;
    if params.range == 1u {
        let scale = f32(1u << (params.bits - 8u));
        if component == 0u {
            code = scale * (16.0 + 219.0 * value);
        } else {
            code = scale * (128.0 + 224.0 * (value - 0.5));
        }
    }
    return u32(clamp(floor(code + 0.5), 0.0, maximum));
}

fn stored_code_byte(code: u32, byte: u32) -> u32 {
    let stored = code << (params.storage_bits - params.bits);
    return (stored >> (byte * 8u)) & 0xffu;
}

fn plane_local(index: u32, offset: u32, stride: u32) -> vec2<u32> {
    let local = index - offset;
    return vec2<u32>(local % stride, local / stride);
}

// The last row ends at its payload, not at the next row stride. A following plane may start
// inside that unused tail. Subtraction/division also avoids overflowing offset + stride * rows.
fn plane_contains(index: u32, offset: u32, stride: u32, row_bytes: u32, height: u32) -> bool {
    if index < offset { return false; }
    let local = index - offset;
    let row = local / stride;
    return row < height && (row + 1u < height || local % stride < row_bytes);
}

fn rgb_component_at(x: u32, y: u32, component: u32) -> f32 {
    if component == 3u { return source_alpha_at(x, y); }
    let rgb = rgb_at(x, y);
    if component == 0u { return rgb.r; }
    if component == 1u { return rgb.g; }
    if component == 2u { return rgb.b; }
    return 1.0;
}

fn rgb_byte_at(x: u32, y: u32, component: u32, byte: u32) -> u32 {
    if params.bits == 32u {
        var word: u32;
        if component == 3u {
            word = source_alpha_word_at(x, y);
        } else if identity_output_color() && params.alpha.x == 0u {
            // A float round trip may flush subnormals or change NaN/zero representations.
            // Preserve an unchanged F32 component as an integer word through packing.
            word = source_rgb_words_at(x, y)[component];
        } else {
            word = bitcast<u32>(rgb_component_at(x, y, component));
        }
        return (word >> (byte * 8u)) & 0xffu;
    }
    return quantize8(rgb_component_at(x, y, component), 0u);
}

fn stored_rgb_component(position: u32) -> u32 {
    if params.order == 4u || params.order == 5u { return select(0u, 3u, position == 1u); }
    if (params.order == 1u || params.order == 3u) && position < 3u {
        return 2u - position;
    }
    return position;
}

fn byte_at(index: u32) -> u32 {
    if index >= params.logical_size { return 0u; }

    let rgb_sample_bytes = params.storage_bits / 8u;
    // Interleaved RGB8 / RGB F32.
    if params.kind == 0u && index >= params.plane0_offset {
        let local = plane_local(index, params.plane0_offset, params.plane0_stride);
        if local.y < params.height && local.x < params.width * params.channels * rgb_sample_bytes {
            let sample = local.x / rgb_sample_bytes;
            let pixel = sample / params.channels;
            let component = stored_rgb_component(sample % params.channels);
            return rgb_byte_at(pixel, local.y, component, local.x % rgb_sample_bytes);
        }
        return 0u;
    }

    // Planar RGB8 / RGB F32. Plane index is also stored channel position.
    if params.kind == 1u {
        var plane = 4u;
        var local = vec2<u32>(0u);
        let row_bytes = params.width * rgb_sample_bytes;
        if plane_contains(index, params.plane0_offset, params.plane0_stride, row_bytes, params.height) {
            plane = 0u; local = plane_local(index, params.plane0_offset, params.plane0_stride);
        } else if params.channels > 1u && plane_contains(index, params.plane1_offset, params.plane1_stride, row_bytes, params.height) {
            plane = 1u; local = plane_local(index, params.plane1_offset, params.plane1_stride);
        } else if params.channels > 2u && plane_contains(index, params.plane2_offset, params.plane2_stride, row_bytes, params.height) {
            plane = 2u; local = plane_local(index, params.plane2_offset, params.plane2_stride);
        } else if params.channels == 4u && plane_contains(index, params.plane3_offset, params.plane3_stride, row_bytes, params.height) {
            plane = 3u; local = plane_local(index, params.plane3_offset, params.plane3_stride);
        }
        if plane < params.channels && local.x < row_bytes && local.y < params.height {
            return rgb_byte_at(local.x / rgb_sample_bytes, local.y, stored_rgb_component(plane), local.x % rgb_sample_bytes);
        }
        return 0u;
    }

    // Luma-only Y8/Y16.
    if (params.kind == 2u || params.kind == 3u) && index >= params.plane0_offset {
        let local = plane_local(index, params.plane0_offset, params.plane0_stride);
        let bytes_per_sample = select(1u, 2u, params.kind == 3u);
        if local.y < params.height && local.x < params.width * bytes_per_sample {
            let pixel = local.x / bytes_per_sample;
            let y = yuv_at(pixel, local.y).x;
            if params.kind == 2u { return quantize8(y, 0u); }
            let code = quantize16(y);
            return (code >> ((local.x & 1u) * 8u)) & 0xffu;
        }
        return 0u;
    }

    // Packed YUYV/UYVY. order 0 is YUYV, 1 is UYVY.
    if params.kind == 6u && index >= params.plane0_offset {
        let local = plane_local(index, params.plane0_offset, params.plane0_stride);
        if local.y < params.height && local.x < ((params.width + 1u) / 2u) * 4u {
            let pair = local.x / 4u;
            let byte = local.x & 3u;
            var yuyv_component = byte;
            if params.order == 1u {
                // UYVY stored positions map to canonical Y0/U/Y1/V positions.
                yuyv_component = select(select(1u, 0u, byte == 1u), select(3u, 2u, byte == 3u), byte >= 2u);
            }
            if yuyv_component == 0u { return quantize8(yuv_at(pair * 2u, local.y).x, 0u); }
            if yuyv_component == 2u { return quantize8(yuv_at(pair * 2u + 1u, local.y).x, 0u); }
            let chroma = chroma_at(pair, local.y);
            return quantize8(select(chroma.x, chroma.y, yuyv_component == 3u), select(1u, 2u, yuyv_component == 3u));
        }
        return 0u;
    }

    // Planar/semi-planar YUV8.
    let bytes_per_sample = params.storage_bits / 8u;
    if plane_contains(index, params.plane0_offset, params.plane0_stride, params.width * bytes_per_sample, params.height) {
        let local = plane_local(index, params.plane0_offset, params.plane0_stride);
        if local.x < params.width * bytes_per_sample && local.y < params.height {
            let sample_x = local.x / bytes_per_sample;
            let byte = local.x % bytes_per_sample;
            return stored_code_byte(quantize_code(yuv_at(sample_x, local.y).x, 0u), byte);
        }
        return 0u;
    }
    let chroma_height = (params.height + params.subsample_y - 1u) / params.subsample_y;
    let chroma_width = (params.width + params.subsample_x - 1u) / params.subsample_x;
    if params.kind == 5u && plane_contains(index, params.plane1_offset, params.plane1_stride, chroma_width * 2u * bytes_per_sample, chroma_height) {
        let local = plane_local(index, params.plane1_offset, params.plane1_stride);
        if local.x < chroma_width * 2u * bytes_per_sample && local.y < chroma_height {
            let stored_sample = local.x / bytes_per_sample;
            let byte = local.x % bytes_per_sample;
            let chroma = chroma_at(stored_sample / 2u, local.y);
            let stored_component = stored_sample & 1u;
            let component = select(stored_component, 1u - stored_component, params.order == 1u);
            return stored_code_byte(quantize_code(select(chroma.x, chroma.y, component == 1u), component + 1u), byte);
        }
        return 0u;
    }
    if params.kind == 4u {
        if plane_contains(index, params.plane1_offset, params.plane1_stride, chroma_width * bytes_per_sample, chroma_height) {
            let local = plane_local(index, params.plane1_offset, params.plane1_stride);
            if local.x < chroma_width * bytes_per_sample && local.y < chroma_height {
                let sample_x = local.x / bytes_per_sample;
                return stored_code_byte(quantize_code(chroma_at(sample_x, local.y).x, 1u), local.x % bytes_per_sample);
            }
        }
        if plane_contains(index, params.plane2_offset, params.plane2_stride, chroma_width * bytes_per_sample, chroma_height) {
            let local = plane_local(index, params.plane2_offset, params.plane2_stride);
            if local.x < chroma_width * bytes_per_sample && local.y < chroma_height {
                let sample_x = local.x / bytes_per_sample;
                return stored_code_byte(quantize_code(chroma_at(sample_x, local.y).y, 2u), local.x % bytes_per_sample);
            }
        }
    }
    return 0u;
}

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_index = gid.y * params.dispatch_width + gid.x;
    let word_count = params.logical_size / 4u + select(0u, 1u, params.logical_size % 4u != 0u);
    if word_index >= word_count { return; }
    let byte_index = word_index * 4u;
    output[word_index] = byte_at(byte_index)
        | (byte_at(byte_index + 1u) << 8u)
        | (byte_at(byte_index + 2u) << 16u)
        | (byte_at(byte_index + 3u) << 24u);
}
