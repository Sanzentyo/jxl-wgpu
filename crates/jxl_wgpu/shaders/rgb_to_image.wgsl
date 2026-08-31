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
    _padding: u32,
};

@group(0) @binding(0) var<storage, read> source_r: array<u32>;
@group(0) @binding(1) var<storage, read> source_g: array<u32>;
@group(0) @binding(2) var<storage, read> source_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

fn source_coordinate(destination: vec2<u32>) -> vec2<u32> {
    let x = destination.x;
    let y = destination.y;
    switch params.orientation {
        case 1u: { return vec2<u32>(params.source_width - 1u - x, y); }
        case 2u: { return vec2<u32>(params.source_width - 1u - x, params.source_height - 1u - y); }
        case 3u: { return vec2<u32>(x, params.source_height - 1u - y); }
        case 4u: { return vec2<u32>(y, x); }
        case 5u: { return vec2<u32>(y, params.source_height - 1u - x); }
        case 6u: { return vec2<u32>(params.source_width - 1u - y, params.source_height - 1u - x); }
        case 7u: { return vec2<u32>(params.source_width - 1u - y, x); }
        default: { return destination; }
    }
}

fn source_rgb_at(x: u32, y: u32) -> vec3<f32> {
    let coordinate = source_coordinate(vec2<u32>(min(x, params.width - 1u), min(y, params.height - 1u)));
    return vec3<f32>(
        bitcast<f32>(source_r[coordinate.y * params.r_stride + coordinate.x]),
        bitcast<f32>(source_g[coordinate.y * params.g_stride + coordinate.x]),
        bitcast<f32>(source_b[coordinate.y * params.b_stride + coordinate.x]),
    );
}

fn transfer_to_linear(value: f32, transfer: u32) -> f32 {
    if transfer == 1u {
        if value <= 0.04045 { return value / 12.92; }
        return pow((value + 0.055) / 1.055, 2.4);
    }
    if transfer == 2u {
        if value < 0.081 { return value / 4.5; }
        return pow((value + 0.099) / 1.099, 1.0 / 0.45);
    }
    return value;
}

fn transfer_from_linear(value: f32, transfer: u32) -> f32 {
    if transfer == 1u {
        if value <= 0.0031308 { return 12.92 * value; }
        return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
    }
    if transfer == 2u {
        if value < 0.018 { return 4.5 * value; }
        return 1.099 * pow(value, 0.45) - 0.099;
    }
    return value;
}

fn rgb_at(x: u32, y: u32) -> vec3<f32> {
    let source = source_rgb_at(x, y);
    let linear = vec3<f32>(
        transfer_to_linear(source.r, params.source_transfer),
        transfer_to_linear(source.g, params.source_transfer),
        transfer_to_linear(source.b, params.source_transfer),
    );
    return vec3<f32>(
        transfer_from_linear(linear.r, params.target_transfer),
        transfer_from_linear(linear.g, params.target_transfer),
        transfer_from_linear(linear.b, params.target_transfer),
    );
}

fn coefficients() -> vec2<f32> {
    switch params.matrix {
        case 0u: { return vec2<f32>(0.299, 0.114); }
        case 1u: { return vec2<f32>(0.2126, 0.0722); }
        default: { return vec2<f32>(0.2126, 0.0722); }
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

fn chroma_at(cx: u32, cy: u32) -> vec2<f32> {
    let origin_x = cx * params.subsample_x;
    let origin_y = cy * params.subsample_y;
    let centered_x = params.siting_x == 0u && params.subsample_x > 1u;
    let centered_y = params.siting_y == 0u && params.subsample_y > 1u;
    if !centered_x && !centered_y {
        return rgb_to_yuv(rgb_at(origin_x, origin_y)).yz;
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
                sum += rgb_to_yuv(rgb_at(x, y)).yz;
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

fn rgb_component(rgb: vec3<f32>, component: u32) -> f32 {
    if component == 0u { return rgb.r; }
    if component == 1u { return rgb.g; }
    if component == 2u { return rgb.b; }
    return 1.0;
}

fn stored_rgb_component(position: u32) -> u32 {
    if (params.order == 1u || params.order == 3u) && position < 3u {
        return 2u - position;
    }
    return position;
}

fn byte_at(index: u32) -> u32 {
    if index >= params.logical_size { return 0u; }

    // RGB8 interleaved.
    if params.kind == 0u && index >= params.plane0_offset {
        let local = plane_local(index, params.plane0_offset, params.plane0_stride);
        if local.y < params.height && local.x < params.width * params.channels {
            let pixel = local.x / params.channels;
            let component = stored_rgb_component(local.x % params.channels);
            return quantize8(rgb_component(rgb_at(pixel, local.y), component), 0u);
        }
        return 0u;
    }

    // RGB8 planar. Plane index is also stored channel position.
    if params.kind == 1u {
        var plane = 4u;
        var local = vec2<u32>(0u);
        if index >= params.plane0_offset && index < params.plane0_offset + params.plane0_stride * params.height {
            plane = 0u; local = plane_local(index, params.plane0_offset, params.plane0_stride);
        } else if index >= params.plane1_offset && index < params.plane1_offset + params.plane1_stride * params.height {
            plane = 1u; local = plane_local(index, params.plane1_offset, params.plane1_stride);
        } else if index >= params.plane2_offset && index < params.plane2_offset + params.plane2_stride * params.height {
            plane = 2u; local = plane_local(index, params.plane2_offset, params.plane2_stride);
        } else if params.channels == 4u && index >= params.plane3_offset && index < params.plane3_offset + params.plane3_stride * params.height {
            plane = 3u; local = plane_local(index, params.plane3_offset, params.plane3_stride);
        }
        if plane < params.channels && local.x < params.width && local.y < params.height {
            return quantize8(rgb_component(rgb_at(local.x, local.y), stored_rgb_component(plane)), 0u);
        }
        return 0u;
    }

    // Luma-only Y8/Y16.
    if (params.kind == 2u || params.kind == 3u) && index >= params.plane0_offset {
        let local = plane_local(index, params.plane0_offset, params.plane0_stride);
        let bytes_per_sample = select(1u, 2u, params.kind == 3u);
        if local.y < params.height && local.x < params.width * bytes_per_sample {
            let pixel = local.x / bytes_per_sample;
            let y = rgb_to_yuv(rgb_at(pixel, local.y)).x;
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
            if yuyv_component == 0u { return quantize8(rgb_to_yuv(rgb_at(pair * 2u, local.y)).x, 0u); }
            if yuyv_component == 2u { return quantize8(rgb_to_yuv(rgb_at(pair * 2u + 1u, local.y)).x, 0u); }
            let chroma = chroma_at(pair, local.y);
            return quantize8(select(chroma.x, chroma.y, yuyv_component == 3u), select(1u, 2u, yuyv_component == 3u));
        }
        return 0u;
    }

    // Planar/semi-planar YUV8.
    let bytes_per_sample = params.storage_bits / 8u;
    if index >= params.plane0_offset && index < params.plane0_offset + params.plane0_stride * params.height {
        let local = plane_local(index, params.plane0_offset, params.plane0_stride);
        if local.x < params.width * bytes_per_sample && local.y < params.height {
            let sample_x = local.x / bytes_per_sample;
            let byte = local.x % bytes_per_sample;
            return stored_code_byte(quantize_code(rgb_to_yuv(rgb_at(sample_x, local.y)).x, 0u), byte);
        }
        return 0u;
    }
    let chroma_height = (params.height + params.subsample_y - 1u) / params.subsample_y;
    let chroma_width = (params.width + params.subsample_x - 1u) / params.subsample_x;
    if params.kind == 5u && index >= params.plane1_offset && index < params.plane1_offset + params.plane1_stride * chroma_height {
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
        if index >= params.plane1_offset && index < params.plane1_offset + params.plane1_stride * chroma_height {
            let local = plane_local(index, params.plane1_offset, params.plane1_stride);
            if local.x < chroma_width * bytes_per_sample && local.y < chroma_height {
                let sample_x = local.x / bytes_per_sample;
                return stored_code_byte(quantize_code(chroma_at(sample_x, local.y).x, 1u), local.x % bytes_per_sample);
            }
        }
        if index >= params.plane2_offset && index < params.plane2_offset + params.plane2_stride * chroma_height {
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
    let byte_index = word_index * 4u;
    if byte_index >= params.logical_size { return; }
    output[word_index] = byte_at(byte_index)
        | (byte_at(byte_index + 1u) << 8u)
        | (byte_at(byte_index + 2u) << 16u)
        | (byte_at(byte_index + 3u) << 24u);
}
