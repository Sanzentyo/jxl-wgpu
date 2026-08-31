override wg_x: u32 = 16u;
override wg_y: u32 = 16u;

struct Params {
    width: u32,
    height: u32,
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
    chroma_width: u32,
    chroma_height: u32,
    transfer: u32,
};

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var destination: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

fn read_byte(offset: u32) -> u32 {
    return (source[offset >> 2u] >> ((offset & 3u) * 8u)) & 0xffu;
}

fn read_code(offset: u32) -> f32 {
    if params.storage_bits == 8u { return f32(read_byte(offset)); }
    let stored = read_byte(offset) | (read_byte(offset + 1u) << 8u);
    return f32(stored >> (params.storage_bits - params.bits));
}

fn maximum_code() -> f32 {
    return f32((1u << params.bits) - 1u);
}

fn normalized_y(code: f32) -> f32 {
    if params.range == 0u { return code / maximum_code(); }
    return (code / f32(1u << (params.bits - 8u)) - 16.0) / 219.0;
}

fn normalized_chroma(code: f32) -> f32 {
    if params.range == 0u {
        return (code - f32(1u << (params.bits - 1u))) / maximum_code();
    }
    return (code / f32(1u << (params.bits - 8u)) - 128.0) / 224.0;
}

fn stored_rgb_position(canonical: u32) -> u32 {
    if (params.order == 1u || params.order == 3u) && canonical < 3u { return 2u - canonical; }
    return canonical;
}

fn rgb_sample(pixel: vec2<u32>, canonical: u32) -> f32 {
    if canonical == 3u && params.channels == 3u { return 1.0; }
    let stored = stored_rgb_position(canonical);
    var offset: u32;
    if params.kind == 0u {
        offset = params.plane0_offset + pixel.y * params.plane0_stride + pixel.x * params.channels + stored;
    } else {
        var plane_offset = params.plane0_offset;
        var plane_stride = params.plane0_stride;
        if stored == 1u { plane_offset = params.plane1_offset; plane_stride = params.plane1_stride; }
        if stored == 2u { plane_offset = params.plane2_offset; plane_stride = params.plane2_stride; }
        if stored == 3u { plane_offset = params.plane3_offset; plane_stride = params.plane3_stride; }
        offset = plane_offset + pixel.y * plane_stride + pixel.x;
    }
    return f32(read_byte(offset)) / 255.0;
}

fn luma_code(pixel: vec2<u32>) -> f32 {
    if params.kind == 6u {
        let pair = pixel.x / 2u;
        let within_pair = pixel.x & 1u;
        var byte = within_pair * 2u;
        if params.order == 1u {
            byte += 1u;
        }
        return f32(read_byte(params.plane0_offset + pixel.y * params.plane0_stride + pair * 4u + byte));
    }
    let bytes_per_sample = params.storage_bits / 8u;
    return read_code(params.plane0_offset + pixel.y * params.plane0_stride + pixel.x * bytes_per_sample);
}

fn packed_chroma(component: u32, coordinate: vec2<i32>) -> f32 {
    let x = u32(clamp(coordinate.x, 0, i32(params.chroma_width) - 1));
    let y = u32(clamp(coordinate.y, 0, i32(params.chroma_height) - 1));
    let pair_offset = params.plane0_offset + y * params.plane0_stride + x * 4u;
    // order 0 = YUYV, 1 = UYVY.
    let byte_offset = select(select(1u, 3u, component == 1u), select(0u, 2u, component == 1u), params.order == 1u);
    return f32(read_byte(pair_offset + byte_offset));
}

fn plane_chroma(component: u32, coordinate: vec2<i32>) -> f32 {
    let x = u32(clamp(coordinate.x, 0, i32(params.chroma_width) - 1));
    let y = u32(clamp(coordinate.y, 0, i32(params.chroma_height) - 1));
    let bytes_per_sample = params.storage_bits / 8u;
    if params.kind == 5u {
        let stored = select(component, 1u - component, params.order == 1u);
        return read_code(params.plane1_offset + y * params.plane1_stride + (x * 2u + stored) * bytes_per_sample);
    }
    let offset = select(params.plane1_offset, params.plane2_offset, component == 1u);
    let stride = select(params.plane1_stride, params.plane2_stride, component == 1u);
    return read_code(offset + y * stride + x * bytes_per_sample);
}

fn chroma_code(component: u32, coordinate: vec2<i32>) -> f32 {
    if params.kind == 6u { return packed_chroma(component, coordinate); }
    return plane_chroma(component, coordinate);
}

fn bilinear_chroma(component: u32, position: vec2<f32>) -> f32 {
    let lower = vec2<i32>(floor(position));
    let fraction = fract(position);
    let top = mix(chroma_code(component, lower), chroma_code(component, lower + vec2<i32>(1, 0)), fraction.x);
    let bottom = mix(chroma_code(component, lower + vec2<i32>(0, 1)), chroma_code(component, lower + vec2<i32>(1, 1)), fraction.x);
    return mix(top, bottom, fraction.y);
}

fn chroma_position(pixel: vec2<u32>) -> vec2<f32> {
    let divisor = vec2<f32>(f32(params.subsample_x), f32(params.subsample_y));
    let position = vec2<f32>(pixel);
    let centered = (position + vec2<f32>(0.5)) / divisor - vec2<f32>(0.5);
    let cosited = position / divisor;
    return vec2<f32>(
        select(centered.x, cosited.x, params.siting_x == 1u),
        select(centered.y, cosited.y, params.siting_y == 1u),
    );
}

fn matrix_coefficients() -> vec2<f32> {
    if params.matrix == 0u { return vec2<f32>(0.299, 0.114); }
    if params.matrix == 1u { return vec2<f32>(0.2126, 0.0722); }
    return vec2<f32>(0.2627, 0.0593);
}

fn yuv_rgb(pixel: vec2<u32>) -> vec3<f32> {
    var y = normalized_y(luma_code(pixel));
    var cb = 0.0;
    var cr = 0.0;
    if params.kind != 2u && params.kind != 3u {
        let position = chroma_position(pixel);
        cb = normalized_chroma(bilinear_chroma(0u, position));
        cr = normalized_chroma(bilinear_chroma(1u, position));
    }
    let coefficient = matrix_coefficients();
    let kr = coefficient.x;
    let kb = coefficient.y;
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = (y - kr * r - kb * b) / (1.0 - kr - kb);
    return vec3<f32>(r, g, b);
}

fn to_linear(encoded: f32) -> f32 {
    let value = clamp(encoded, 0.0, 1.0);
    if params.transfer == 0u {
        return value;
    }
    if params.transfer == 1u {
        if value <= 0.04045 {
            return value / 12.92;
        }
        return pow((value + 0.055) / 1.055, 2.4);
    }
    if value < 0.081 {
        return value / 4.5;
    }
    return pow((value + 0.099) / 1.099, 1.0 / 0.45);
}

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x >= params.width || invocation.y >= params.height { return; }
    let pixel = invocation.xy;
    var rgba: vec4<f32>;
    if params.kind == 0u || params.kind == 1u {
        rgba = vec4<f32>(rgb_sample(pixel, 0u), rgb_sample(pixel, 1u), rgb_sample(pixel, 2u), rgb_sample(pixel, 3u));
    } else {
        rgba = vec4<f32>(yuv_rgb(pixel), 1.0);
    }
    rgba = vec4<f32>(
        to_linear(rgba.r),
        to_linear(rgba.g),
        to_linear(rgba.b),
        rgba.a,
    );
    textureStore(destination, vec2<i32>(pixel), rgba);
}
