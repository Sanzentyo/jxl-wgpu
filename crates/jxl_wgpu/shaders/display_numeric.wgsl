override wg_x: u32 = 16u;
override wg_y: u32 = 16u;

struct NumericParams {
    width: u32,
    height: u32,
    sample_kind: u32,
    bits: u32,
    components: u32,
    plane_offset: u32,
    plane_stride: u32,
    visualization: u32,
    non_finite: u32,
    transfer: u32,
    clamp: u32,
    _reserved: u32,
    scale: f32,
    bias: f32,
    _padding0: u32,
    _padding1: u32,
};

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var destination: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: NumericParams;

fn read_byte(offset: u32) -> u32 {
    return (source[offset >> 2u] >> ((offset & 3u) * 8u)) & 0xffu;
}

fn read_u16(offset: u32) -> u32 {
    return read_byte(offset) | (read_byte(offset + 1u) << 8u);
}

fn read_u32(offset: u32) -> u32 {
    return read_u16(offset) | (read_u16(offset + 2u) << 16u);
}

/*__JXL_NUMERIC_F64__*/

fn read_scalar(offset: u32) -> f32 {
    if params.sample_kind == 0u {
        if params.bits == 8u { return f32(read_byte(offset)); }
        if params.bits == 16u { return f32(read_u16(offset)); }
        return f32(read_u32(offset));
    }
    if params.sample_kind == 1u {
        if params.bits == 8u { return f32(bitcast<i32>(read_byte(offset) << 24u) >> 24u); }
        if params.bits == 16u { return f32(bitcast<i32>(read_u16(offset) << 16u) >> 16u); }
        return f32(bitcast<i32>(read_u32(offset)));
    }
    if params.bits == 32u {
        return bitcast<f32>(read_u32(offset)) * params.scale + params.bias;
    }
    return normalized_f64(offset);
}

fn handle_non_finite(value: f32) -> f32 {
    if value != value { return 0.0; }
    if value > 3.402823466e+38 {
        return select(0.0, 1.0, params.non_finite == 1u);
    }
    if value < -3.402823466e+38 { return 0.0; }
    return value;
}

fn normalized_component(pixel: vec2<u32>, component: u32) -> f32 {
    let bytes_per_component = params.bits / 8u;
    let offset = params.plane_offset + pixel.y * params.plane_stride
        + (pixel.x * params.components + component) * bytes_per_component;
    var value = read_scalar(offset);
    if params.sample_kind != 2u {
        value = value * params.scale + params.bias;
    }
    value = handle_non_finite(value);
    if params.clamp == 0u { value = clamp(value, 0.0, 1.0); }
    return value;
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 { return value / 12.92; }
    return pow((value + 0.055) / 1.055, 2.4);
}

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x >= params.width || invocation.y >= params.height { return; }
    let pixel = invocation.xy;
    let x = normalized_component(pixel, 0u);
    var rgba = vec4<f32>(x, x, x, 1.0);
    if params.visualization == 1u {
        rgba.a = normalized_component(pixel, 1u);
    } else if params.visualization == 2u {
        rgba = vec4<f32>(x, normalized_component(pixel, 1u), 0.0, 1.0);
    }
    if params.transfer == 1u {
        rgba = vec4<f32>(
            srgb_to_linear(rgba.r),
            srgb_to_linear(rgba.g),
            srgb_to_linear(rgba.b),
            rgba.a,
        );
    }
    textureStore(destination, vec2<i32>(pixel), rgba);
}
