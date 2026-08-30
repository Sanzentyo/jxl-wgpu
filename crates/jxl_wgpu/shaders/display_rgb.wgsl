struct DisplayRgbParams {
    width: u32,
    height: u32,
    channels: u32,
    sample_type: u32,
    storage_layout: u32,
    logical_samples: u32,
    _padding0: u32,
    _padding1: u32,
};

@group(0) @binding(0)
var<storage, read> source: array<u32>;

@group(0) @binding(1)
var destination: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> params: DisplayRgbParams;

fn source_index(pixel: u32, channel: u32) -> u32 {
    if params.storage_layout == 0u {
        return channel * (params.width * params.height) + pixel;
    }
    return pixel * params.channels + channel;
}

fn read_u8(index: u32) -> f32 {
    let word = source[index >> 2u];
    let shift = (index & 3u) * 8u;
    return f32((word >> shift) & 0xffu) / 255.0;
}

fn read_u16(index: u32) -> f32 {
    let word = source[index >> 1u];
    let shift = (index & 1u) * 16u;
    return f32((word >> shift) & 0xffffu) / 65535.0;
}

fn read_f16(index: u32) -> f32 {
    let halves = unpack2x16float(source[index >> 1u]);
    return select(halves.x, halves.y, (index & 1u) != 0u);
}

fn read_sample(index: u32) -> f32 {
    // 0 = F32, 1 = F16, 2 = U16, 3 = U8.
    switch params.sample_type {
        case 0u: {
            return bitcast<f32>(source[index]);
        }
        case 1u: {
            return read_f16(index);
        }
        case 2u: {
            return read_u16(index);
        }
        default: {
            return read_u8(index);
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x >= params.width || invocation.y >= params.height {
        return;
    }
    let pixel = invocation.y * params.width + invocation.x;
    let r = read_sample(source_index(pixel, 0u));
    let g = read_sample(source_index(pixel, 1u));
    let b = read_sample(source_index(pixel, 2u));
    var a = 1.0;
    if params.channels == 4u {
        a = read_sample(source_index(pixel, 3u));
    }
    textureStore(destination, vec2<i32>(invocation.xy), vec4<f32>(r, g, b, a));
}
