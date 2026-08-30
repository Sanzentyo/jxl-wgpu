struct Params {
    width: u32,
    height: u32,
    source_stride: u32,
    channels: u32,
    channel: u32,
    output_layout: u32,
    orientation: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn output_size() -> vec2<u32> {
    if (params.orientation >= 4u) {
        return vec2<u32>(params.height, params.width);
    }
    return vec2<u32>(params.width, params.height);
}

// Inverse of JPEG XL's source-to-display orientation mapping.
fn source_coordinate(destination: vec2<u32>) -> vec2<u32> {
    let x = destination.x;
    let y = destination.y;
    switch params.orientation {
        case 1u: { return vec2<u32>(params.width - 1u - x, y); }
        case 2u: { return vec2<u32>(params.width - 1u - x, params.height - 1u - y); }
        case 3u: { return vec2<u32>(x, params.height - 1u - y); }
        case 4u: { return vec2<u32>(y, x); }
        case 5u: { return vec2<u32>(y, params.height - 1u - x); }
        case 6u: { return vec2<u32>(params.width - 1u - y, params.height - 1u - x); }
        case 7u: { return vec2<u32>(params.width - 1u - y, x); }
        default: { return destination; }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = output_size();
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let source_xy = source_coordinate(gid.xy);
    let source_index = source_xy.y * params.source_stride + source_xy.x;
    let pixel_index = gid.y * size.x + gid.x;
    var output_index: u32;
    if (params.output_layout == 0u) {
        output_index = params.channel * size.x * size.y + pixel_index;
    } else {
        output_index = pixel_index * params.channels + params.channel;
    }
    output[output_index] = source[source_index];
}
