struct Params {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    input_stride: u32,
    output_stride: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn mirror_coordinate(value: i32, size: u32) -> u32 {
    if size <= 1u {
        return 0u;
    }
    var reflected = value;
    let signed_size = i32(size);
    loop {
        if reflected < 0 {
            reflected = -reflected - 1;
        } else if reflected >= signed_size {
            reflected = signed_size * 2 - reflected - 1;
        } else {
            break;
        }
    }
    return u32(reflected);
}

fn load_mirrored(x: i32, y: i32) -> f32 {
    let mirrored_x = mirror_coordinate(x, params.input_width);
    let mirrored_y = mirror_coordinate(y, params.input_height);
    return input[mirrored_y * params.input_stride + mirrored_x];
}

fn horizontal_sample(output_x: u32, input_y: i32) -> f32 {
    let source_x = i32(output_x / 2u);
    let neighbor_x = source_x + select(-1, 1, (output_x & 1u) != 0u);
    let current = load_mirrored(source_x, input_y);
    let neighbor = load_mirrored(neighbor_x, input_y);
    return fma(neighbor, 0.25, current * 0.75);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.output_width || gid.y >= params.output_height {
        return;
    }

    let source_y = i32(gid.y / 2u);
    let neighbor_y = source_y + select(-1, 1, (gid.y & 1u) != 0u);
    let current = horizontal_sample(gid.x, source_y);
    let neighbor = horizontal_sample(gid.x, neighbor_y);
    output[gid.y * params.output_stride + gid.x] = fma(neighbor, 0.25, current * 0.75);
}
