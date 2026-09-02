override wg_x: u32 = 16u;
override wg_y: u32 = 16u;

struct Params {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    input_stride_b: u32,
    factor: u32,
    output_stride_x: u32,
    output_stride_y: u32,
    output_stride_b: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> input_x: array<f32>;
@group(0) @binding(1) var<storage, read> input_y: array<f32>;
@group(0) @binding(2) var<storage, read> input_b: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_x: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_b: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

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

fn load_mirrored(channel: u32, x: i32, y: i32) -> f32 {
    let mirrored_x = mirror_coordinate(x, params.input_width);
    let mirrored_y = mirror_coordinate(y, params.input_height);
    switch channel {
        case 0u: {
            return input_x[mirrored_y * params.input_stride_x + mirrored_x];
        }
        case 1u: {
            return input_y[mirrored_y * params.input_stride_y + mirrored_x];
        }
        default: {
            return input_b[mirrored_y * params.input_stride_b + mirrored_x];
        }
    }
}

fn upsample(channel: u32, output_x_coord: u32, output_y_coord: u32) -> f32 {
    let source_x = i32(output_x_coord / params.factor);
    let source_y = i32(output_y_coord / params.factor);
    let phase_x = output_x_coord % params.factor;
    let phase_y = output_y_coord % params.factor;
    let kernel_offset = (phase_y * params.factor + phase_x) * 25u;

    var minimum = 3.402823466e+38;
    var maximum = -3.402823466e+38;
    for (var ky = 0u; ky < 5u; ky = ky + 1u) {
        for (var kx = 0u; kx < 5u; kx = kx + 1u) {
            let value = load_mirrored(
                channel,
                source_x + i32(kx) - 2,
                source_y + i32(ky) - 2,
            );
            minimum = min(minimum, value);
            maximum = max(maximum, value);
        }
    }

    var accumulator0 = load_mirrored(channel, source_x - 2, source_y - 2)
        * weights[kernel_offset];
    var accumulator1 = load_mirrored(channel, source_x - 1, source_y - 2)
        * weights[kernel_offset + 1u];
    var accumulator2 = load_mirrored(channel, source_x, source_y - 2)
        * weights[kernel_offset + 2u];
    for (var index = 3u; index < 25u; index = index + 1u) {
        let kernel_y = index / 5u;
        let kernel_x = index % 5u;
        let value = load_mirrored(
            channel,
            source_x + i32(kernel_x) - 2,
            source_y + i32(kernel_y) - 2,
        );
        let weight = weights[kernel_offset + index];
        switch index % 3u {
            case 0u: { accumulator0 = fma(value, weight, accumulator0); }
            case 1u: { accumulator1 = fma(value, weight, accumulator1); }
            default: { accumulator2 = fma(value, weight, accumulator2); }
        }
    }
    return clamp(accumulator0 + accumulator1 + accumulator2, minimum, maximum);
}

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.output_width || gid.y >= params.output_height {
        return;
    }
    let x = gid.x;
    let y = gid.y;
    output_x[y * params.output_stride_x + x] = upsample(0u, x, y);
    output_y[y * params.output_stride_y + x] = upsample(1u, x, y);
    output_b[y * params.output_stride_b + x] = upsample(2u, x, y);
}
