override wg_x: u32 = 16u;
override wg_y: u32 = 16u;

struct Params {
    width: u32,
    height: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    input_stride_b: u32,
    output_stride_x: u32,
    output_stride_y: u32,
    output_stride_b: u32,
    weight0_x: f32,
    weight1_x: f32,
    weight2_x: f32,
    _pad_x: f32,
    weight0_y: f32,
    weight1_y: f32,
    weight2_y: f32,
    _pad_y: f32,
    weight0_b: f32,
    weight1_b: f32,
    weight2_b: f32,
    _pad_b: f32,
};

@group(0) @binding(0) var<storage, read> input_x: array<f32>;
@group(0) @binding(1) var<storage, read> input_y: array<f32>;
@group(0) @binding(2) var<storage, read> input_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_b: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

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

fn sample_x(x: i32, y: i32) -> f32 {
    return input_x[
        mirror_coordinate(y, params.height) * params.input_stride_x
            + mirror_coordinate(x, params.width)
    ];
}

fn sample_y(x: i32, y: i32) -> f32 {
    return input_y[
        mirror_coordinate(y, params.height) * params.input_stride_y
            + mirror_coordinate(x, params.width)
    ];
}

fn sample_b(x: i32, y: i32) -> f32 {
    return input_b[
        mirror_coordinate(y, params.height) * params.input_stride_b
            + mirror_coordinate(x, params.width)
    ];
}

fn filter_x(x: i32, y: i32) -> f32 {
    let center = sample_x(x, y) * params.weight0_x;
    let axial = sample_x(x, y - 1) + sample_x(x - 1, y)
        + sample_x(x, y + 1) + sample_x(x + 1, y);
    let diagonal = sample_x(x - 1, y - 1) + sample_x(x + 1, y - 1)
        + sample_x(x - 1, y + 1) + sample_x(x + 1, y + 1);
    return fma(diagonal, params.weight2_x, fma(axial, params.weight1_x, center));
}

fn filter_y(x: i32, y: i32) -> f32 {
    let center = sample_y(x, y) * params.weight0_y;
    let axial = sample_y(x, y - 1) + sample_y(x - 1, y)
        + sample_y(x, y + 1) + sample_y(x + 1, y);
    let diagonal = sample_y(x - 1, y - 1) + sample_y(x + 1, y - 1)
        + sample_y(x - 1, y + 1) + sample_y(x + 1, y + 1);
    return fma(diagonal, params.weight2_y, fma(axial, params.weight1_y, center));
}

fn filter_b(x: i32, y: i32) -> f32 {
    let center = sample_b(x, y) * params.weight0_b;
    let axial = sample_b(x, y - 1) + sample_b(x - 1, y)
        + sample_b(x, y + 1) + sample_b(x + 1, y);
    let diagonal = sample_b(x - 1, y - 1) + sample_b(x + 1, y - 1)
        + sample_b(x - 1, y + 1) + sample_b(x + 1, y + 1);
    return fma(diagonal, params.weight2_b, fma(axial, params.weight1_b, center));
}

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    output_x[gid.y * params.output_stride_x + gid.x] = filter_x(x, y);
    output_y[gid.y * params.output_stride_y + gid.x] = filter_y(x, y);
    output_b[gid.y * params.output_stride_b + gid.x] = filter_b(x, y);
}
