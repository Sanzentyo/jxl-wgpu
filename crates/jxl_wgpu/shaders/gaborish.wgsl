override wg_x: u32 = 16u;
override wg_y: u32 = 16u;

struct Params {
    width: u32,
    height: u32,
    input_stride: u32,
    output_stride: u32,
    weight0: f32,
    weight1: f32,
    weight2: f32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn load_clamped(x: i32, y: i32) -> f32 {
    let clamped_x = u32(clamp(x, 0, i32(params.width) - 1));
    let clamped_y = u32(clamp(y, 0, i32(params.height) - 1));
    return input[clamped_y * params.input_stride + clamped_x];
}

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let center = load_clamped(x, y) * params.weight0;
    let axial = load_clamped(x, y - 1) + load_clamped(x - 1, y)
        + load_clamped(x, y + 1) + load_clamped(x + 1, y);
    let diagonal = load_clamped(x - 1, y - 1) + load_clamped(x + 1, y - 1)
        + load_clamped(x - 1, y + 1) + load_clamped(x + 1, y + 1);
    output[gid.y * params.output_stride + gid.x] =
        center + axial * params.weight1 + diagonal * params.weight2;
}
