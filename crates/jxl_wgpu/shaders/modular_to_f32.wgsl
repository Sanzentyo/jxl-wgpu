struct Params {
    width: u32,
    height: u32,
    input_stride: u32,
    output_stride: u32,
    multiplier: f32,
    bias: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let input_index = gid.y * params.input_stride + gid.x;
    let output_index = gid.y * params.output_stride + gid.x;
    output[output_index] = f32(input[input_index]) * params.multiplier + params.bias;
}
