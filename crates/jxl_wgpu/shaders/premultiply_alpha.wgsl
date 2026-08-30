struct Params {
    width: u32,
    height: u32,
    color_stride: u32,
    alpha_stride: u32,
    output_stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> color: array<f32>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let color_index = gid.y * params.color_stride + gid.x;
    let alpha_index = gid.y * params.alpha_stride + gid.x;
    let output_index = gid.y * params.output_stride + gid.x;
    output[output_index] = color[color_index] * alpha[alpha_index];
}
