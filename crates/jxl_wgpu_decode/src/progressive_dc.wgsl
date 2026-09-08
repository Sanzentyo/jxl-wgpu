override wg_x: u32 = 64u;

struct PackParams {
    // width, height, pixel count, reserved
    geometry: vec4<u32>,
    // F32 row strides for X, Y, and B input planes, reserved
    input_strides: vec4<u32>,
    // LF vec4 offset, LF vec4 row stride, reserved
    destination: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> input_x: array<f32>;
@group(0) @binding(1) var<storage, read> input_y: array<f32>;
@group(0) @binding(2) var<storage, read> input_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> resources: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> pack_params: PackParams;

@compute @workgroup_size(wg_x, 1, 1)
fn pack_lf(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    let width = pack_params.geometry.x;
    if index >= pack_params.geometry.z {
        return;
    }
    let y = index / width;
    let x = index - y * width;
    let output_index = pack_params.destination.x + y * pack_params.destination.y + x;
    resources[output_index] = vec4<f32>(
        input_x[y * pack_params.input_strides.x + x],
        input_y[y * pack_params.input_strides.y + x],
        input_b[y * pack_params.input_strides.z + x],
        0.0,
    );
}
