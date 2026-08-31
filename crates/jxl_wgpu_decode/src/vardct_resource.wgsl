override wg_x: u32 = 64u;
override wg_y: u32 = 1u;

struct Params {
    geometry: vec4<u32>,
    offsets: vec4<u32>,
    scales: vec4<f32>,
    _reserved: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> quantized_lf: array<u32>;
@group(0) @binding(1) var<storage, read_write> dequantized_lf: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(wg_x, wg_y, 1)
fn prepare_lf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let block_count = params.geometry.z;
    if index >= block_count { return; }
    let raw_y = bitcast<i32>(quantized_lf[params.offsets.x + index]);
    let raw_x = bitcast<i32>(quantized_lf[params.offsets.y + index]);
    let raw_b = bitcast<i32>(quantized_lf[params.offsets.z + index]);
    let y = f32(raw_y) * params.scales.y;
    let x = index % params.geometry.x;
    let row = index / params.geometry.x;
    dequantized_lf[params.offsets.w + row * params.geometry.w + x] = vec4<f32>(
        f32(raw_x) * params.scales.x,
        y,
        f32(raw_b) * params.scales.z + y * params.scales.w,
        0.0,
    );
}
