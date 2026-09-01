struct RawMatrixParams {
    denominator: f32,
    width: u32,
    height: u32,
    target_count: u32,
    plane_offsets: vec4<u32>,
    plane_strides: vec4<u32>,
    target_offsets: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> arena: array<i32>;
@group(0) @binding(1) var<storage, read_write> resources: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: RawMatrixParams;

override wg_x: u32 = 64u;

const ERROR_RAW_MATRIX_VALUE: u32 = 15u;

@compute @workgroup_size(wg_x, 1, 1)
fn overlay(@builtin(global_invocation_id) id: vec3<u32>) {
    let sample_index = id.x;
    let sample_count = params.width * params.height;
    if sample_index >= sample_count {
        return;
    }
    let x = sample_index % params.width;
    let y = sample_index / params.width;
    var weights = vec3<f32>();
    for (var channel = 0u; channel < 3u; channel += 1u) {
        let source_index = params.plane_offsets[channel] + y * params.plane_strides[channel] + x;
        weights[channel] = f32(arena[source_index]) * params.denominator;
    }
    if any(weights <= vec3<f32>(0.0)) || any(weights >= vec3<f32>(1.0e8)) {
        atomicStore(&status[0], ERROR_RAW_MATRIX_VALUE);
        return;
    }
    let packed = vec4<f32>(weights, 0.0);
    for (var target_index = 0u; target_index < params.target_count; target_index += 1u) {
        resources[params.target_offsets[target_index] + sample_index] = packed;
    }
}
