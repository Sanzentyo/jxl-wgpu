/*__JXL_MODULAR_SAMPLE__*/

struct Params {
    source: vec4<u32>,
    mapping: vec4<u32>,
};
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn normalize(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.source.x || id.y >= params.source.y { return; }
    output[id.y * params.mapping.y + id.x] = modular_sample_f32_bits(
        source[params.source.w + id.y * params.source.z + id.x], params.mapping.x);
}
