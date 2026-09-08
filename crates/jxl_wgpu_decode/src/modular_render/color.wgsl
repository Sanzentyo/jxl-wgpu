/*__JXL_MODULAR_SAMPLE__*/
struct Params {
    sources: array<vec4<u32>, 3>,
    encodings: vec4<u32>,
    multipliers: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_b: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn source_at(channel: u32, coordinate: vec2<u32>) -> u32 {
    let plane = params.sources[channel];
    return source[plane.w + coordinate.y * plane.z + coordinate.x];
}
@compute @workgroup_size(16, 16, 1)
fn normalize_color(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.sources[0].x || id.y >= params.sources[0].y { return; }
    let words = vec3<u32>(source_at(0u, id.xy), source_at(1u, id.xy), source_at(2u, id.xy));
    var color: vec3<f32>;
    if params.encodings.w == 1u {
        // Modular stores Y/X/(B-Y). The sum is performed in the signed working-word domain.
        color = vec3<f32>(f32(bitcast<i32>(words.y)), f32(bitcast<i32>(words.x)),
            f32(bitcast<i32>(words.z + words.x))) * params.multipliers.xyz;
    } else {
        color = vec3<f32>(bitcast<f32>(modular_sample_f32_bits(words.x, params.encodings.x)),
            bitcast<f32>(modular_sample_f32_bits(words.y, params.encodings.y)),
            bitcast<f32>(modular_sample_f32_bits(words.z, params.encodings.z)));
    }
    let index = id.y * params.sources[0].x + id.x;
    output_x[index] = color.x;
    output_y[index] = color.y;
    output_b[index] = color.z;
}
