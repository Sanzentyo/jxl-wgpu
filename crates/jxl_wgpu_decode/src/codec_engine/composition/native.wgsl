struct Params {
    extent: vec4<u32>, // output width/height, source width/height
    format: vec4<u32>, // channels, valid bits, bytes per sample, row bytes
    output: vec4<u32>, // logical bytes, dispatch width, orientation, reserved
};
@group(0) @binding(0) var<storage, read> source: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn output_byte(offset: u32) -> u32 {
    if offset >= params.output.x { return 0u; }
    let row = offset / params.format.w;
    let in_row = offset % params.format.w;
    let sample = in_row / params.format.z;
    let p = image_source_coordinate(vec2<u32>(sample / params.format.x, row), params.extent.zw, params.output.z);
    let value = source[p.y * params.extent.z + p.x][sample % params.format.x];
    let mask = (1u << params.format.y) - 1u;
    let code = u32(floor(clamp(value, 0.0, 1.0) * f32(mask) + 0.5));
    return (code >> ((in_row % params.format.z) * 8u)) & 255u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.y * params.output.y + id.x;
    if word >= (params.output.x + 3u) / 4u { return; }
    let offset = word * 4u;
    destination[word] = output_byte(offset) | (output_byte(offset + 1u) << 8u)
        | (output_byte(offset + 2u) << 16u) | (output_byte(offset + 3u) << 24u);
}
