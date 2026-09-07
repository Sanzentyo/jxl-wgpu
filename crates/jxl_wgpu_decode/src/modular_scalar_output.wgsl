override wg_x: u32 = 256u;
struct ScalarParams {
    source: vec4<u32>,
    destination: vec4<u32>,
    encoding: vec4<u32>,
    bounds: vec4<u32>,
};
@group(0) @binding(0) var<storage, read> samples: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: ScalarParams;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;

fn output_byte(byte_offset: u32) -> u32 {
    if byte_offset < params.destination.w || byte_offset >= params.bounds.x { return 0u; }
    let relative = byte_offset - params.destination.w;
    let y = relative / params.destination.z;
    let row_byte = relative % params.destination.z;
    let x = row_byte / params.encoding.y;
    if x >= params.destination.x || y >= params.destination.y { return 0u; }
    let point = image_source_coordinate(vec2<u32>(x, y), params.source.xy, params.encoding.w);
    let sample = samples[params.source.w + point.y * params.source.z + point.x];
    var code = bitcast<u32>(sample);
    if params.encoding.z != 0u {
        // The integer arena may contain negative or overshoot values after lossy prediction.
        // Normalization preserves those values; it never applies color conversion or a mask.
        code = bitcast<u32>(f32(sample) / f32(params.encoding.x));
    } else if sample < 0i || u32(sample) > params.encoding.x {
        // NativeUnsigned promises exact codes; an unrepresentable value must not wrap or clip.
        atomicStore(&status, 1u);
        code = 0u;
    }
    return (code >> ((row_byte % params.encoding.y) * 8u)) & 255u;
}

@compute @workgroup_size(wg_x, 1, 1)
fn pack(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.y * params.bounds.z + id.x;
    if word >= params.bounds.y { return; }
    let byte = word * 4u;
    output[word] = output_byte(byte) | (output_byte(byte + 1u) << 8u)
        | (output_byte(byte + 2u) << 16u) | (output_byte(byte + 3u) << 24u);
}
