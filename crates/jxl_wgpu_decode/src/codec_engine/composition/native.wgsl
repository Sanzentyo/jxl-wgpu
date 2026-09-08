/*__JXL_MODULAR_SAMPLE__*/
struct Params {
    extent: vec4<u32>, // output width/height, source width/height
    format: vec4<u32>, // channels, valid bits, bytes per sample, row bytes
    output: vec4<u32>, // logical bytes, dispatch width, orientation, alpha conversion
    source: vec4<u32>, // plane stride, first alpha plane, selected scalar plane, flags (F32, linear RGB)
};
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn surface_value(word: u32) -> f32 { return source[word]; }
fn original_rgb(rgb: vec3<f32>) -> vec3<f32> {
    if (params.source.w & 2u) == 0u { return rgb; }
    let magnitude = abs(rgb);
    return sign(rgb) * select(1.055 * pow(magnitude, vec3<f32>(1.0 / 2.4)) - 0.055,
        12.92 * magnitude, magnitude <= vec3<f32>(0.0031308));
}

fn output_byte(offset: u32) -> u32 {
    if offset >= params.output.x { return 0u; }
    let row = offset / params.format.w;
    let in_row = offset % params.format.w;
    let sample = in_row / params.format.z;
    let p = image_source_coordinate(vec2<u32>(sample / params.format.x, row), params.extent.zw, params.output.z);
    let position = p.y * params.extent.z + p.x;
    let channel = sample % params.format.x;
    let source_channel = select(channel, params.source.z, params.format.x == 1u);
    var alpha = 1.0;
    if params.source.y != 0xffffffffu { alpha = source[params.source.y * params.source.x + position]; }
    var value = alpha;
    if channel != 3u {
        value = source[source_channel * params.source.x + position];
        if source_channel < 3u {
            let rgb = original_rgb(present_rgb(vec3<f32>(source[position], source[params.source.x + position],
                source[2u * params.source.x + position]), position));
            value = rgb[source_channel] * image_alpha_multiplier(alpha, params.output.w);
        }
    }
    if (params.source.w & 1u) != 0u {
        return (bitcast<u32>(value) >> ((in_row % params.format.z) * 8u)) & 255u;
    }
    let mask = (1u << params.format.y) - 1u;
    let code = modular_quantize_unsigned(value, mask);
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
