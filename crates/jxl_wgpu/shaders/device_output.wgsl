struct DeviceOutputParams {
    geometry: vec4<u32>,
    channels: vec4<u32>,
    mapping: vec4<u32>,
    sizes: vec4<u32>,
    input_offsets: array<vec4<u32>, 4>,
    input_strides: array<vec4<u32>, 4>,
    output_offsets: array<vec4<u32>, 4>,
    output_strides: array<vec4<u32>, 4>,
}
@group(0) @binding(0) var<storage, read> device_samples: array<f32>;
@group(0) @binding(3) var<storage, read_write> device_output: array<u32>;
@group(0) @binding(4) var<uniform> device_params: DeviceOutputParams;
override wg_x: u32 = 64u;
override wg_y: u32 = 1u;

fn device_sample(c: u32, position: vec2<u32>) -> f32 {
    return device_samples[device_params.input_offsets[c / 4u][c % 4u]
        + position.y * device_params.input_strides[c / 4u][c % 4u] + position.x];
}

fn device_value(c: u32, position: vec2<u32>) -> f32 {
    let source = image_source_coordinate(position, device_params.geometry.zw, device_params.channels.w);
    var alpha = 1.0;
    if device_params.mapping.x != 0u { alpha = device_sample(device_params.channels.x, source); }
    if c == device_params.channels.x { return alpha; }
    var value = device_sample(c, source);
    if device_params.mapping.y == 1u { value = 1.0 - value; }
    if device_params.mapping.z != 0u { value *= image_alpha_multiplier(alpha, device_params.mapping.z); }
    return value;
}

fn device_byte(address: u32) -> u32 {
    if address >= device_params.sizes.x { return 0u; }
    for (var c = 0u; c < device_params.channels.x + device_params.channels.y; c++) {
        let offset = device_params.output_offsets[c / 4u][c % 4u];
        let stride = device_params.output_strides[c / 4u][c % 4u];
        if address < offset { continue; }
        let delta = address - offset;
        let y = delta / stride;
        let row = delta % stride;
        let x = row / device_params.mapping.w;
        let byte = row % device_params.mapping.w;
        if y >= device_params.geometry.y || x >= device_params.geometry.x || byte >= device_params.channels.z { continue; }
        let value = device_value(c, vec2<u32>(x, y));
        if device_params.channels.z == 1u { return u32(floor(clamp(value, 0.0, 1.0) * 255.0 + 0.5)); }
        return (bitcast<u32>(value) >> (byte * 8u)) & 255u;
    }
    return 0u;
}

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.y * device_params.sizes.y + id.x;
    if word >= (device_params.sizes.x + 3u) / 4u { return; }
    let start = word * 4u;
    device_output[word] = device_byte(start) | (device_byte(start + 1u) << 8u)
        | (device_byte(start + 2u) << 16u) | (device_byte(start + 3u) << 24u);
}
