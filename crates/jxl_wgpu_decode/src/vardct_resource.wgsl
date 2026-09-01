override wg_x: u32 = 64u;
override wg_y: u32 = 1u;

struct Params {
    geometry: vec4<u32>,
    source_geometry: array<vec4<u32>, 3>,
    destination_geometry: array<vec4<u32>, 3>,
    scales: vec4<f32>,
    correlation: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> quantized_lf: array<u32>;
@group(0) @binding(1) var<storage, read_write> dequantized_lf: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(wg_x, wg_y, 1)
fn prepare_lf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.geometry.z) { return; }

    if (params.geometry.w != 0u) {
        let raw_y = bitcast<i32>(quantized_lf[params.source_geometry[1].z + index]);
        let raw_x = bitcast<i32>(quantized_lf[params.source_geometry[0].z + index]);
        let raw_b = bitcast<i32>(quantized_lf[params.source_geometry[2].z + index]);
        let y = f32(raw_y) * params.scales.y;
        let destination = params.destination_geometry[0];
        let x = index % params.source_geometry[0].x;
        let row = index / params.source_geometry[0].x;
        var output = vec4<f32>(
            f32(raw_x) * params.scales.x,
            y,
            f32(raw_b) * params.scales.z,
            0.0,
        );
        if (params.geometry.w == 1u) {
            output.x = fma(y, params.correlation.x, output.x);
            output.z = fma(y, params.correlation.y, output.z);
        }
        dequantized_lf[destination.w + (destination.z + row) * destination.x
            + destination.y + x] = output;
        return;
    }

    for (var channel = 0u; channel < 3u; channel += 1u) {
        let source = params.source_geometry[channel];
        let samples = source.x * source.y;
        if (index >= samples) { continue; }
        let raw = bitcast<i32>(quantized_lf[source.z + index]);
        let destination = params.destination_geometry[channel];
        let x = index % source.x;
        let row = index / source.x;
        let value = f32(raw) * params.scales[channel];
        var vector = vec4<f32>(0.0);
        vector[channel] = value;
        dequantized_lf[destination.w + (destination.z + row) * destination.x
            + destination.y + x] = vector;
    }
}
