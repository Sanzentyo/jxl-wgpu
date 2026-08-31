override wg_x: u32 = 64u;

struct Params {
    geometry: vec4<u32>,
    destination: vec4<u32>,
    artifact_status_offset_words: u32,
    task_metadata_offset_words: u32,
    global_scale: u32,
    quant_mul: f32,
    sharp_lut: array<vec4<f32>, 2>,
};

@group(0) @binding(0) var<storage, read> raw_metadata: array<i32>;
@group(0) @binding(1) var<storage, read> artifact: array<u32>;
@group(0) @binding(2) var<storage, read_write> sigma: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn sharpness_value(index: u32) -> f32 {
    return params.sharp_lut[index / 4u][index % 4u];
}

@compute @workgroup_size(wg_x, 1, 1)
fn build_epf_sigma(@builtin(global_invocation_id) gid: vec3<u32>) {
    let task_index = gid.x;
    let status = params.artifact_status_offset_words;
    if artifact[status] != 0u
        || task_index >= params.geometry.z
        || task_index >= artifact[status + 4u]
    {
        return;
    }

    let task = params.task_metadata_offset_words + task_index * 12u;
    let block_x = artifact[task + 2u];
    let block_y = artifact[task + 3u];
    let block_width = artifact[task + 4u];
    let block_height = artifact[task + 5u];
    let hf_mul = max(artifact[task + 6u], 1u);
    let quant_scale = 1.0 / (65536.0 / f32(params.global_scale));
    let sigma_quant = params.quant_mul
        / (quant_scale * f32(hf_mul) * -1.1715728752538099024);
    for (var iy = 0u; iy < block_height; iy++) {
        for (var ix = 0u; ix < block_width; ix++) {
            let x = block_x + ix;
            let y = block_y + iy;
            if x >= params.geometry.x || y >= params.geometry.y {
                continue;
            }
            let raster = y * params.geometry.x + x;
            let sharpness = min(
                bitcast<u32>(raw_metadata[params.geometry.w + raster]),
                7u,
            );
            let quantized = min(sigma_quant * sharpness_value(sharpness), -0.0001);
            let destination_x = params.destination.z + x;
            let destination_y = params.destination.w + y;
            sigma[destination_y * params.destination.x + destination_x] = 1.0 / quantized;
        }
    }
}
