struct AdaptiveLfParams {
    // width, height, input vec4 offset, output vec4 offset
    extent_and_offsets: vec4<u32>,
    // X, Y, B channel thresholds; W is ABI padding.
    lf_scale: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> lf_input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> lf_output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: AdaptiveLfParams;

const TILE_WIDTH: u32 = 16u;
const HALO_WIDTH: u32 = 18u;
const HALO_AREA: u32 = 324u;
const SCALE_SELF: f32 = 0.052262735;
const SCALE_SIDE: f32 = 0.2034514;
const SCALE_DIAG: f32 = 0.03348292;

// 18 * 18 * 16 = 5184 bytes, comfortably below WebGPU's portable 16 KiB workgroup limit.
var<workgroup> tile: array<vec4<f32>, 324>;

fn clamped_input(tile_index: u32, workgroup_id: vec2<u32>) -> vec4<f32> {
    let width = params.extent_and_offsets.x;
    let height = params.extent_and_offsets.y;
    let tile_x = i32(tile_index % HALO_WIDTH) - 1i;
    let tile_y = i32(tile_index / HALO_WIDTH) - 1i;
    let source_x = u32(clamp(i32(workgroup_id.x * TILE_WIDTH) + tile_x, 0i, i32(width) - 1i));
    let source_y = u32(clamp(i32(workgroup_id.y * TILE_WIDTH) + tile_y, 0i, i32(height) - 1i));
    return lf_input[params.extent_and_offsets.z + source_y * width + source_x];
}
@compute @workgroup_size(16, 16, 1)
fn adaptive_lf_smoothing(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let width = params.extent_and_offsets.x;
    let height = params.extent_and_offsets.y;
    if (width == 0u || height == 0u) {
        return;
    }

    // All lanes participate in the halo load and barrier, including lanes outside an odd tail.
    for (var tile_index = lane; tile_index < HALO_AREA; tile_index += 256u) {
        tile[tile_index] = clamped_input(tile_index, workgroup_id.xy);
    }
    workgroupBarrier();

    let x = workgroup_id.x * TILE_WIDTH + local_id.x;
    let y = workgroup_id.y * TILE_WIDTH + local_id.y;
    if (x >= width || y >= height) {
        return;
    }
    let tile_index = (local_id.y + 1u) * HALO_WIDTH + local_id.x + 1u;
    let center = tile[tile_index];
    let output_index = params.extent_and_offsets.w + y * width + x;
    if (width <= 2u || height <= 2u || x == 0u || y == 0u || x + 1u == width || y + 1u == height) {
        lf_output[output_index] = center;
        return;
    }

    let side = tile[tile_index - 1u] + tile[tile_index + 1u]
        + tile[tile_index - HALO_WIDTH] + tile[tile_index + HALO_WIDTH];
    let diagonal = tile[tile_index - HALO_WIDTH - 1u]
        + tile[tile_index - HALO_WIDTH + 1u]
        + tile[tile_index + HALO_WIDTH - 1u]
        + tile[tile_index + HALO_WIDTH + 1u];
    let weighted = center * SCALE_SELF + side * SCALE_SIDE + diagonal * SCALE_DIAG;
    let gap_by_channel = abs(weighted.xyz - center.xyz) / params.lf_scale.xyz;
    let gap = max(0.5, max(gap_by_channel.x, max(gap_by_channel.y, gap_by_channel.z)));
    let gap_scale = max(3.0 - 4.0 * gap, 0.0);
    lf_output[output_index] = vec4<f32>(
        (weighted.xyz - center.xyz) * gap_scale + center.xyz,
        center.w,
    );
}
