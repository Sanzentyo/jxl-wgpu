override wg_x: u32 = 16u;
override wg_y: u32 = 16u;

struct Params {
    width: u32,
    height: u32,
    frame_width: u32,
    frame_height: u32,
    frame_stride: u32,
    reference_stride: u32,
    output_stride: u32,
    origin_x: i32,
    origin_y: i32,
    has_reference: u32,
    _pad0: u32,
    _pad1: u32,
};

// Copying the raw word preserves both i32 Modular samples and f32 render samples exactly.
@group(0) @binding(0) var<storage, read> frame: array<u32>;
@group(0) @binding(1) var<storage, read> reference: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let frame_x = i32(gid.x) - params.origin_x;
    let frame_y = i32(gid.y) - params.origin_y;
    let inside = frame_x >= 0 && frame_y >= 0
        && u32(frame_x) < params.frame_width
        && u32(frame_y) < params.frame_height;
    let output_index = gid.y * params.output_stride + gid.x;
    if (inside) {
        output[output_index] = frame[u32(frame_y) * params.frame_stride + u32(frame_x)];
    } else if (params.has_reference != 0u) {
        output[output_index] = reference[gid.y * params.reference_stride + gid.x];
    } else {
        output[output_index] = 0u;
    }
}
