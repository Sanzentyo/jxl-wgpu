override wg_x: u32 = 16u;
override wg_y: u32 = 16u;

struct Params {
    width: u32,
    height: u32,
    cb_stride: u32,
    y_stride: u32,
    cr_stride: u32,
    output_stride: u32,
    component: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> cb_plane: array<f32>;
@group(0) @binding(1) var<storage, read> y_plane: array<f32>;
@group(0) @binding(2) var<storage, read> cr_plane: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let cb = cb_plane[gid.y * params.cb_stride + gid.x];
    let y = y_plane[gid.y * params.y_stride + gid.x] + 128.0 / 255.0;
    let cr = cr_plane[gid.y * params.cr_stride + gid.x];
    var value: f32;
    switch params.component {
        case 0u: { value = fma(cr, 1.402, y); }
        case 1u: {
            value = fma(cr, -0.299 * 1.402 / 0.587,
                fma(cb, -0.114 * 1.772 / 0.587, y));
        }
        default: { value = fma(cb, 1.772, y); }
    }
    output[gid.y * params.output_stride + gid.x] = value;
}
