struct Pixel {
    position: u32,
    value_x: f32,
    value_y: f32,
    value_b: f32,
};

struct Params {
    pixel_count: u32,
    output_width: u32,
    output_stride_x: u32,
    output_stride_y: u32,
    output_stride_b: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> pixels: array<Pixel>;
@group(0) @binding(1) var<storage, read_write> output_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_b: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.pixel_count) {
        return;
    }
    let pixel = pixels[gid.x];
    let x = pixel.position % params.output_width;
    let y = pixel.position / params.output_width;
    output_x[y * params.output_stride_x + x] = pixel.value_x;
    output_y[y * params.output_stride_y + x] = pixel.value_y;
    output_b[y * params.output_stride_b + x] = pixel.value_b;
}
