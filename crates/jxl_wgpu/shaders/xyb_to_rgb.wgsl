struct Params {
    width: u32,
    height: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    input_stride_b: u32,
    output_stride_r: u32,
    output_stride_g: u32,
    output_stride_b: u32,
    matrix_r: vec4<f32>,
    matrix_g: vec4<f32>,
    matrix_b: vec4<f32>,
    bias_cbrt: vec4<f32>,
    scaled_bias: vec4<f32>,
    intensity_scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> x_plane: array<f32>;
@group(0) @binding(1) var<storage, read> y_plane: array<f32>;
@group(0) @binding(2) var<storage, read> b_plane: array<f32>;
@group(0) @binding(3) var<storage, read_write> r_plane: array<f32>;
@group(0) @binding(4) var<storage, read_write> g_plane: array<f32>;
@group(0) @binding(5) var<storage, read_write> rgb_b_plane: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let x = x_plane[gid.y * params.input_stride_x + gid.x];
    let y = y_plane[gid.y * params.input_stride_y + gid.x];
    let b = b_plane[gid.y * params.input_stride_b + gid.x];

    // JPEG XL's XYB inverse first reconstructs biased LMS, then applies the
    // sign-preserving cube and the stream-selected inverse opsin matrix.
    let mixed = vec3<f32>(
        y + x - params.bias_cbrt.x,
        y - x - params.bias_cbrt.y,
        b - params.bias_cbrt.z,
    );
    let lms = mixed * mixed * (mixed * params.intensity_scale)
        + params.scaled_bias.xyz;
    let rgb = vec3<f32>(
        dot(params.matrix_r.xyz, lms),
        dot(params.matrix_g.xyz, lms),
        dot(params.matrix_b.xyz, lms),
    );

    r_plane[gid.y * params.output_stride_r + gid.x] = rgb.x;
    g_plane[gid.y * params.output_stride_g + gid.x] = rgb.y;
    rgb_b_plane[gid.y * params.output_stride_b + gid.x] = rgb.z;
}
