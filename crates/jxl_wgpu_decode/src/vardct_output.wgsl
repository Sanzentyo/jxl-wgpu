override wg_x: u32 = 256u;
override wg_y: u32 = 1u;

struct Params {
    width: u32,
    height: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    input_stride_b: u32,
    pixel_count: u32,
    output_word_count: u32,
    dispatch_width: u32,
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
@group(0) @binding(3) var<storage, read_write> output_words: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

fn signed_value(magnitude: f32, source: f32) -> f32 {
    return select(magnitude, -magnitude, source < 0.0);
}

fn linear_to_srgb(value: f32) -> f32 {
    let magnitude = abs(value);
    let encoded = select(
        1.055 * pow(magnitude, 1.0 / 2.4) - 0.055,
        12.92 * magnitude,
        magnitude <= 0.0031308,
    );
    return signed_value(encoded, value);
}

fn quantize_srgb8(value: f32) -> u32 {
    return u32(clamp(floor(linear_to_srgb(value) * 255.0 + 0.5), 0.0, 255.0));
}

fn rgb8_at(pixel: u32) -> vec3<u32> {
    if (pixel >= params.pixel_count) {
        return vec3<u32>(0u);
    }

    let row = pixel / params.width;
    let column = pixel - row * params.width;
    let x = x_plane[row * params.input_stride_x + column];
    let y = y_plane[row * params.input_stride_y + column];
    let b = b_plane[row * params.input_stride_b + column];

    // This is deliberately identical to jxl_wgpu's XYB inverse contract:
    // reconstruct biased LMS, apply the sign-preserving cube, then the
    // codestream-selected inverse opsin matrix.
    let mixed = vec3<f32>(
        y + x - params.bias_cbrt.x,
        y - x - params.bias_cbrt.y,
        b - params.bias_cbrt.z,
    );
    let lms = mixed * mixed * (mixed * params.intensity_scale)
        + params.scaled_bias.xyz;
    let linear_rgb = vec3<f32>(
        dot(params.matrix_r.xyz, lms),
        dot(params.matrix_g.xyz, lms),
        dot(params.matrix_b.xyz, lms),
    );
    return vec3<u32>(
        quantize_srgb8(linear_rgb.r),
        quantize_srgb8(linear_rgb.g),
        quantize_srgb8(linear_rgb.b),
    );
}

fn pack_rgb8_word(word_index: u32) -> u32 {
    let first_byte = word_index * 4u;
    let first_pixel = first_byte / 3u;
    let phase = first_byte - first_pixel * 3u;
    let first = rgb8_at(first_pixel);
    let second = rgb8_at(first_pixel + 1u);

    if (phase == 0u) {
        return first.r
            | (first.g << 8u)
            | (first.b << 16u)
            | (second.r << 24u);
    }
    if (phase == 1u) {
        return first.g
            | (first.b << 8u)
            | (second.r << 16u)
            | (second.g << 24u);
    }
    return first.b
        | (second.r << 8u)
        | (second.g << 16u)
        | (second.b << 24u);
}

@compute @workgroup_size(wg_x, wg_y, 1)
fn pack_rgb8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_index = gid.y * params.dispatch_width + gid.x;
    if (word_index >= params.output_word_count) {
        return;
    }
    output_words[word_index] = pack_rgb8_word(word_index);
}
