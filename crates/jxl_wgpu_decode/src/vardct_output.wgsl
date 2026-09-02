override wg_x: u32 = 256u;
override wg_y: u32 = 1u;

struct Params {
    image: vec4<u32>,
    dispatch: vec4<u32>,
    plane_geometry: array<vec4<u32>, 3>,
    matrix_r: vec4<f32>,
    matrix_g: vec4<f32>,
    matrix_b: vec4<f32>,
    bias_cbrt: vec4<f32>,
    scaled_bias: vec4<f32>,
    intensity_scale: f32,
    format_selector: u32,
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

fn quantize_encoded8(value: f32) -> u32 {
    return u32(clamp(floor(value * 255.0 + 0.5), 0.0, 255.0));
}

fn plane_value(channel: u32, x: u32, y: u32) -> f32 {
    let index = y * params.plane_geometry[channel].x + x;
    if (channel == 0u) { return x_plane[index]; }
    if (channel == 1u) { return y_plane[index]; }
    return b_plane[index];
}

fn jpeg_sample(channel: u32, x: u32, y: u32) -> f32 {
    let geometry = params.plane_geometry[channel];
    let horizontal_shift = geometry.w & 1u;
    let vertical_shift = (geometry.w >> 1u) & 1u;
    var x0 = x;
    var x1 = x;
    var x_weight = 0.0;
    if (horizontal_shift != 0u) {
        let center = x >> 1u;
        if ((x & 1u) == 0u) {
            x0 = select(0u, center - 1u, center != 0u);
            x1 = center;
            x_weight = 0.75;
        } else {
            x0 = center;
            x1 = min(center + 1u, geometry.y - 1u);
            x_weight = 0.25;
        }
    }
    var y0 = y;
    var y1 = y;
    var y_weight = 0.0;
    if (vertical_shift != 0u) {
        let center = y >> 1u;
        if ((y & 1u) == 0u) {
            y0 = select(0u, center - 1u, center != 0u);
            y1 = center;
            y_weight = 0.75;
        } else {
            y0 = center;
            y1 = min(center + 1u, geometry.z - 1u);
            y_weight = 0.25;
        }
    }
    let top = mix(plane_value(channel, x0, y0), plane_value(channel, x1, y0), x_weight);
    let bottom = mix(plane_value(channel, x0, y1), plane_value(channel, x1, y1), x_weight);
    return mix(top, bottom, y_weight);
}

fn rgb8_at(pixel: u32) -> vec3<u32> {
    if (pixel >= params.image.z) {
        return vec3<u32>(0u);
    }

    let row = pixel / params.image.x;
    let column = pixel - row * params.image.x;
    if (params.dispatch.y == 1u) {
        let cb = jpeg_sample(0u, column, row);
        let y = jpeg_sample(1u, column, row) + 128.0 / 255.0;
        let cr = jpeg_sample(2u, column, row);
        return vec3<u32>(
            quantize_encoded8(y + 1.402 * cr),
            quantize_encoded8(y - (0.114 * 1.772 / 0.587) * cb - (0.299 * 1.402 / 0.587) * cr),
            quantize_encoded8(y + 1.772 * cb),
        );
    }
    let x = plane_value(0u, column, row);
    let y = plane_value(1u, column, row);
    let b = plane_value(2u, column, row);

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

const FORMAT_RGB8: u32 = 0u;
const FORMAT_RGBA8: u32 = 1u;
const FORMAT_BGR8: u32 = 2u;
const FORMAT_BGRA8: u32 = 3u;

fn pack_rgb_word(word_index: u32, is_bgr: bool) -> u32 {
    let first_byte = word_index * 4u;
    let first_pixel = first_byte / 3u;
    let phase = first_byte - first_pixel * 3u;
    var first = rgb8_at(first_pixel);
    var second = rgb8_at(first_pixel + 1u);
    if (is_bgr) {
        first = first.bgr;
        second = second.bgr;
    }

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
    let index = gid.y * params.dispatch.x + gid.x;
    if (params.format_selector == FORMAT_RGBA8 || params.format_selector == FORMAT_BGRA8) {
        if (index >= params.image.z) {
            return;
        }
        let color = rgb8_at(index);
        let is_bgr = params.format_selector == FORMAT_BGRA8;
        let r = select(color.r, color.b, is_bgr);
        let g = color.g;
        let b = select(color.b, color.r, is_bgr);
        output_words[index] = r | (g << 8u) | (b << 16u) | (255u << 24u);
    } else {
        if (index >= params.image.w) {
            return;
        }
        let is_bgr = params.format_selector == FORMAT_BGR8;
        output_words[index] = pack_rgb_word(index, is_bgr);
    }
}
