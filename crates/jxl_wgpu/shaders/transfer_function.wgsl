struct Params {
    width: u32,
    height: u32,
    input_stride_r: u32,
    input_stride_g: u32,
    input_stride_b: u32,
    output_stride_r: u32,
    output_stride_g: u32,
    output_stride_b: u32,
    transfer: u32,
    gamma: f32,
    intensity_target: f32,
    min_nits: f32,
    luminance_rgb: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> input_r: array<f32>;
@group(0) @binding(1) var<storage, read> input_g: array<f32>;
@group(0) @binding(2) var<storage, read> input_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_r: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_g: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_b: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

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

fn linear_to_bt709(value: f32) -> f32 {
    let magnitude = abs(value);
    let encoded = select(
        1.099 * pow(magnitude, 0.45) - 0.099,
        4.5 * magnitude,
        magnitude <= 0.018,
    );
    return signed_value(encoded, value);
}

fn linear_to_pq(value: f32) -> f32 {
    let m1 = 2610.0 / 16384.0;
    let m2 = (2523.0 / 4096.0) * 128.0;
    let c1 = 3424.0 / 4096.0;
    let c2 = (2413.0 / 4096.0) * 32.0;
    let c3 = (2392.0 / 4096.0) * 32.0;
    let magnitude = abs(value);
    if magnitude == 0.0 {
        return 0.0;
    }
    let powered = pow(magnitude * params.intensity_target / 10000.0, m1);
    let encoded = pow((c1 + c2 * powered) / (1.0 + c3 * powered), m2);
    return signed_value(encoded, value);
}

fn scene_to_hlg(value: f32) -> f32 {
    let hlg_a = 0.17883277;
    let hlg_b = 1.0 - 4.0 * hlg_a;
    let hlg_c = 0.5599107295;
    let magnitude = abs(value);
    let encoded = select(
        hlg_a * log(12.0 * magnitude - hlg_b) + hlg_c,
        sqrt(3.0 * magnitude),
        magnitude <= 1.0 / 12.0,
    );
    return signed_value(encoded, value);
}

fn transfer_triplet(linear: vec3<f32>) -> vec3<f32> {
    switch params.transfer {
        case 0u: { return linear; }
        case 1u: {
            return vec3<f32>(
                linear_to_srgb(linear.x),
                linear_to_srgb(linear.y),
                linear_to_srgb(linear.z),
            );
        }
        case 2u: {
            return vec3<f32>(
                linear_to_bt709(linear.x),
                linear_to_bt709(linear.y),
                linear_to_bt709(linear.z),
            );
        }
        case 3u: {
            return vec3<f32>(
                linear_to_pq(linear.x),
                linear_to_pq(linear.y),
                linear_to_pq(linear.z),
            );
        }
        case 4u: {
            let system_gamma = 1.2 * pow(1.111, log2(params.intensity_target / 1000.0));
            let exponent = (1.0 - system_gamma) / system_gamma;
            var scene = linear;
            let luminance = dot(linear, params.luminance_rgb.xyz);
            if abs(exponent) >= 0.1 && luminance > 0.0 {
                scene *= pow(luminance, exponent);
            }
            return vec3<f32>(
                scene_to_hlg(scene.x),
                scene_to_hlg(scene.y),
                scene_to_hlg(scene.z),
            );
        }
        default: {
            return vec3<f32>(
                signed_value(pow(abs(linear.x), params.gamma), linear.x),
                signed_value(pow(abs(linear.y), params.gamma), linear.y),
                signed_value(pow(abs(linear.z), params.gamma), linear.z),
            );
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let linear = vec3<f32>(
        input_r[gid.y * params.input_stride_r + gid.x],
        input_g[gid.y * params.input_stride_g + gid.x],
        input_b[gid.y * params.input_stride_b + gid.x],
    );
    let encoded = transfer_triplet(linear);
    output_r[gid.y * params.output_stride_r + gid.x] = encoded.x;
    output_g[gid.y * params.output_stride_g + gid.x] = encoded.y;
    output_b[gid.y * params.output_stride_b + gid.x] = encoded.z;
}
