struct Params {
    width: u32,
    height: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    input_stride_b: u32,
    output_stride_x: u32,
    output_stride_y: u32,
    output_stride_b: u32,
    sigma_width: u32,
    sigma_height: u32,
    sigma_stride: u32,
    sigma_is_plane: u32,
    sigma_scale: f32,
    border_sad_mul: f32,
    channel_scale_x: f32,
    channel_scale_y: f32,
    channel_scale_b: f32,
    min_sigma: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input_x: array<f32>;
@group(0) @binding(1) var<storage, read> input_y: array<f32>;
@group(0) @binding(2) var<storage, read> input_b: array<f32>;
@group(0) @binding(3) var<storage, read> sigma_values: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_x: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_b: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

fn mirror_coordinate(value: i32, size: u32) -> u32 {
    if size <= 1u {
        return 0u;
    }
    var reflected = value;
    let signed_size = i32(size);
    loop {
        if reflected < 0 {
            reflected = -reflected - 1;
        } else if reflected >= signed_size {
            reflected = signed_size * 2 - reflected - 1;
        } else {
            break;
        }
    }
    return u32(reflected);
}

fn mirrored_index(x: i32, y: i32, stride: u32) -> u32 {
    let mirrored_x = mirror_coordinate(x, params.width);
    let mirrored_y = mirror_coordinate(y, params.height);
    return mirrored_y * stride + mirrored_x;
}

fn input_at(channel: u32, x: i32, y: i32) -> f32 {
    switch channel {
        case 0u: {
            return input_x[mirrored_index(x, y, params.input_stride_x)];
        }
        case 1u: {
            return input_y[mirrored_index(x, y, params.input_stride_y)];
        }
        default: {
            return input_b[mirrored_index(x, y, params.input_stride_b)];
        }
    }
}

fn channel_scale(channel: u32) -> f32 {
    switch channel {
        case 0u: { return params.channel_scale_x; }
        case 1u: { return params.channel_scale_y; }
        default: { return params.channel_scale_b; }
    }
}

fn write_output(channel: u32, x: u32, y: u32, value: f32) {
    switch channel {
        case 0u: {
            output_x[y * params.output_stride_x + x] = value;
        }
        case 1u: {
            output_y[y * params.output_stride_y + x] = value;
        }
        default: {
            output_b[y * params.output_stride_b + x] = value;
        }
    }
}

fn sigma_at(x: u32, y: u32) -> f32 {
    if params.sigma_is_plane == 0u {
        return sigma_values[0u];
    }
    // The host validates that the plane covers all 8x8 image blocks. The min
    // still prevents an out-of-range read if a future tiled dispatch includes
    // a dependency halo at the frame edge.
    let sigma_x = min(x / 8u, params.sigma_width - 1u);
    let sigma_y = min(y / 8u, params.sigma_height - 1u);
    return sigma_values[sigma_y * params.sigma_stride + sigma_x];
}

fn sad_multiplier(x: u32, y: u32) -> f32 {
    let sm = params.sigma_scale * 1.65;
    let x_in_block = x % 8u;
    let y_in_block = y % 8u;
    let border = x_in_block == 0u || x_in_block == 7u
        || y_in_block == 0u || y_in_block == 7u;
    return select(sm, sm * params.border_sad_mul, border);
}

fn epf_weight(sad: f32, inv_sigma: f32) -> f32 {
    return max(fma(sad, inv_sigma, 1.0), 0.0);
}

fn run_epf2(gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let sigma = sigma_at(gid.x, gid.y);
    if params.min_sigma > sigma {
        for (var channel = 0u; channel < 3u; channel++) {
            write_output(channel, gid.x, gid.y, input_at(channel, x, y));
        }
        return;
    }

    let inv_sigma = sigma * sad_multiplier(gid.x, gid.y);
    let x_center = input_at(0u, x, y);
    let y_center = input_at(1u, x, y);
    let b_center = input_at(2u, x, y);
    var weight_sum = 1.0;
    var x_acc = x_center;
    var y_acc = y_center;
    var b_acc = b_center;

    const OFFSETS = array<vec2<i32>, 4>(
        vec2<i32>(0, -1),
        vec2<i32>(-1, 0),
        vec2<i32>(1, 0),
        vec2<i32>(0, 1),
    );
    for (var i = 0u; i < 4u; i++) {
        let offset = OFFSETS[i];
        let cx = input_at(0u, x + offset.x, y + offset.y);
        let cy = input_at(1u, x + offset.x, y + offset.y);
        let cb = input_at(2u, x + offset.x, y + offset.y);
        let sad = fma(
            abs(cx - x_center),
            params.channel_scale_x,
            fma(
                abs(cy - y_center),
                params.channel_scale_y,
                abs(cb - b_center) * params.channel_scale_b,
            ),
        );
        let weight = epf_weight(sad, inv_sigma);
        weight_sum += weight;
        x_acc = fma(weight, cx, x_acc);
        y_acc = fma(weight, cy, y_acc);
        b_acc = fma(weight, cb, b_acc);
    }
    let inverse_weight = 1.0 / weight_sum;
    write_output(0u, gid.x, gid.y, x_acc * inverse_weight);
    write_output(1u, gid.x, gid.y, y_acc * inverse_weight);
    write_output(2u, gid.x, gid.y, b_acc * inverse_weight);
}

fn run_epf1(gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let sigma = sigma_at(gid.x, gid.y);
    if params.min_sigma > sigma {
        for (var channel = 0u; channel < 3u; channel++) {
            write_output(channel, gid.x, gid.y, input_at(channel, x, y));
        }
        return;
    }

    var sads: array<f32, 4>;
    for (var channel = 0u; channel < 3u; channel++) {
        let scale = channel_scale(channel);
        let p20 = input_at(channel, x, y - 2);
        let p11 = input_at(channel, x - 1, y - 1);
        let p21 = input_at(channel, x, y - 1);
        let p31 = input_at(channel, x + 1, y - 1);
        let p02 = input_at(channel, x - 2, y);
        let p12 = input_at(channel, x - 1, y);
        let p22 = input_at(channel, x, y);
        let p32 = input_at(channel, x + 1, y);
        let p42 = input_at(channel, x + 2, y);
        let p13 = input_at(channel, x - 1, y + 1);
        let p23 = input_at(channel, x, y + 1);
        let p33 = input_at(channel, x + 1, y + 1);
        let p24 = input_at(channel, x, y + 2);
        let d20_21 = abs(p20 - p21);
        let d11_21 = abs(p11 - p21);
        let d22_21 = abs(p22 - p21);
        let d31_21 = abs(p31 - p21);
        let d02_12 = abs(p02 - p12);
        let d11_12 = abs(p11 - p12);
        let d12_22 = abs(p22 - p12);
        let d31_32 = abs(p31 - p32);
        let d22_32 = abs(p22 - p32);
        let d42_32 = abs(p42 - p32);
        let d13_12 = abs(p13 - p12);
        let d22_23 = abs(p22 - p23);
        let d13_23 = abs(p13 - p23);
        let d33_23 = abs(p33 - p23);
        let d33_32 = abs(p33 - p32);
        let d24_23 = abs(p24 - p23);
        sads[0] = fma(d20_21 + d11_12 + d22_21 + d31_32 + d22_23, scale, sads[0]);
        sads[1] = fma(d11_21 + d02_12 + d12_22 + d22_32 + d13_23, scale, sads[1]);
        sads[2] = fma(d31_21 + d12_22 + d22_32 + d42_32 + d33_23, scale, sads[2]);
        sads[3] = fma(d22_21 + d13_12 + d22_23 + d33_32 + d24_23, scale, sads[3]);
    }

    let inv_sigma = sigma * sad_multiplier(gid.x, gid.y);
    var weight_sum = 1.0;
    for (var i = 0u; i < 4u; i++) {
        sads[i] = epf_weight(sads[i], inv_sigma);
        weight_sum += sads[i];
    }
    let inverse_weight = 1.0 / weight_sum;
    for (var channel = 0u; channel < 3u; channel++) {
        var value = input_at(channel, x, y);
        value = fma(input_at(channel, x, y + 1), sads[3], value);
        value = fma(input_at(channel, x + 1, y), sads[2], value);
        value = fma(input_at(channel, x - 1, y), sads[1], value);
        value = fma(input_at(channel, x, y - 1), sads[0], value);
        write_output(channel, gid.x, gid.y, value * inverse_weight);
    }
}

fn run_epf0(gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let sigma = sigma_at(gid.x, gid.y);
    if params.min_sigma > sigma {
        for (var channel = 0u; channel < 3u; channel++) {
            write_output(channel, gid.x, gid.y, input_at(channel, x, y));
        }
        return;
    }

    var sads: array<f32, 12>;
    for (var channel = 0u; channel < 3u; channel++) {
        let scale = channel_scale(channel);
        let p30 = input_at(channel, x, y - 3);
        let p21 = input_at(channel, x - 1, y - 2);
        let p31 = input_at(channel, x, y - 2);
        let p41 = input_at(channel, x + 1, y - 2);
        let p12 = input_at(channel, x - 2, y - 1);
        let p22 = input_at(channel, x - 1, y - 1);
        let p32 = input_at(channel, x, y - 1);
        let p42 = input_at(channel, x + 1, y - 1);
        let p52 = input_at(channel, x + 2, y - 1);
        let p03 = input_at(channel, x - 3, y);
        let p13 = input_at(channel, x - 2, y);
        let p23 = input_at(channel, x - 1, y);
        let p33 = input_at(channel, x, y);
        let p43 = input_at(channel, x + 1, y);
        let p53 = input_at(channel, x + 2, y);
        let p63 = input_at(channel, x + 3, y);
        let p14 = input_at(channel, x - 2, y + 1);
        let p24 = input_at(channel, x - 1, y + 1);
        let p34 = input_at(channel, x, y + 1);
        let p44 = input_at(channel, x + 1, y + 1);
        let p54 = input_at(channel, x + 2, y + 1);
        let p25 = input_at(channel, x - 1, y + 2);
        let p35 = input_at(channel, x, y + 2);
        let p45 = input_at(channel, x + 1, y + 2);
        let p36 = input_at(channel, x, y + 3);
        let d32_30 = abs(p32 - p30);
        let d32_21 = abs(p32 - p21);
        let d32_31 = abs(p32 - p31);
        let d32_41 = abs(p32 - p41);
        let d32_12 = abs(p32 - p12);
        let d32_22 = abs(p32 - p22);
        let d32_42 = abs(p32 - p42);
        let d32_52 = abs(p32 - p52);
        let d32_23 = abs(p32 - p23);
        let d32_34 = abs(p32 - p34);
        let d32_43 = abs(p32 - p43);
        let d32_33 = abs(p32 - p33);
        let d23_21 = abs(p23 - p21);
        let d23_12 = abs(p23 - p12);
        let d23_22 = abs(p23 - p22);
        let d23_03 = abs(p23 - p03);
        let d23_13 = abs(p23 - p13);
        let d23_33 = abs(p23 - p33);
        let d23_43 = abs(p23 - p43);
        let d23_14 = abs(p23 - p14);
        let d23_24 = abs(p23 - p24);
        let d23_34 = abs(p23 - p34);
        let d23_25 = abs(p23 - p25);
        let d33_31 = abs(p33 - p31);
        let d33_22 = abs(p33 - p22);
        let d33_42 = abs(p33 - p42);
        let d33_13 = abs(p33 - p13);
        let d33_43 = abs(p33 - p43);
        let d33_53 = abs(p33 - p53);
        let d33_24 = abs(p33 - p24);
        let d33_34 = abs(p33 - p34);
        let d33_44 = abs(p33 - p44);
        let d33_35 = abs(p33 - p35);
        let d43_41 = abs(p43 - p41);
        let d43_42 = abs(p43 - p42);
        let d43_52 = abs(p43 - p52);
        let d43_53 = abs(p43 - p53);
        let d43_63 = abs(p43 - p63);
        let d43_34 = abs(p43 - p34);
        let d43_44 = abs(p43 - p44);
        let d43_54 = abs(p43 - p54);
        let d43_45 = abs(p43 - p45);
        let d34_14 = abs(p34 - p14);
        let d34_24 = abs(p34 - p24);
        let d34_44 = abs(p34 - p44);
        let d34_54 = abs(p34 - p54);
        let d34_25 = abs(p34 - p25);
        let d34_35 = abs(p34 - p35);
        let d34_45 = abs(p34 - p45);
        let d34_36 = abs(p34 - p36);
        sads[0] = fma(d32_30 + d23_21 + d33_31 + d43_41 + d32_34, scale, sads[0]);
        sads[1] = fma(d32_21 + d23_12 + d33_22 + d32_43 + d23_34, scale, sads[1]);
        sads[2] = fma(d32_31 + d23_22 + d32_33 + d43_42 + d33_34, scale, sads[2]);
        sads[3] = fma(d32_41 + d32_23 + d33_42 + d43_52 + d43_34, scale, sads[3]);
        sads[4] = fma(d32_12 + d23_03 + d33_13 + d23_43 + d34_14, scale, sads[4]);
        sads[5] = fma(d32_22 + d23_13 + d23_33 + d33_43 + d34_24, scale, sads[5]);
        sads[6] = fma(d32_42 + d23_33 + d33_43 + d43_53 + d34_44, scale, sads[6]);
        sads[7] = fma(d32_52 + d23_43 + d33_53 + d43_63 + d34_54, scale, sads[7]);
        sads[8] = fma(d32_23 + d23_14 + d33_24 + d43_34 + d34_25, scale, sads[8]);
        sads[9] = fma(d32_33 + d23_24 + d33_34 + d43_44 + d34_35, scale, sads[9]);
        sads[10] = fma(d32_43 + d23_34 + d33_44 + d43_54 + d34_45, scale, sads[10]);
        sads[11] = fma(d32_34 + d23_25 + d33_35 + d43_45 + d34_36, scale, sads[11]);
    }

    let inv_sigma = sigma * sad_multiplier(gid.x, gid.y);
    var weight_sum = 1.0;
    for (var i = 0u; i < 12u; i++) {
        sads[i] = epf_weight(sads[i], inv_sigma);
        weight_sum += sads[i];
    }
    let inverse_weight = 1.0 / weight_sum;
    const OFFSETS = array<vec2<i32>, 12>(
        vec2<i32>(0, -2),
        vec2<i32>(-1, -1),
        vec2<i32>(0, -1),
        vec2<i32>(1, -1),
        vec2<i32>(-2, 0),
        vec2<i32>(-1, 0),
        vec2<i32>(1, 0),
        vec2<i32>(2, 0),
        vec2<i32>(-1, 1),
        vec2<i32>(0, 1),
        vec2<i32>(1, 1),
        vec2<i32>(0, 2),
    );
    for (var channel = 0u; channel < 3u; channel++) {
        var value = input_at(channel, x, y);
        for (var reverse = 12u; reverse > 0u; reverse--) {
            let i = reverse - 1u;
            let offset = OFFSETS[i];
            value = fma(input_at(channel, x + offset.x, y + offset.y), sads[i], value);
        }
        write_output(channel, gid.x, gid.y, value * inverse_weight);
    }
}

@compute @workgroup_size(16, 16, 1)
fn epf0(@builtin(global_invocation_id) gid: vec3<u32>) {
    run_epf0(gid);
}

@compute @workgroup_size(16, 16, 1)
fn epf1(@builtin(global_invocation_id) gid: vec3<u32>) {
    run_epf1(gid);
}

@compute @workgroup_size(16, 16, 1)
fn epf2(@builtin(global_invocation_id) gid: vec3<u32>) {
    run_epf2(gid);
}
