override wg_x: u32 = 64u;

struct ConvertParams {
    // width, height, pixel count, reserved
    geometry: vec4<u32>,
    // word offset, row stride, width, height for the final Y plane
    source_y: vec4<u32>,
    // word offset, row stride, width, height for the final X plane
    source_x: vec4<u32>,
    // word offset, row stride, width, height for the final B plane
    source_b: vec4<u32>,
    // F32 row strides for X, Y, and B outputs, reserved
    output_strides: vec4<u32>,
    // LF multipliers in X, Y, and B order; the kernel divides each by 128
    multipliers: vec4<f32>,
};

struct PackParams {
    // width, height, pixel count, reserved
    geometry: vec4<u32>,
    // F32 row strides for X, Y, and B input planes, reserved
    input_strides: vec4<u32>,
    // LF vec4 offset, LF vec4 row stride, reserved
    destination: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> arena: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_b: array<f32>;
@group(0) @binding(4) var<uniform> convert_params: ConvertParams;

@group(0) @binding(5) var<storage, read> input_x: array<f32>;
@group(0) @binding(6) var<storage, read> input_y: array<f32>;
@group(0) @binding(7) var<storage, read> input_b: array<f32>;
@group(0) @binding(8) var<storage, read_write> resources: array<vec4<f32>>;
@group(0) @binding(9) var<uniform> pack_params: PackParams;

fn saturating_add_i32(left: i32, right: i32) -> i32 {
    let left_bits = bitcast<u32>(left);
    let right_bits = bitcast<u32>(right);
    let sum_bits = left_bits + right_bits;
    let same_sign = ((left_bits ^ right_bits) & 0x80000000u) == 0u;
    let changed_sign = ((left_bits ^ sum_bits) & 0x80000000u) != 0u;
    if same_sign && changed_sign {
        if (left_bits & 0x80000000u) != 0u {
            return bitcast<i32>(0x80000000u);
        }
        return 0x7fffffffi;
    }
    return bitcast<i32>(sum_bits);
}

fn load_i32(plane: vec4<u32>, x: u32, y: u32) -> i32 {
    return bitcast<i32>(arena[plane.x + y * plane.y + x]);
}

@compute @workgroup_size(wg_x, 1, 1)
fn convert_modular(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    let width = convert_params.geometry.x;
    if index >= convert_params.geometry.z {
        return;
    }
    let y = index / width;
    let x = index - y * width;
    let source_y = load_i32(convert_params.source_y, x, y);
    let source_x = load_i32(convert_params.source_x, x, y);
    let source_b = saturating_add_i32(
        load_i32(convert_params.source_b, x, y),
        source_y,
    );
    let scale = 1.0 / 128.0;
    output_x[y * convert_params.output_strides.x + x] =
        f32(source_x) * convert_params.multipliers.x * scale;
    output_y[y * convert_params.output_strides.y + x] =
        f32(source_y) * convert_params.multipliers.y * scale;
    output_b[y * convert_params.output_strides.z + x] =
        f32(source_b) * convert_params.multipliers.z * scale;
}

@compute @workgroup_size(wg_x, 1, 1)
fn pack_lf(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    let width = pack_params.geometry.x;
    if index >= pack_params.geometry.z {
        return;
    }
    let y = index / width;
    let x = index - y * width;
    let output_index = pack_params.destination.x + y * pack_params.destination.y + x;
    resources[output_index] = vec4<f32>(
        input_x[y * pack_params.input_strides.x + x],
        input_y[y * pack_params.input_strides.y + x],
        input_b[y * pack_params.input_strides.z + x],
        0.0,
    );
}
