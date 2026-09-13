override wg_x: u32 = 16u;
override wg_y: u32 = 16u;

struct Params {
    extent_channels: vec4<u32>,
    input_offsets: vec4<u32>,
    input_strides: vec4<u32>,
    output_offsets: vec4<u32>,
    output_strides: vec4<u32>,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> program: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn parameter(base: u32, index: u32) -> f32 { return bitcast<f32>(program[base + 4u + index]); }
fn sample_value(base: u32, index: u32) -> f32 { return f32(program[base + 12u + index]) / 65535.0; }

struct SamplePosition { left: u32, weight: f32 }

// Full u32 product as (low, high), using 16-bit partial products. Every carry fits u32.
fn multiply_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 65535u;
    let a1 = a >> 16u;
    let b0 = b & 65535u;
    let b1 = b >> 16u;
    let first = a0 * b0;
    let second = a1 * b0 + (first >> 16u);
    let third = a0 * b1 + (second & 65535u);
    return vec2<u32>((third << 16u) | (first & 65535u), a1 * b1 + (second >> 16u) + (third >> 16u));
}

fn binary_fraction(word: u32, shift: u32) -> f32 {
    // Split the exponent so both factors remain normal even for subnormal input coordinates.
    let first = min(shift, 100u);
    let second = shift - first;
    return (f32(word) * bitcast<f32>((127u - first) << 23u)) * bitcast<f32>((127u - second) << 23u);
}

fn sample_position(x: f32, intervals: u32) -> SamplePosition {
    // x is strictly between zero and one. Preserve its exact significand product; a rounded
    // F32 x*intervals can move a pixel across a sharp table edge or lose its interpolation weight.
    let bits = bitcast<u32>(x);
    let exponent = bits >> 23u;
    let significand = (bits & 0x7fffffu) | select(0x800000u, 0u, exponent == 0u);
    let shift = select(150u - exponent, 149u, exponent == 0u);
    let product = multiply_wide(significand, intervals);
    if shift < 32u {
        let left = (product.y << (32u - shift)) | (product.x >> shift);
        let remainder = product.x & ((1u << shift) - 1u);
        return SamplePosition(left, binary_fraction(remainder, shift));
    }
    if shift >= 64u {
        return SamplePosition(0u, binary_fraction(product.y, shift - 32u) + binary_fraction(product.x, shift));
    }
    let high_shift = shift - 32u;
    let left = product.y >> high_shift;
    let remainder = product.y & ((1u << high_shift) - 1u);
    return SamplePosition(left, binary_fraction(remainder, high_shift) + binary_fraction(product.x, shift));
}

fn forward_curve(base: u32, value: f32) -> f32 {
    let x = clamp(value, 0.0, 1.0);
    let mode = program[base];
    if mode == 0u { return x; }
    let g = parameter(base, 0u);
    if mode == 1u { return pow(x, g); }
    if mode == 2u {
        let count = program[base + 2u];
        if x == 0.0 { return sample_value(base, 0u); }
        if x == 1.0 { return sample_value(base, count - 1u); }
        let position = sample_position(x, count - 1u);
        return mix(sample_value(base, position.left), sample_value(base, position.left + 1u), position.weight);
    }
    let function = program[base + 1u];
    if function == 0u { return pow(x, g); }
    let a = parameter(base, 1u);
    let b = parameter(base, 2u);
    let c = parameter(base, 3u);
    if function <= 2u {
        var y = 0.0;
        if x >= -b / a { y = pow(max(0.0, a * x + b), g); }
        if function == 2u { y += c; }
        return clamp(y, 0.0, 1.0);
    }
    let d = parameter(base, 4u);
    if x < d {
        var y = c * x;
        if function == 4u { y += parameter(base, 6u); }
        return clamp(y, 0.0, 1.0);
    }
    var y = pow(max(0.0, a * x + b), g);
    if function == 4u { y += parameter(base, 5u); }
    return clamp(y, 0.0, 1.0);
}

fn ordered_sample(base: u32, index: u32, decreasing: bool) -> f32 {
    let value = sample_value(base, index);
    return select(value, 1.0 - value, decreasing);
}

fn inverse_samples(base: u32, value: f32) -> f32 {
    let count = program[base + 2u];
    let decreasing = program[base + 3u] != 0u;
    let query = clamp(select(value, 1.0 - value, decreasing), ordered_sample(base, 0u, decreasing), ordered_sample(base, count - 1u, decreasing));
    let terminal = query >= ordered_sample(base, count - 1u, decreasing);
    var low = 0u;
    var high = count;
    // Annex F.1: interior plateaus select their last x, final plateaus their first x.
    while low < high {
        let middle = low + (high - low) / 2u;
        let sample = ordered_sample(base, middle, decreasing);
        if sample < query || (sample == query && !terminal) { low = middle + 1u; }
        else { high = middle; }
    }
    if terminal { return f32(low) / f32(count - 1u); }
    if low == 0u { return 0.0; }
    if low >= count { return 1.0; }
    let a = ordered_sample(base, low - 1u, decreasing);
    let b = ordered_sample(base, low, decreasing);
    return (f32(low - 1u) + (query - a) / (b - a)) / f32(count - 1u);
}

fn inverse_curve(base: u32, value: f32) -> f32 {
    let mode = program[base];
    if mode == 0u { return clamp(value, 0.0, 1.0); }
    if mode == 1u { return pow(clamp(value, 0.0, 1.0), 1.0 / parameter(base, 0u)); }
    if mode == 2u { return inverse_samples(base, value); }
    let function = program[base + 1u];
    let query = clamp(value, 0.0, 1.0);
    let g = parameter(base, 0u);
    if function == 0u { return pow(query, 1.0 / g); }
    let a = parameter(base, 1u);
    let b = parameter(base, 2u);
    let c = parameter(base, 3u);
    var offset = 0.0;
    if function == 2u { offset = c; }
    if function == 4u { offset = parameter(base, 5u); }
    // Solve the mathematical curve. Searching the rounded F32 forward function would
    // invent a black plateau whenever a small power rounds away when adding an offset.
    let power_root = (pow(max(0.0, query - offset), 1.0 / g) - b) / a;
    if function <= 2u { return clamp(power_root, 0.0, 1.0); }
    let d = parameter(base, 4u);
    var lower_offset = 0.0;
    if function == 4u { lower_offset = parameter(base, 6u); }
    if d <= 0.0 { return clamp(power_root, 0.0, 1.0); }
    if d > 1.0 { return clamp((query - lower_offset) / c, 0.0, 1.0); }
    let lower_limit = clamp(c * d + lower_offset, 0.0, 1.0);
    let upper_start = forward_curve(base, d);
    let upper_x = clamp(power_root, d, 1.0);
    // d is positive and finite here. Its previous representable value remains on the
    // lower branch across an upward jump, including a constant lower segment.
    let below_d = bitcast<f32>(bitcast<u32>(d) - 1u);
    var lower_x = below_d;
    if c > 0.0 { lower_x = clamp((query - lower_offset) / c, 0.0, below_d); }
    if query >= upper_start {
        if query >= forward_curve(base, 1.0) && query <= lower_limit { return lower_x; }
        return upper_x;
    }
    if query <= lower_limit || query - lower_limit <= upper_start - query { return lower_x; }
    return upper_x;
}

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.extent_channels.x || id.y >= params.extent_channels.y { return; }
    var linear = vec3<f32>(0.0);
    for (var channel = 0u; channel < params.extent_channels.z; channel++) {
        let value = input[params.input_offsets[channel] + id.y * params.input_strides[channel] + id.x];
        linear[channel] = forward_curve(program[12u + channel], value);
    }
    for (var channel = 0u; channel < params.extent_channels.w; channel++) {
        let base = channel * 4u;
        let row = vec3<f32>(bitcast<f32>(program[base]), bitcast<f32>(program[base + 1u]), bitcast<f32>(program[base + 2u]));
        output[params.output_offsets[channel] + id.y * params.output_strides[channel] + id.x] = inverse_curve(program[16u + channel], dot(row, linear));
    }
}
