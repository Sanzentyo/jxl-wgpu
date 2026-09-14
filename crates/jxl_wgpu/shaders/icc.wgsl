const FORWARD_CURVES: u32 = 0u;
const INVERSE_CURVES: u32 = 1u;
const AFFINE: u32 = 2u;
const CLUT: u32 = 3u;
const SEGMENTED_CURVES: u32 = 4u;
const LAB_TO_XYZ: u32 = 5u;
const XYZ_TO_LAB: u32 = 6u;
const CLAMPED_AFFINE: u32 = 7u;

override wg_x: u32 = 16u;
override wg_y: u32 = 16u;

struct Params {
    extent_channels: vec4<u32>,
    input_offsets: array<vec4<u32>, 4>,
    input_strides: array<vec4<u32>, 4>,
    output_offsets: array<vec4<u32>, 4>,
    output_strides: array<vec4<u32>, 4>,
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

// MPE formulas are not bounded ICC device TRCs. Integer powers retain negative bases.
fn real_power(x: f32, g: f32) -> f32 {
    if g == 0.0 { return 1.0; }
    if g == 1.0 { return x; }
    if x == 0.0 && g > 0.0 { return 0.0; }
    if x < 0.0 {
        let odd = g - 2.0 * floor(g * 0.5) != 0.0;
        return select(1.0, -1.0, odd) * pow(-x, g);
    }
    return pow(x, g);
}

fn float_order(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    if (bits & 0x7fffffffu) == 0u { return 0x80000000u; }
    return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn segmented_curve(base: u32, x: f32) -> f32 {
    let count = program[base];
    var low = 0u;
    var high = count - 1u;
    // First containing segment owns a breakpoint, including repeated breakpoints.
    while low < high {
        let mid = low + (high - low) / 2u;
        // Numeric F32 comparisons may flush subnormals on portable GPUs. A curve
        // can jump at zero, so losing the original side of the breakpoint is not safe.
        if float_order(x) <= float_order(bitcast<f32>(program[base + 1u + mid * 10u])) { high = mid; }
        else { low = mid + 1u; }
    }
    let record = base + 1u + low * 10u;
    let mode = program[record + 2u];
    if mode == 3u {
        let lower = bitcast<f32>(program[record + 1u]);
        let upper = bitcast<f32>(program[record]);
        let count = program[record + 3u];
        let samples = program[record + 9u];
        let t = clamp((x - lower) / (upper - lower), 0.0, 1.0);
        if t == 0.0 { return bitcast<f32>(program[samples]); }
        if t == 1.0 { return bitcast<f32>(program[samples + count - 1u]); }
        let position = sample_position(t, count - 1u);
        return mix(bitcast<f32>(program[samples + position.left]), bitcast<f32>(program[samples + position.left + 1u]), position.weight);
    }
    let p0 = bitcast<f32>(program[record + 4u]);
    let p1 = bitcast<f32>(program[record + 5u]);
    let p2 = bitcast<f32>(program[record + 6u]);
    let p3 = bitcast<f32>(program[record + 7u]);
    let p4 = bitcast<f32>(program[record + 8u]);
    if mode == 0u { return real_power(p1 * x + p2, p0) + p3; }
    if mode == 1u { return p1 * log2(p2 * real_power(x, p0) + p3) / log2(10.0) + p4; }
    return p0 * real_power(p1, p2 * x + p3) + p4;
}

fn clut_value(base: u32, dimensions: u32, channel: u32, values: ptr<function, array<f32, 16>>) -> f32 {
    var weights: array<f32, 16>;
    var strides: array<u32, 16>;
    var origin = base + dimensions * 2u + channel;
    for (var axis = 0u; axis < dimensions; axis++) {
        let grid = program[base + axis * 2u];
        let stride = program[base + axis * 2u + 1u];
        let value = clamp((*values)[axis], 0.0, 1.0);
        var position = SamplePosition(0u, 0.0);
        if value == 1.0 { position = SamplePosition(grid - 2u, 1.0); }
        else if value > 0.0 { position = sample_position(value, grid - 1u); }
        weights[axis] = position.weight;
        strides[axis] = stride;
        origin += position.left * stride;
    }
    // One/two dimensions use linear/bilinear interpolation. Higher dimensions use
    // tetrahedra on the last three axes and linear interpolation on preceding axes.
    let tail = min(dimensions, 3u);
    let leading = dimensions - tail;
    var order = array<u32, 3>(leading, leading + 1u, leading + 2u);
    if tail == 3u {
        for (var i = 1u; i < 3u; i++) {
            var j = i;
            while j > 0u && weights[order[j]] > weights[order[j - 1u]] {
                let swap = order[j]; order[j] = order[j - 1u]; order[j - 1u] = swap;
                j--;
            }
        }
    }
    var result = 0.0;
    for (var corner = 0u; corner < (1u << leading); corner++) {
        var index = origin;
        var weight = 1.0;
        for (var axis = 0u; axis < leading; axis++) {
            let upper = (corner & (1u << axis)) != 0u;
            if upper { index += strides[axis]; }
            weight *= select(1.0 - weights[axis], weights[axis], upper);
        }
        var value = 0.0;
        if tail == 3u {
            let a = bitcast<f32>(program[index]);
            index += strides[order[0]];
            let b = bitcast<f32>(program[index]);
            index += strides[order[1]];
            let c = bitcast<f32>(program[index]);
            index += strides[order[2]];
            let d = bitcast<f32>(program[index]);
            value = a + weights[order[0]] * (b - a) + weights[order[1]] * (c - b) + weights[order[2]] * (d - c);
        } else {
            for (var point = 0u; point < (1u << tail); point++) {
                var address = index;
                var factor = 1.0;
                for (var axis = 0u; axis < tail; axis++) {
                    let upper = (point & (1u << axis)) != 0u;
                    if upper { address += strides[axis]; }
                    factor *= select(1.0 - weights[axis], weights[axis], upper);
                }
                value += factor * bitcast<f32>(program[address]);
            }
        }
        result += weight * value;
    }
    return result;
}

fn lab_f(t: f32) -> f32 {
    if t > 216.0 / 24389.0 { return pow(t, 1.0 / 3.0); }
    return ((24389.0 / 27.0) * t + 16.0) / 116.0;
}
fn lab_inverse(t: f32) -> f32 {
    if t > 6.0 / 29.0 { return t * t * t; }
    return (116.0 * t - 16.0) * (27.0 / 24389.0);
}

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.extent_channels.x || id.y >= params.extent_channels.y { return; }
    var values: array<f32, 16>;
    for (var c = 0u; c < params.extent_channels.z; c++) {
        values[c] = input[params.input_offsets[c / 4u][c % 4u] + id.y * params.input_strides[c / 4u][c % 4u] + id.x];
    }
    for (var stage = 0u; stage < program[0]; stage++) {
        let record = 4u + stage * 4u;
        let opcode = program[record];
        let p = program[record + 1u];
        let q = program[record + 2u];
        let base = program[record + 3u];
        var next: array<f32, 16>;
        for (var c = 0u; c < q; c++) {
            if opcode == FORWARD_CURVES { next[c] = forward_curve(program[base + c], values[c]); }
            else if opcode == INVERSE_CURVES { next[c] = inverse_curve(program[base + c], values[c]); }
            else if opcode == AFFINE || opcode == CLAMPED_AFFINE {
                let row = base + c * (p + 1u);
                var value = 0.0;
                if p == 3u {
                    value = dot(vec3<f32>(bitcast<f32>(program[row]), bitcast<f32>(program[row + 1u]), bitcast<f32>(program[row + 2u])), vec3<f32>(values[0], values[1], values[2]));
                } else {
                    for (var i = 0u; i < p; i++) { value += bitcast<f32>(program[row + i]) * values[i]; }
                }
                value += bitcast<f32>(program[row + p]);
                if opcode == CLAMPED_AFFINE { value = clamp(value, 0.0, 1.0); }
                next[c] = value;
            }
            else if opcode == CLUT { next[c] = clut_value(base, p, c, &values); }
            else if opcode == SEGMENTED_CURVES { next[c] = segmented_curve(program[base + c], values[c]); }
        }
        if opcode == LAB_TO_XYZ {
            let fy = (values[0] + 16.0) / 116.0;
            next[0] = 0.9642 * lab_inverse(fy + values[1] / 500.0);
            next[1] = lab_inverse(fy);
            next[2] = 0.8249 * lab_inverse(fy - values[2] / 200.0);
        } else if opcode == XYZ_TO_LAB {
            let fx = lab_f(values[0] / 0.9642);
            let fy = lab_f(values[1]);
            let fz = lab_f(values[2] / 0.8249);
            next[0] = 116.0 * fy - 16.0;
            next[1] = 500.0 * (fx - fy);
            next[2] = 200.0 * (fy - fz);
        }
        values = next;
    }
    for (var c = 0u; c < params.extent_channels.w; c++) {
        output[params.output_offsets[c / 4u][c % 4u] + id.y * params.output_strides[c / 4u][c % 4u] + id.x] = values[c];
    }
}
