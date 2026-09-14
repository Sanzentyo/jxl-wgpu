const FORWARD_CURVES: u32 = 0u;
const INVERSE_CURVES: u32 = 1u;
const AFFINE: u32 = 2u;
const CLUT: u32 = 3u;
const SEGMENTED_CURVES: u32 = 4u;
const LAB_TO_XYZ: u32 = 5u;
const XYZ_TO_LAB: u32 = 6u;
const CLAMPED_AFFINE: u32 = 7u;
const MULTILINEAR_CLUT: u32 = 8u;
const BLACK_POINT_CONNECTION: u32 = 9u;
const RGB_TRANSFER: u32 = 10u;
const TONE_MAPPING: u32 = 11u;

override wg_x: u32 = 16u;
override wg_y: u32 = 16u;

struct Params {
    extent_channels: vec4<u32>,
    input_offsets: array<vec4<u32>, 4>,
    input_strides: array<vec4<u32>, 4>,
    output_offsets: array<vec4<u32>, 4>,
    output_strides: array<vec4<u32>, 4>,
    connection_scale: vec4<f32>,
    connection_offset: vec3<f32>,
    status: u32,
    sample_encoding: vec4<u32>,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> program: array<u32>;
@group(0) @binding(3) var<storage, read_write> params: Params;

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

// MPE intermediates can exceed F32 even when the final result is representable.
// Keep F32 significand precision with a separate exponent until the element output.
// Bit normalization also preserves computed subnormals on GPUs that flush arithmetic.
struct ScaledFloat { significand: f32, exponent: i32 }

fn scaled(value: f32) -> ScaledFloat {
    let bits = bitcast<u32>(value);
    let magnitude = bits & 0x7fffffffu;
    if magnitude == 0u { return ScaledFloat(value, 0); }
    let encoded_exponent = magnitude >> 23u;
    var fraction = magnitude & 0x7fffffu;
    var exponent = i32(encoded_exponent) - 127;
    if encoded_exponent == 0u {
        let shift = countLeadingZeros(fraction) - 8u;
        fraction = (fraction << shift) & 0x7fffffu;
        exponent = -126 - i32(shift);
    }
    return ScaledFloat(bitcast<f32>((bits & 0x80000000u) | 0x3f800000u | fraction), exponent);
}

fn scaled_add(a: ScaledFloat, b: ScaledFloat) -> ScaledFloat {
    if a.significand == 0.0 { return b; }
    if b.significand == 0.0 { return a; }
    var larger = a;
    var smaller = b;
    if a.exponent < b.exponent { larger = b; smaller = a; }
    let distance = larger.exponent - smaller.exponent;
    if distance > 25 { return larger; }
    let factor = bitcast<f32>(u32(127 - distance) << 23u);
    var result = scaled(larger.significand + smaller.significand * factor);
    result.exponent += larger.exponent;
    return result;
}

fn scaled_multiply(a: ScaledFloat, b: ScaledFloat) -> ScaledFloat {
    var result = scaled(a.significand * b.significand);
    result.exponent += a.exponent + b.exponent;
    return result;
}

fn scaled_divide(a: ScaledFloat, b: ScaledFloat) -> ScaledFloat {
    var result = scaled(a.significand / b.significand);
    result.exponent += a.exponent - b.exponent;
    return result;
}

// Keep the remainder of an affine base before a nonlinear operation can amplify it.
// WGSL fma may be unfused; integer significands make the product and two-term sums
// independent of floating-point contraction or expression reassociation.
struct ScaledPair { high: ScaledFloat, low: ScaledFloat }

fn pair_digits(upper: u32, lower: u32, exponent: i32, sign: f32) -> ScaledPair {
    if upper == 0u {
        var high = scaled(sign * f32(lower));
        high.exponent += exponent;
        return ScaledPair(high, scaled(0.0));
    }
    let shift = 32u - countLeadingZeros(upper);
    let leading = (upper << (24u - shift)) | (lower >> shift);
    let remainder = lower & ((1u << shift) - 1u);
    var high = scaled(sign * f32(leading));
    var low = scaled(sign * f32(remainder));
    high.exponent += exponent + i32(shift);
    low.exponent += exponent;
    return ScaledPair(high, low);
}

fn exact_product(a: ScaledFloat, b: ScaledFloat) -> ScaledPair {
    if a.significand == 0.0 || b.significand == 0.0 {
        return ScaledPair(scaled(0.0), scaled(0.0));
    }
    let sa = (bitcast<u32>(a.significand) & 0x7fffffu) | 0x800000u;
    let sb = (bitcast<u32>(b.significand) & 0x7fffffu) | 0x800000u;
    let product = multiply_wide(sa, sb);
    let sign = select(1.0, -1.0, (a.significand < 0.0) != (b.significand < 0.0));
    return pair_digits((product.y << 8u) | (product.x >> 24u), product.x & 0xffffffu, a.exponent + b.exponent - 46, sign);
}

fn exact_sum(a: ScaledFloat, b: ScaledFloat) -> ScaledPair {
    if a.significand == 0.0 { return ScaledPair(b, scaled(0.0)); }
    if b.significand == 0.0 { return ScaledPair(a, scaled(0.0)); }
    var larger = a;
    var smaller = b;
    if a.exponent < b.exponent || (a.exponent == b.exponent && abs(a.significand) < abs(b.significand)) {
        larger = b; smaller = a;
    }
    let distance = u32(larger.exponent - smaller.exponent);
    if distance > 24u { return ScaledPair(larger, smaller); }
    let large_significand = (bitcast<u32>(larger.significand) & 0x7fffffu) | 0x800000u;
    let small_significand = (bitcast<u32>(smaller.significand) & 0x7fffffu) | 0x800000u;
    var upper = large_significand >> (24u - distance);
    var lower = (large_significand << distance) & 0xffffffu;
    if (larger.significand < 0.0) == (smaller.significand < 0.0) {
        lower += small_significand;
        upper += lower >> 24u;
    } else {
        upper -= u32(lower < small_significand);
        lower -= small_significand;
    }
    return pair_digits(upper, lower & 0xffffffu, smaller.exponent - 23, select(1.0, -1.0, larger.significand < 0.0));
}

fn pair_add(value: ScaledPair, term: ScaledFloat) -> ScaledPair {
    let first = exact_sum(value.high, term);
    let tail = exact_sum(first.low, value.low);
    let combined = exact_sum(first.high, tail.high);
    return exact_sum(combined.high, scaled_add(combined.low, tail.low));
}

fn pair_value(value: ScaledPair) -> ScaledFloat {
    return scaled_add(value.high, value.low);
}

fn affine_pair(a: f32, x: f32, b: f32) -> ScaledPair {
    return pair_add(exact_product(scaled(a), scaled(x)), scaled(b));
}

fn scaled_value(value: ScaledFloat) -> f32 {
    let bits = bitcast<u32>(value.significand);
    let sign = bits & 0x80000000u;
    if value.significand == 0.0 { return bitcast<f32>(sign); }
    if value.exponent > 127 { return bitcast<f32>(sign | 0x7f800000u); }
    if value.exponent >= -126 {
        return bitcast<f32>(sign | (u32(value.exponent + 127) << 23u) | (bits & 0x7fffffu));
    }
    if value.exponent < -150 { return bitcast<f32>(sign); }
    let shift = u32(-126 - value.exponent);
    let significand = (bits & 0x7fffffu) | 0x800000u;
    var rounded = significand >> shift;
    let remainder = significand & ((1u << shift) - 1u);
    let half = 1u << (shift - 1u);
    if remainder > half || (remainder == half && (rounded & 1u) != 0u) { rounded++; }
    return bitcast<f32>(sign | rounded);
}

fn log2_near_ratio(z: f32) -> f32 {
    // log((1+z)/(1-z)) = 2*atanh(z). For |z| <= 1/3 the omitted
    // base-2 terms are below 1.5e-11; the small numerator never cancels a rounded log.
    let square = z * z;
    var series = 1.0 / 19.0;
    for (var denominator = 17; denominator >= 1; denominator -= 2) {
        series = 1.0 / f32(denominator) + square * series;
    }
    return (2.0 / log(2.0)) * z * series;
}

fn scaled_log2(value: ScaledFloat) -> f32 {
    if value.exponent == -1 || value.exponent == 0 {
        let x = abs(value.significand) * select(1.0, 0.5, value.exponent == -1);
        return log2_near_ratio((x - 1.0) / (x + 1.0));
    }
    return f32(value.exponent) + log2(abs(value.significand));
}

fn scaled_log_ratio(a: ScaledFloat, b: ScaledFloat) -> f32 {
    let numerator = ScaledFloat(abs(a.significand), a.exponent);
    let denominator = ScaledFloat(abs(b.significand), b.exponent);
    let ratio = scaled_divide(numerator, denominator);
    if ratio.exponent == -1 || ratio.exponent == 0 {
        let difference = scaled_add(numerator, ScaledFloat(-denominator.significand, denominator.exponent));
        let sum = scaled_add(numerator, denominator);
        return log2_near_ratio(scaled_value(scaled_divide(difference, sum)));
    }
    return scaled_log2(ratio);
}

fn scaled_exp2(value: ScaledFloat) -> ScaledFloat {
    // No finite F32 multiplier or offset can rescue a power beyond these bounds.
    // Logarithmic curves never materialize their power and do not use this cutoff.
    if value.significand != 0.0 && value.exponent >= 10 {
        if value.significand < 0.0 { return scaled(0.0); }
        return ScaledFloat(1.0, 4096);
    }
    let exponent = scaled_value(value);
    let whole = floor(exponent);
    var result = scaled(exp2(exponent - whole));
    result.exponent += i32(whole);
    return result;
}

fn scaled_power(base: ScaledFloat, gamma: ScaledFloat) -> ScaledFloat {
    if gamma.significand == 0.0 { return scaled(1.0); }
    if gamma.significand == 1.0 && gamma.exponent == 0 { return base; }
    if base.significand == 0.0 { return scaled(0.0); }
    let integral_log = scaled_multiply(gamma, scaled(f32(base.exponent)));
    var result: ScaledFloat;
    if base.exponent == -1 || base.exponent == 0 {
        result = scaled_exp2(scaled_multiply(gamma, scaled(scaled_log2(base))));
    } else if integral_log.significand == 0.0 || integral_log.exponent < 10 {
        // Separate the integral exponent before adding the fractional logarithm.
        // Rounding log2(MAX_F32^2) to 256 would otherwise make its square root infinite.
        let integral = scaled_value(integral_log);
        let whole = floor(integral);
        let fractional = scaled_multiply(gamma, scaled(log2(abs(base.significand))));
        result = scaled_exp2(scaled_add(scaled(integral - whole), fractional));
        result.exponent += i32(whole);
    } else {
        result = scaled_exp2(scaled_multiply(gamma, scaled(scaled_log2(base))));
    }
    if base.significand < 0.0 {
        let g = scaled_value(gamma);
        if g - 2.0 * floor(g * 0.5) != 0.0 { result.significand = -result.significand; }
    }
    return result;
}

fn logarithmic_argument(x: f32, gamma: f32, b: f32, c: f32) -> ScaledFloat {
    let sx = scaled(x);
    if b == 0.0 || (sx.significand == 0.0 && gamma > 0.0) {
        return scaled(scaled_log2(scaled(c)));
    }
    var power_log = scaled(0.0);
    if gamma != 0.0 { power_log = scaled_multiply(scaled(gamma), scaled(scaled_log2(sx))); }
    if c == 0.0 { return scaled_add(scaled(scaled_log2(scaled(b))), power_log); }
    let constant_log = scaled(scaled_log2(scaled(c)));
    let delta = scaled_add(power_log, scaled(scaled_log_ratio(scaled(b), scaled(c))));
    var larger = scaled_add(constant_log, delta);
    if delta.significand < 0.0 { larger = constant_log; }
    let absolute_delta = ScaledFloat(abs(delta.significand), delta.exponent);
    let odd = gamma - 2.0 * floor(gamma * 0.5) != 0.0;
    let negative_term = (b < 0.0) != (sx.significand < 0.0 && odd);
    let same_sign = negative_term == (c < 0.0);
    if absolute_delta.significand != 0.0 && (absolute_delta.exponent > 2 || (absolute_delta.exponent == 2 && absolute_delta.significand > 1.25)) {
        // A small log1p term may become significant after multiplication by a large a.
        // Preserve it in scaled form; forming 1 + ratio would round it away.
        let ratio = scaled_exp2(ScaledFloat(-absolute_delta.significand, absolute_delta.exponent));
        let t = scaled_value(ratio);
        let sign = select(-1.0, 1.0, same_sign);
        let series = 1.0 + t * (-sign * 0.5 + t * (1.0 / 3.0 + t * (-sign * 0.25 + t / 5.0)));
        let correction = scaled_multiply(ratio, scaled(sign * series / log(2.0)));
        return scaled_add(larger, correction);
    }
    let distance = scaled_value(absolute_delta);
    if same_sign {
        return scaled_add(larger, scaled(log2(1.0 + exp2(-distance))));
    }
    // log(1-exp(-u)) must not lose a small, positive difference to rounding at 1.
    // The fourth-degree series has remainder below 1.4e-9 for 0 <= u <= 1/16.
    let u = distance * log(2.0);
    var correction = 0.0;
    if u < 0.0625 {
        let ratio = 1.0 + u * (-0.5 + u * (1.0 / 6.0 + u * (-1.0 / 24.0 + u / 120.0)));
        correction = scaled_log2(absolute_delta) + log2(log(2.0)) + log2(ratio);
    } else {
        correction = log2(1.0 - exp2(-distance));
    }
    return scaled_add(larger, scaled(correction));
}

fn pair_log_near_one(value: ScaledPair) -> ScaledFloat {
    let numerator = pair_value(pair_add(value, scaled(-1.0)));
    let denominator = pair_value(pair_add(value, scaled(1.0)));
    let z = scaled_divide(numerator, denominator);
    let coordinate = scaled_value(z);
    let square = coordinate * coordinate;
    var series = 1.0 / 19.0;
    for (var divisor = 17; divisor >= 1; divisor -= 2) { series = 1.0 / f32(divisor) + square * series; }
    return scaled_multiply(z, scaled((2.0 / log(2.0)) * series));
}

fn small_exp2_increment(exponent: ScaledFloat) -> ScaledFloat {
    // exp(u)-1, |u| < ln(2)/16. Keeping u separate preserves even tiny increments.
    let u = scaled_multiply(exponent, scaled(log(2.0)));
    let x = scaled_value(u);
    let factor = 1.0 + x * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x * (1.0 / 120.0 + x / 720.0))));
    return scaled_multiply(u, scaled(factor));
}

fn exp2_with_offset(exponent: ScaledFloat, sign: f32, offset: f32) -> f32 {
    if exponent.significand == 0.0 || exponent.exponent < -4 {
        let increment = scaled_multiply(small_exp2_increment(exponent), scaled(sign));
        return scaled_value(pair_value(pair_add(exact_sum(scaled(sign), scaled(offset)), increment)));
    }
    return scaled_value(scaled_add(scaled_multiply(scaled(sign), scaled_exp2(exponent)), scaled(offset)));
}

fn affine_power(x: f32, gamma: f32, a: f32, b: f32, offset: f32) -> f32 {
    if gamma == 0.0 { return scaled_value(scaled_add(scaled(1.0), scaled(offset))); }
    let base = affine_pair(a, x, b);
    if gamma == 1.0 { return scaled_value(pair_value(pair_add(base, scaled(offset)))); }
    let center = pair_value(base);
    if center.significand == 0.0 { return offset; }
    let negative = center.significand < 0.0;
    let odd = gamma - 2.0 * floor(gamma * 0.5) != 0.0;
    let sign = select(1.0, -1.0, negative && odd);
    if center.exponent == -1 || center.exponent == 0 {
        let factor = select(1.0, -1.0, negative);
        let magnitude = ScaledPair(scaled_multiply(base.high, scaled(factor)), scaled_multiply(base.low, scaled(factor)));
        let exponent = scaled_multiply(scaled(gamma), pair_log_near_one(magnitude));
        return exp2_with_offset(exponent, sign, offset);
    }
    // Away from unity a finite result cannot combine an enormous power with a cancelling
    // logarithm. Preserve the existing integral/fractional exponent split at MAX_F32.
    let ratio = scaled_divide(base.low, base.high);
    let r = scaled_value(ratio);
    let logarithm = scaled_multiply(ratio, scaled((1.0 + r * (-0.5 + r / 3.0)) / log(2.0)));
    let correction = scaled_multiply(scaled(gamma), logarithm);
    let exponent = scaled_add(scaled_multiply(scaled(gamma), scaled(scaled_log2(base.high))), correction);
    if exponent.significand == 0.0 || exponent.exponent < -4 {
        return exp2_with_offset(exponent, sign, offset);
    }
    let power = scaled_power(base.high, scaled(gamma));
    if correction.significand == 0.0 || correction.exponent < -4 {
        let increment = scaled_multiply(power, small_exp2_increment(correction));
        return scaled_value(pair_value(pair_add(exact_sum(power, scaled(offset)), increment)));
    }
    return scaled_value(scaled_add(scaled_multiply(power, scaled_exp2(correction)), scaled(offset)));
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
        let numerator = scaled_add(scaled(x), scaled(-lower));
        let denominator = scaled_add(scaled(upper), scaled(-lower));
        let t = clamp(scaled_value(scaled_divide(numerator, denominator)), 0.0, 1.0);
        if t == 0.0 { return bitcast<f32>(program[samples]); }
        if t == 1.0 { return bitcast<f32>(program[samples + count - 1u]); }
        let position = sample_position(t, count - 1u);
        let a = scaled(bitcast<f32>(program[samples + position.left]));
        let b = scaled(bitcast<f32>(program[samples + position.left + 1u]));
        return scaled_value(scaled_add(scaled_multiply(a, scaled(1.0 - position.weight)), scaled_multiply(b, scaled(position.weight))));
    }
    let p0 = bitcast<f32>(program[record + 4u]);
    let p1 = bitcast<f32>(program[record + 5u]);
    let p2 = bitcast<f32>(program[record + 6u]);
    let p3 = bitcast<f32>(program[record + 7u]);
    let p4 = bitcast<f32>(program[record + 8u]);
    if mode == 0u {
        return affine_power(x, p0, p1, p2, p3);
    }
    if mode == 1u {
        if p1 == 0.0 { return p4; }
        let logarithm = logarithmic_argument(x, p0, p2, p3);
        let scale = scaled_multiply(scaled(p1), scaled(1.0 / log2(10.0)));
        return scaled_value(scaled_add(scaled_multiply(scale, logarithm), scaled(p4)));
    }
    if p0 == 0.0 { return p4; }
    let exponent = scaled_add(scaled_multiply(scaled(p2), scaled(x)), scaled(p3));
    return scaled_value(scaled_add(scaled_multiply(scaled(p0), scaled_power(scaled(p1), exponent)), scaled(p4)));
}

fn clut_value(base: u32, dimensions: u32, channel: u32, values: ptr<function, array<f32, 16>>, multilinear: bool) -> f32 {
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
    let tail = select(min(dimensions, 3u), 0u, multilinear);
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

fn process_program(start: u32, input_values: array<f32, 16>) -> array<f32, 16> {
    var values = input_values;
    for (var stage = 0u; stage < program[start]; stage++) {
        let record = start + 4u + stage * 4u;
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
            else if opcode == CLUT || opcode == MULTILINEAR_CLUT { next[c] = clut_value(base, p, c, &values, opcode == MULTILINEAR_CLUT); }
            else if opcode == BLACK_POINT_CONNECTION { next[c] = params.connection_scale[c] * values[c] + params.connection_offset[c]; }
            else if opcode == SEGMENTED_CURVES { next[c] = segmented_curve(program[base + c], values[c]); }
        }
        if opcode == RGB_TRANSFER {
            let transfer = program[base];
            let gamma = bitcast<f32>(program[base + 1u]);
            let intensity = bitcast<f32>(program[base + 3u]);
            let luminance = vec4<f32>(bitcast<f32>(program[base + 4u]), bitcast<f32>(program[base + 5u]),
                bitcast<f32>(program[base + 6u]), bitcast<f32>(program[base + 7u]));
            let rgb = vec3<f32>(values[0], values[1], values[2]);
            var converted: vec3<f32>;
            if program[base + 2u] != 0u { converted = display_to_linear(rgb, transfer, gamma, intensity, luminance); }
            else { converted = display_from_linear(rgb, transfer, gamma, intensity, luminance); }
            next[0] = converted.x;
            next[1] = converted.y;
            next[2] = converted.z;
        } else if opcode == TONE_MAPPING {
            var tone: ToneMappingParams;
            tone.range = vec4<f32>(bitcast<f32>(program[base]), bitcast<f32>(program[base + 1u]),
                bitcast<f32>(program[base + 2u]), bitcast<f32>(program[base + 3u]));
            tone.curve = vec4<f32>(bitcast<f32>(program[base + 4u]), bitcast<f32>(program[base + 5u]),
                bitcast<f32>(program[base + 6u]), bitcast<f32>(program[base + 7u]));
            tone.knee = vec4<f32>(bitcast<f32>(program[base + 8u]), bitcast<f32>(program[base + 9u]),
                bitcast<f32>(program[base + 10u]), bitcast<f32>(program[base + 11u]));
            let converted = tone_map_light(vec3<f32>(values[0], values[1], values[2]),
                vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.9642, 1.0, 0.8249), tone);
            next[0] = converted.x;
            next[1] = converted.y;
            next[2] = converted.z;
        } else if opcode == LAB_TO_XYZ {
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
    return values;
}

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if params.status != 0u || id.x >= params.extent_channels.x || id.y >= params.extent_channels.y { return; }
    var values: array<f32, 16>;
    for (var c = 0u; c < params.extent_channels.z; c++) {
        values[c] = input[params.input_offsets[c / 4u][c % 4u] + id.y * params.input_strides[c / 4u][c % 4u] + id.x];
        if params.sample_encoding.x == 1u { values[c] = 1.0 - values[c]; }
    }
    values = process_program(0u, values);
    for (var c = 0u; c < params.extent_channels.w; c++) {
        if params.sample_encoding.y == 1u { values[c] = 1.0 - values[c]; }
        output[params.output_offsets[c / 4u][c % 4u] + id.y * params.output_strides[c / 4u][c % 4u] + id.x] = values[c];
    }
}

fn finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

// Each dispatch owns these coefficients. The immutable program may be used concurrently.
@compute @workgroup_size(1, 1, 1)
fn prepare_black_point() {
    let base = program[1];
    var input_values: array<f32, 16>;
    for (var c = 0u; c < program[base + 1u]; c++) {
        input_values[c] = bitcast<f32>(program[base + 8u + c]);
    }
    let evaluated = process_program(program[base], input_values);
    var black = vec3<f32>(evaluated[0], evaluated[1], evaluated[2]);
    let white = vec3<f32>(0.9642, 1.0, 0.8249);
    let fy = lab_f(black.y);
    let lightness = 116.0 * fy - 16.0;
    let clipped = select(clamp(lightness, 0.0, 50.0), 0.0, lightness > 95.0);
    if lightness != clipped {
        let delta = (clipped + 16.0) / 116.0 - fy;
        for (var c = 0u; c < 3u; c++) {
            black[c] = white[c] * lab_inverse(lab_f(black[c] / white[c]) + delta);
        }
    }
    for (var c = 0u; c < 3u; c++) {
        let target_black = bitcast<f32>(program[base + 4u + c]);
        let scale = (white[c] - target_black) / (white[c] - black[c]);
        let offset = target_black - scale * black[c];
        if !finite(black[c]) || !finite(scale) || !finite(offset) {
            params.status = 1u;
        }
        params.connection_scale[c] = scale;
        params.connection_offset[c] = offset;
    }
}
