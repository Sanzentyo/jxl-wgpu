// Geometry, count, tile-prefix and emission phases share one bounded continuation.
// Status words 0..7 contain only errors, phase and allocation/work counts.
struct Params {
    image: vec4<u32>, // width, height, tile columns, tile count
    run: vec4<u32>, // reset, initial phase, record capacity, tile-reference capacity
    correlation: vec4<f32>,
    work: vec4<u32>, // total step limit, spline-header words, tile side, per-dispatch steps
};
struct Splat {
    geometry: vec4<f32>, // center x/y, inverse sigma, sigma * intensity / 4
    color: vec4<f32>,
    bounds: vec4<u32>, // exclusive right/bottom
};
@group(0) @binding(0) var<storage, read_write> program: array<u32>;
@group(0) @binding(1) var<storage, read_write> state: array<u32>;
@group(0) @binding(2) var<storage, read_write> tiles: array<u32>;
@group(0) @binding(3) var<storage, read_write> splats: array<Splat>;
@group(0) @binding(4) var<storage, read_write> references: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

const ARCS: u32 = 0u;
const COUNT: u32 = 1u;
const PREFIX: u32 = 2u;
const READY: u32 = 3u;
const EMIT: u32 = 4u;
const DONE: u32 = 5u;
const ERROR_WORK: u32 = 16u;
const ERROR_RECORDS: u32 = 17u;
const ERROR_REFERENCES: u32 = 18u;
const ERROR_FINITE: u32 = 19u;
const ERROR_REPLAY: u32 = 20u;

fn finite(value: f32) -> bool { return abs(value) <= 3.402823466e38; }
fn header() -> u32 { return 4u + state[8] * params.work.y; }
fn control(base: u32, index: u32) -> vec2<f32> {
    var offset = program[base] + (index - 1u) * 2u;
    if index == 0u { offset = base + 2u; }
    return vec2<f32>(f32(bitcast<i32>(program[offset])), f32(bitcast<i32>(program[offset + 1u])));
}
fn extended(base: u32, index: i32) -> vec2<f32> {
    let count = program[base + 1u];
    if index < 0 {
        let first = control(base, 0u);
        return first + (first - control(base, 1u));
    }
    if u32(index) >= count {
        let last = control(base, count - 1u);
        return last + (last - control(base, count - 2u));
    }
    return control(base, u32(index));
}
fn intermediate(base: u32, index: u32) -> vec2<f32> {
    let count = program[base + 1u];
    if count == 1u || index == (count - 1u) * 16u { return control(base, count - 1u); }
    let interval = i32(index / 16u);
    let p1 = extended(base, interval);
    if index % 16u == 0u { return p1; }
    let p0 = extended(base, interval - 1);
    let p2 = extended(base, interval + 1);
    let p3 = extended(base, interval + 2);
    let d0 = sqrt(length(p1 - p0));
    let d1 = sqrt(length(p2 - p1));
    let d2 = sqrt(length(p3 - p2));
    let t = d0 + (f32(index % 16u) / 16.0) * d1;
    let a0 = p0 + (t / d0) * (p1 - p0);
    let a1 = p1 + ((t - d0) / d1) * (p2 - p1);
    let a2 = p2 + ((t - (d0 + d1)) / d2) * (p3 - p2);
    let b0 = a0 + (t / (d0 + d1)) * (a1 - a0);
    let b1 = a1 + ((t - d0) / (d1 + d2)) * (a2 - a1);
    return b0 + ((t - d0) / d1) * (b1 - b0);
}
fn dequantize(base: u32) {
    let adjustment = f32(bitcast<i32>(program[1]));
    var inverse = 1.0 - adjustment * 0.125;
    if adjustment >= 0.0 { inverse = 1.0 / (1.0 + adjustment * 0.125); }
    for (var i = 0u; i < 32u; i += 1u) {
        let scale = select(1.0, 0.7071067811865476, i == 0u);
        let y = f32(bitcast<i32>(program[base + 36u + i])) * scale * 0.075 * inverse;
        let x = f32(bitcast<i32>(program[base + 4u + i])) * scale * 0.0042 * inverse
            + params.correlation.x * y;
        let b = f32(bitcast<i32>(program[base + 68u + i])) * scale * 0.07 * inverse
            + params.correlation.y * y;
        let sigma = f32(bitcast<i32>(program[base + 100u + i])) * scale * 0.3333 * inverse;
        program[base + 4u + i] = bitcast<u32>(x);
        program[base + 36u + i] = bitcast<u32>(y);
        program[base + 68u + i] = bitcast<u32>(b);
        program[base + 100u + i] = bitcast<u32>(sigma);
    }
}
fn continuous(base: u32, t: f32) -> vec4<f32> {
    var result = vec4<f32>(0.0);
    for (var i = 0u; i < 32u; i += 1u) {
        let coefficients = vec4<f32>(
            bitcast<f32>(program[base + 4u + i]), bitcast<f32>(program[base + 36u + i]),
            bitcast<f32>(program[base + 68u + i]), bitcast<f32>(program[base + 100u + i]));
        let basis = cos((3.141592653589793 / 32.0 * f32(i)) * (t + 0.5));
        result += 1.4142135623730951 * coefficients * basis;
    }
    return result;
}
fn nearest(value: f32) -> f32 {
    // llround semantics, including negative half-integers; WGSL round uses ties-to-even.
    if value < 0.0 { return -floor(-value + 0.5); }
    return floor(value + 0.5);
}
fn accept_sample(point: vec2<f32>, intensity: f32, last: bool) {
    let base = header();
    let index = state[10];
    state[10] += 1u;
    if state[1] == ARCS {
        if last {
            program[base + 132u] = bitcast<u32>(f32(state[10] - 2u) + intensity);
        }
        return;
    }
    let arc = bitcast<f32>(program[base + 132u]);
    if arc <= 0.0 || intensity == 0.0 { return; }
    let values = continuous(base, 31.0 * min(1.0, f32(index) / arc));
    let sigma = values.w;
    if !all(vec4<bool>(finite(values.x), finite(values.y), finite(values.z), finite(sigma))) {
        state[0] = ERROR_FINITE; return;
    }
    if sigma == 0.0 || !finite(1.0 / sigma) { return; }
    let color_intensity = abs(values.xyz * intensity);
    let maximum_color = max(0.01, max(color_intensity.x, max(color_intensity.y, color_intensity.z)));
    // Reference high-precision cutoff: maximum per-sample tail amplitude 1e-5.
    // Factor sigma out of the square root to avoid overflowing sigma squared.
    let distance = abs(sigma) * sqrt(2.0 * (11.512925464970229 + log(maximum_color)));
    let bounds = vec4<u32>(
        u32(clamp(nearest(point.x - distance), 0.0, f32(params.image.x))),
        u32(clamp(nearest(point.y - distance), 0.0, f32(params.image.y))),
        u32(clamp(nearest(point.x + distance) + 1.0, 0.0, f32(params.image.x))),
        u32(clamp(nearest(point.y + distance) + 1.0, 0.0, f32(params.image.y))));
    if bounds.x >= bounds.z || bounds.y >= bounds.w { return; }
    if state[2] >= params.run.z { state[0] = ERROR_RECORDS; return; }
    state[27] = state[2];
    state[2] += 1u;
    if state[1] == EMIT {
        splats[state[27]] = Splat(
            vec4<f32>(point, 1.0 / sigma, 0.25 * sigma * intensity),
            vec4<f32>(values.xyz, 0.0), bounds);
    }
    state[22] = bounds.x / params.work.z;
    state[23] = bounds.y / params.work.z;
    state[24] = state[22];
    state[25] = (bounds.z - 1u) / params.work.z + 1u;
    state[26] = (bounds.w - 1u) / params.work.z + 1u;
    state[15] = 1u;
}
fn tile_reference() {
    if state[3] >= params.run.w { state[0] = ERROR_REFERENCES; return; }
    let tile = state[23] * params.image.z + state[22];
    if state[1] == COUNT {
        tiles[tile] += 1u;
    } else {
        let cursor_word = 2u * params.image.w + 1u + tile;
        let cursor = tiles[cursor_word];
        if cursor >= tiles[tile] { state[0] = ERROR_REPLAY; return; }
        references[tiles[params.image.w + tile] + cursor] = state[27];
        tiles[cursor_word] = cursor + 1u;
        state[4] = max(state[4], cursor + 1u);
    }
    state[3] += 1u;
    state[22] += 1u;
    if state[22] == state[25] {
        state[22] = state[24];
        state[23] += 1u;
        if state[23] == state[26] { state[15] = 0u; }
    }
}
fn walk() {
    if state[8] == program[0] {
        state[8] = 0u;
        state[9] = 0u;
        if state[1] == ARCS { state[1] = COUNT; }
        else if state[1] == COUNT { state[1] = PREFIX; }
        else { state[1] = DONE; }
        return;
    }
    let base = header();
    if state[9] == 0u {
        if state[1] == ARCS { dequantize(base); }
        else if bitcast<f32>(program[base + 132u]) <= 0.0 { state[8] += 1u; return; }
        let first = control(base, 0u);
        state[10] = 0u;
        state[11] = 0u;
        state[12] = bitcast<u32>(first.x);
        state[13] = bitcast<u32>(first.y);
        state[14] = 0u;
        state[9] = 1u;
        accept_sample(first, 1.0, false);
        return;
    }
    let previous = vec2<f32>(bitcast<f32>(state[12]), bitcast<f32>(state[13]));
    let remainder = bitcast<f32>(state[14]);
    let point_count = (program[base + 1u] - 1u) * 16u + 1u;
    if state[11] == point_count {
        accept_sample(previous, remainder, true);
        state[8] += 1u;
        state[9] = 0u;
        return;
    }
    let next = intermediate(base, state[11]);
    let distance = length(next - previous);
    if !finite(distance) { state[0] = ERROR_FINITE; return; }
    if remainder + distance >= 1.0 {
        let current = previous + ((1.0 - remainder) / distance) * (next - previous);
        state[12] = bitcast<u32>(current.x);
        state[13] = bitcast<u32>(current.y);
        state[14] = 0u;
        accept_sample(current, 1.0, false);
    } else {
        state[12] = bitcast<u32>(next.x);
        state[13] = bitcast<u32>(next.y);
        state[14] = bitcast<u32>(remainder + distance);
        state[11] += 1u;
    }
}
fn prefix_tiles() {
    let tile = state[16];
    tiles[params.image.w + tile] = state[17];
    if tile == params.image.w {
        if state[17] != state[3] { state[0] = ERROR_REPLAY; }
        state[1] = READY;
        return;
    }
    state[17] += tiles[tile];
    state[4] = max(state[4], tiles[tile]);
    state[16] += 1u;
}
@compute @workgroup_size(1)
fn main() {
    if params.run.x != 0u {
        for (var i = 0u; i < 64u; i += 1u) { state[i] = 0u; }
        state[1] = params.run.y;
    }
    for (var step = 0u; step < params.work.w && state[0] == 0u; step += 1u) {
        if state[1] == READY || state[1] == DONE { break; }
        if state[5] >= params.work.x { state[0] = ERROR_WORK; break; }
        state[5] += 1u;
        if state[1] == PREFIX { prefix_tiles(); }
        else if state[15] != 0u { tile_reference(); }
        else { walk(); }
    }
}
