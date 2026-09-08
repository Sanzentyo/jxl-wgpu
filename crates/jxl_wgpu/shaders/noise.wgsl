// JPEG XL noise synthesis. All 64-bit integer arithmetic uses portable u32 pairs (low, high).
struct NoiseParams {
    geometry: vec4<u32>, // width, height, group dimension, plane pixels
    seed: vec4<u32>, // visible frame, nonvisible frame, reserved
    strides: vec4<u32>,
    correlation: vec4<f32>,
    lut: array<vec4<f32>, 2>,
}
@group(0) @binding(0) var<uniform> p: NoiseParams;
@group(0) @binding(1) var<storage, read_write> random: array<f32>;
@group(0) @binding(2) var<storage, read_write> image_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> image_y: array<f32>;
@group(0) @binding(4) var<storage, read_write> image_b: array<f32>;

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + u32(lo < a.x));
}
fn shr64(a: vec2<u32>, n: u32) -> vec2<u32> {
    return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}
fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let a0 = a.x & 65535u;
    let a1 = a.x >> 16u;
    let b0 = b.x & 65535u;
    let b1 = b.x >> 16u;
    let p0 = a0 * b0;
    let p1 = a0 * b1;
    let p2 = a1 * b0;
    let lo1 = p0 + (p1 << 16u);
    let lo2 = lo1 + (p2 << 16u);
    let hi = a.y * b.x + a.x * b.y + a1 * b1 + (p1 >> 16u) + (p2 >> 16u)
        + u32(lo1 < p0) + u32(lo2 < lo1);
    return vec2<u32>(lo2, hi);
}
fn splitmix64(value: vec2<u32>) -> vec2<u32> {
    var z = mul64(value ^ shr64(value, 30u), vec2<u32>(0x1ce4e5b9u, 0xbf58476du));
    z = mul64(z ^ shr64(z, 27u), vec2<u32>(0x133111ebu, 0x94d049bbu));
    return z ^ shr64(z, 31u);
}
fn random_float(bits: u32) -> f32 {
    return bitcast<f32>((bits >> 9u) | 0x3f800000u);
}

// Eight independent lanes share the specified row/channel sequence. Partial row tails still
// advance every lane. Upsampling subdivisions are represented by full-resolution tiles.
@compute @workgroup_size(8)
fn generate(@builtin(workgroup_id) tile: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let origin = tile.xy * p.geometry.z;
    let size = min(vec2<u32>(p.geometry.z), p.geometry.xy - origin);
    let golden = vec2<u32>(0x7f4a7c15u, 0x9e3779b9u);
    var s0 = splitmix64(add64(vec2<u32>(p.seed.y, p.seed.x), golden));
    var s1 = splitmix64(add64(vec2<u32>(origin.y, origin.x), golden));
    for (var i = 0u; i < lane; i += 1u) {
        s0 = splitmix64(s0);
        s1 = splitmix64(s1);
    }
    for (var channel = 0u; channel < 3u; channel += 1u) {
        for (var y = 0u; y < size.y; y += 1u) {
            for (var batch_x = 0u; batch_x < size.x; batch_x += 16u) {
                let bits = add64(s0, s1);
                let a = s0 ^ vec2<u32>(s0.x << 23u, (s0.y << 23u) | (s0.x >> 9u));
                s0 = s1;
                s1 = a ^ s1 ^ shr64(a, 18u) ^ shr64(s1, 5u);
                let x = batch_x + 2u * lane;
                let offset = channel * p.geometry.w + (origin.y + y) * p.geometry.x + origin.x + x;
                if x < size.x { random[offset] = random_float(bits.x); }
                if x + 1u < size.x { random[offset + 1u] = random_float(bits.y); }
            }
        }
    }
}

fn mirror(coordinate: i32, extent: u32) -> u32 {
    let period = i32(extent) * 2;
    var wrapped = coordinate % period;
    if wrapped < 0 { wrapped += period; }
    return u32(select(wrapped, period - 1 - wrapped, wrapped >= i32(extent)));
}
fn noise_at(x: i32, y: i32) -> vec3<f32> {
    let offset = mirror(y, p.geometry.y) * p.geometry.x + mirror(x, p.geometry.x);
    return vec3<f32>(random[offset], random[offset + p.geometry.w], random[offset + 2u * p.geometry.w]);
}
fn strength(value: f32) -> f32 {
    let scaled = clamp(value * 6.0, 0.0, 7.0);
    let lo = min(u32(floor(scaled)), 6u);
    let a = p.lut[lo / 4u][lo % 4u];
    let b = p.lut[(lo + 1u) / 4u][(lo + 1u) % 4u];
    return clamp(a + (b - a) * (scaled - f32(lo)), 0.0, 1.0);
}

// Convolution reads only the immutable random planes; each invocation updates one XYB pixel.
@compute @workgroup_size(16, 16)
fn apply(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= p.geometry.xy) { return; }
    let x = i32(id.x);
    let y = i32(id.y);
    var others = vec3<f32>(0.0);
    for (var dx = -2; dx <= 2; dx += 1) {
        others += noise_at(x + dx, y - 2);
        others += noise_at(x + dx, y - 1);
        others += noise_at(x + dx, y + 1);
        others += noise_at(x + dx, y + 2);
    }
    others += noise_at(x - 2, y);
    others += noise_at(x - 1, y);
    others += noise_at(x + 1, y);
    others += noise_at(x + 2, y);
    let rnd = (others * 0.16 + noise_at(x, y) * -3.84) * 0.22;
    let offset = id.y * p.strides.xyz + vec3<u32>(id.x);
    let vx = image_x[offset.x];
    let vy = image_y[offset.y];
    let red = strength((vy + vx) * 0.5) * (rnd.x * 0.0078125 + rnd.z * 0.9921875);
    let green = strength((vy - vx) * 0.5) * (rnd.y * 0.0078125 + rnd.z * 0.9921875);
    let sum = red + green;
    image_x[offset.x] = vx + (p.correlation.x * sum + red - green);
    image_y[offset.y] = vy + sum;
    image_b[offset.z] += p.correlation.y * sum;
}
