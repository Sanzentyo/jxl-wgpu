struct Splat {
    geometry: vec4<f32>,
    color: vec4<f32>,
    bounds: vec4<u32>,
};
struct Params {
    image: vec4<u32>, // width, height, tile columns, tile count
    stride: vec4<u32>,
    batch: vec4<u32>, // first reference within tile, exclusive last, tile side, reserved
};
@group(0) @binding(0) var<storage, read_write> plane_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> plane_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> plane_b: array<f32>;
@group(0) @binding(3) var<storage, read> splats: array<Splat>;
@group(0) @binding(4) var<storage, read> references: array<u32>;
@group(0) @binding(5) var<storage, read> tiles: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;

fn erf_approx(value: f32) -> f32 {
    let a = abs(value);
    let d1 = fma(a, 0.0777394369, 0.000205260015);
    let d2 = fma(d1, a, 0.232120216);
    let d3 = fma(d2, a, 0.277820801);
    let d4 = fma(d3, a, 1.0);
    let inverse = 1.0 / (d4 * d4);
    return sign(value) * (1.0 - inverse * inverse);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.image.x || id.y >= params.image.y { return; }
    let tile = (id.y / params.batch.z) * params.image.z + id.x / params.batch.z;
    let begin = tiles[params.image.w + tile];
    let count = tiles[tile];
    if params.batch.x >= count { return; }
    let indices = id.y * params.stride.xyz + vec3<u32>(id.x);
    var pixel = vec3<f32>(plane_x[indices.x], plane_y[indices.y], plane_b[indices.z]);
    for (var i = params.batch.x; i < min(count, params.batch.y); i += 1u) {
        let splat = splats[references[begin + i]];
        if id.x < splat.bounds.x || id.y < splat.bounds.y
            || id.x >= splat.bounds.z || id.y >= splat.bounds.w { continue; }
        let delta = vec2<f32>(id.xy) - splat.geometry.xy;
        let distance = sqrt(fma(delta.x, delta.x, delta.y * delta.y));
        let factor = erf_approx(fma(distance, 0.5, 0.353553391) * splat.geometry.z)
            - erf_approx(fma(distance, 0.5, -0.353553391) * splat.geometry.z);
        let intensity = splat.geometry.w * (factor * factor);
        pixel = fma(splat.color.xyz, vec3<f32>(intensity), pixel);
    }
    plane_x[indices.x] = pixel.x;
    plane_y[indices.y] = pixel.y;
    plane_b[indices.z] = pixel.z;
}
