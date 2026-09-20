struct Params {
    geometry: vec4<u32>,
    offsets: vec4<u32>,
    strides: vec4<u32>,
}
@group(0) @binding(0) var<storage, read> arena: array<i32>;
@group(0) @binding(1) var<storage, read> source_status: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
// words 0/1 are captured sample count / sticky capture error; later words belong to restoration.
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn capture(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= 192u) { return; }
    if (arrayLength(&status) < 4u) { return; }
    if (params.geometry.x != 8u || params.geometry.y != 8u || params.geometry.w != 192u ||
        arrayLength(&source_status) < 4u || arrayLength(&output) < 192u) {
        atomicOr(&status[1], 1u);
        return;
    }
    // The shared Modular executor's success code is one. A zero-initialized status is not success.
    if (source_status[0] != 1u || source_status[1] != params.geometry.z) {
        atomicOr(&status[1], 2u);
        return;
    }
    let channel = index / 64u;
    let y = (index % 64u) / 8u;
    let x = index % 8u;
    let stride = params.strides[channel];
    let offset = params.offsets[channel];
    let length = arrayLength(&arena);
    if (stride < 8u || offset >= length || y > (length - 1u - offset) / stride) {
        atomicOr(&status[1], 4u);
        return;
    }
    let row = offset + y * stride;
    if (x >= length - row) {
        atomicOr(&status[1], 4u);
        return;
    }
    let value = arena[row + x];
    if (value <= 0 || value > 65535) {
        atomicOr(&status[1], 8u);
        return;
    }
    output[channel * 64u + x * 8u + y] = u32(value);
    atomicAdd(&status[0], 1u);
}
