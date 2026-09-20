// Original-byte assembly. The output remains private until all four status words validate.
struct Params {
    count: u32, source_offset: u32, destination_offset: u32, dispatch_width: u32,
    channel_mask: u32, qprecision: u32, components: u32, coefficient_words: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read> coefficients: array<i32>;
@group(0) @binding(3) var<storage, read> metadata: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> status: array<atomic<u32>>;
const ZIGZAG = array<u32, 64>(
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,
    12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63);
fn index(id: vec3<u32>) -> u32 { return id.x + id.y * params.dispatch_width * 64u; }
fn put(at: u32, value: u32) { atomicOr(&output[at / 4u], value << ((at & 3u) * 8u)); }
@compute @workgroup_size(64)
fn copy_bytes(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = index(id);
    if (i >= params.count) { return; }
    let at = params.source_offset + i;
    put(params.destination_offset + i, (source[at / 4u] >> ((at & 3u) * 8u)) & 255u);
    atomicAdd(&status[1], 1u);
}
@compute @workgroup_size(64)
fn patch_quantizers(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = index(id);
    if (i >= 64u) { return; }
    let k = ZIGZAG[i];
    let q = coefficients[params.source_offset + k];
    if (q < 1 || q > 65535 || (params.qprecision == 0u && q > 255)) {
        atomicMax(&status[0], 1u); return;
    }
    for (var c = 0u; c < 3u; c++) {
        if ((params.channel_mask & (1u << c)) != 0u && coefficients[c * 64u + k] != q) {
            atomicMax(&status[0], 2u); return;
        }
    }
    let at = params.destination_offset + i * (params.qprecision + 1u);
    if (params.qprecision == 1u) { put(at, u32(q) >> 8u); }
    put(at + params.qprecision, u32(q) & 255u);
    atomicAdd(&status[2], params.qprecision + 1u);
}
@compute @workgroup_size(64)
fn validate_coefficients(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = index(id);
    if (i >= params.coefficient_words) { return; }
    let absolute = i + 192u;
    for (var c = 0u; c < params.components; c++) {
        let base = metadata[c * 4u];
        let count = metadata[c * 4u + 1u];
        if (absolute >= base && absolute - base < count) {
            let ordinal = (absolute - base) & 63u;
            let at = absolute - ordinal + ZIGZAG[ordinal];
            let value = coefficients[at];
            let mask = metadata[metadata[c * 4u + 2u] + ordinal];
            let bits = select(u32(abs(value)), u32(value) & 65535u, ordinal == 0u);
            if ((bits & ~mask) != 0u) { atomicMax(&status[0], 3u); }
            atomicAdd(&status[3], 1u);
            return;
        }
    }
    atomicMax(&status[0], 4u);
}
