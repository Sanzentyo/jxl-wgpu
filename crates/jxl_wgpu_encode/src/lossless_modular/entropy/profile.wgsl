@group(0) @binding(0) var<storage, read_write> words: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> metadata: array<u32>;

@compute @workgroup_size(1)
fn profile(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= metadata[7u] || id.y >= PROFILE_COUNT { return; }
    let descriptor = metadata[4u] + id.x * 4u;
    let events = metadata[descriptor];
    let length = atomicLoad(&words[metadata[descriptor + 1u]]);
    let stats = metadata[5u];
    let bins = stats + 2u + id.y * 5u * PROFILE_ALPHABET;
    let context = metadata[descriptor + 3u];
    let config = metadata[metadata[6u] + id.y];
    if length > metadata[descriptor + 2u] || context == 0u || context > 4u {
        atomicAdd(&words[stats + 1u], 1u); return;
    }
    for (var event = 0u; event < length; event += 1u) {
        let base = events + event * 4u;
        let kind = atomicLoad(&words[base]);
        if kind == 1u && metadata[2u] == 0u {
            atomicAdd(&words[bins + context * PROFILE_ALPHABET], 1u);
            atomicAdd(&words[bins + 1u], 1u);
        } else if kind == 0u || (kind == 3u && metadata[2u] == 1u) {
            let token = atomicLoad(&words[base + 1u]);
            let count = atomicLoad(&words[base + 2u]);
            let extra = atomicLoad(&words[base + 3u]);
            if !canonical_valid(token, count, extra) {
                atomicAdd(&words[stats + 1u], 1u); return;
            }
            let encoded = hybrid_uint(canonical_value(token, count, extra), config);
            if encoded.token >= PROFILE_ALPHABET { atomicAdd(&words[stats + 1u], 1u); return; }
            atomicAdd(&words[bins + select(context, 0u, kind == 3u) * PROFILE_ALPHABET + encoded.token], 1u);
        } else if kind != 2u || metadata[2u] != 1u {
            atomicAdd(&words[stats + 1u], 1u); return;
        }
    }
    atomicAdd(&words[stats], 1u);
}
