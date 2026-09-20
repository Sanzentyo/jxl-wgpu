struct Params {
    lf: array<vec4<u32>, 3>,
    outputs: array<vec4<u32>, 3>,
    shifts: array<vec4<u32>, 3>,
    group: vec4<u32>,
    artifact: vec4<u32>,
    config: vec4<u32>,
}
@group(0) @binding(0) var<storage, read> lf: array<i32>;
@group(0) @binding(1) var<storage, read> coefficients: array<i32>;
@group(0) @binding(2) var<storage, read> artifact: array<u32>;
@group(0) @binding(3) var<storage, read> raw_metadata: array<i32>;
@group(0) @binding(4) var<storage, read_write> output: array<i32>;
@group(0) @binding(5) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: Params;

fn fail(code: u32) { atomicOr(&status[3], code); }

@compute @workgroup_size(64)
fn restore(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) k: u32) {
    if (arrayLength(&status) < 4u) { return; }
    let task = group_id.y * params.config.w + group_id.x;
    if (task >= params.artifact.z) { return; }
    if (atomicLoad(&status[0]) != 192u || atomicLoad(&status[1]) != 0u) { fail(1u); return; }
    if (params.config.z > 3u || params.artifact.y >= arrayLength(&artifact) || arrayLength(&artifact) - params.artifact.y < 10u) { fail(2u); return; }
    if (artifact[params.artifact.y] != 0u) { fail(4u); return; }
    let count = artifact[params.artifact.y + 4u];
    if (count > params.artifact.z) { fail(8u); return; }
    if (task >= count) { return; }
    let base = params.artifact.x + task * 12u;
    if (base >= arrayLength(&artifact) || arrayLength(&artifact) - base < 12u) { fail(16u); return; }
    // Only original JPEG DCT8 blocks have the required exact integer interpretation.
    if (artifact[base] != 0u || artifact[base + 4u] != 1u || artifact[base + 5u] != 1u || artifact[base + 9u] != 192u) { fail(32u); return; }
    let x = artifact[base + 2u];
    let y = artifact[base + 3u];
    if (x >= params.group.z || y >= params.group.w) { fail(64u); return; }
    let ac = artifact[base + 8u];
    if (ac > arrayLength(&coefficients) || arrayLength(&coefficients) - ac < 192u) { fail(128u); return; }
    // The entropy sink and raw quantization images use transposed DCT8 order.
    let internal_k = (k % 8u) * 8u + k / 8u;
    for (var channel = 0u; channel < 3u; channel += 1u) {
        let dst = params.outputs[channel];
        if (dst.x == 0xffffffffu) { continue; }
        let shift = params.shifts[channel].xy;
        if ((x & ((1u << shift.x) - 1u)) != 0u || (y & ((1u << shift.y) - 1u)) != 0u) { continue; }
        if ((artifact[base + 11u] & (1u << (8u + channel))) == 0u) { fail(256u); continue; }
        let bx = (params.group.x + x) >> shift.x;
        let by = (params.group.y + y) >> shift.y;
        if (bx >= dst.y || by >= dst.z || dst.w + k >= 192u || arrayLength(&output) < 192u) { fail(512u); continue; }
        let destination = dst.x + (by * dst.y + bx) * 64u + k;
        if (destination >= arrayLength(&output)) { fail(1024u); continue; }
        let quant = output[dst.w + k];
        if (quant < 1 || quant > 65535) { fail(2048u); continue; }
        var value = 0;
        if (k == 0u) {
            let src = params.lf[channel];
            let sx = x >> shift.x;
            let sy = y >> shift.y;
            if (sx >= src.x || sy >= src.y || src.z >= arrayLength(&lf) || sy * src.x + sx >= arrayLength(&lf) - src.z) { fail(4096u); continue; }
            let unit = i32(1u << params.config.z);
            let offset = select(0, 1024 / quant, params.config.x != 0u);
            let raw = lf[src.z + sy * src.x + sx];
            value = (clamp(raw, (offset - 2047) * unit, (offset + 2047) * unit) - offset * unit) / unit;
        } else {
            let raw = coefficients[ac + channel * 64u + internal_k];
            var correction = 0;
            if (params.config.y != 0u && channel != 1u) {
                let quant_y = output[64u + k];
                let width = (params.group.z + 7u) / 8u;
                let count = width * ((params.group.w + 7u) / 8u);
                let correlation_index = (y / 8u) * width + x / 8u + select(0u, count, channel == 2u);
                if (quant_y < 1 || quant_y > 65535 || correlation_index >= arrayLength(&raw_metadata)) { fail(8192u); continue; }
                let correlation = raw_metadata[correlation_index];
                let ratio = u32(quant_y) * 2048u / u32(quant);
                let luma = coefficients[ac + 64u + internal_k];
                if (correlation < -128 || correlation > 127 || ratio > 524287u || luma < -2047 || luma > 2047) { fail(16384u); continue; }
                let scale = correlation * 2048 / 84;
                let coefficient_scale = (i32(ratio) * scale + 1024) >> 11u;
                correction = (luma * coefficient_scale + 1024) >> 11u;
            }
            if (raw < -2047 - correction || raw > 2047 - correction) { fail(32768u); continue; }
            value = raw + correction;
        }
        output[destination] = value;
        atomicAdd(&status[2], 1u);
    }
}
