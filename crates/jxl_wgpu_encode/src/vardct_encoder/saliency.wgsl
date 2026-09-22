// Sum absolute RGB differences to the left/up image neighbors. Source bytes and
// integer addition make the heuristic independent of transform/prefix policy.
var<workgroup> contrast_sums: array<u32, 256>;

@compute @workgroup_size(wg_x, 1, 1)
fn group_saliency(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let left = group.x * 256u;
    let top = group.y * 256u;
    let width = min(256u, params.width - left);
    let height = min(256u, params.height - top);
    var sum = 0u;
    for (var pixel = lane; pixel < width * height; pixel += wg_x) {
        let x = left + pixel % width;
        let y = top + pixel / width;
        let address = params.byte_offset + y * params.row_stride + x * 3u;
        for (var channel = 0u; channel < 3u; channel += 1u) {
            let value = i32(load_u8(address + channel));
            if x > 0u { sum += u32(abs(value - i32(load_u8(address + channel - 3u)))); }
            if y > 0u { sum += u32(abs(value - i32(load_u8(address + channel - params.row_stride)))); }
        }
    }
    // A group has at most 131072 directed edges, each contributing at most 765.
    // The complete sum fits u32; every supported tree shape has identical integer results.
    contrast_sums[lane] = sum;
    workgroupBarrier();
    for (var stride = wg_x / 2u; stride > 0u; stride /= 2u) {
        if lane < stride { contrast_sums[lane] += contrast_sums[lane + stride]; }
        workgroupBarrier();
    }
    if lane == 0u {
        let id = group.y * ((params.width + 255u) / 256u) + group.x;
        let edges = (width - u32(left == 0u)) * height + (height - u32(top == 0u)) * width;
        let offset = params.saliency_offset + 4u * id;
        artifact_words[offset + 1u] = id;
        artifact_words[offset + 2u] = edges;
        artifact_words[offset + 3u] = contrast_sums[0];
        artifact_words[offset] = 0x53414c59u;
    }
}
