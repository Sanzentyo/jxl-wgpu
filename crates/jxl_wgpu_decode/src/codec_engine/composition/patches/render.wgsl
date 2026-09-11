struct Params {
    canvas: vec4<u32>, // width, height, plane words, channel count
    jobs: vec4<u32>, // first command, end command, command words, dispatch row width
    flags: vec4<u32>, // has alpha
    references: array<vec4<u32>, 4>,
};
@group(0) @binding(0) var<storage, read> reference0: array<f32>;
@group(0) @binding(1) var<storage, read> reference1: array<f32>;
@group(0) @binding(2) var<storage, read> reference2: array<f32>;
@group(0) @binding(3) var<storage, read> reference3: array<f32>;
@group(0) @binding(4) var<storage, read> commands: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

fn reference(slot: u32, channel: u32, pixel: u32) -> f32 {
    let index = channel * params.references[slot].z + pixel;
    switch slot {
        case 0u: { return reference0[index]; }
        case 1u: { return reference1[index]; }
        case 2u: { return reference2[index]; }
        default: { return reference3[index]; }
    }
}
fn blend(old: f32, incoming: f32, old_alpha: f32, new_alpha: f32, mode: u32, flags: u32, own_alpha: bool) -> f32 {
    if mode == 0u { return old; }
    if mode == 1u { return incoming; }
    if mode == 2u { return old + incoming; }
    if mode == 3u { return old * select(incoming, clamp(incoming, 0.0, 1.0), (flags & 1u) != 0u); }
    let below = mode == 5u || mode == 7u;
    let top = select(incoming, old, below);
    let bottom = select(old, incoming, below);
    var alpha = select(new_alpha, old_alpha, below);
    if (flags & 1u) != 0u { alpha = clamp(alpha, 0.0, 1.0); }
    let base_alpha = select(old_alpha, new_alpha, below);
    if mode >= 6u {
        if own_alpha { return bottom; }
        return bottom + top * alpha;
    }
    let merged = 1.0 - (1.0 - alpha) * (1.0 - base_alpha);
    if own_alpha { return merged; }
    if (flags & 2u) != 0u { return top + bottom * (1.0 - alpha); }
    if merged > 0.0 { return (top * alpha + bottom * base_alpha * (1.0 - alpha)) / merged; }
    return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = id.y * params.jobs.w + id.x;
    if pixel >= params.canvas.x * params.canvas.y { return; }
    let x = pixel % params.canvas.x;
    let y = pixel / params.canvas.x;
    for (var job = params.jobs.x; job < params.jobs.y; job += 1u) {
        let record = job * params.jobs.z;
        let dx = commands[record + 5u];
        let dy = commands[record + 6u];
        if x < dx || y < dy || x - dx >= commands[record + 3u] || y - dy >= commands[record + 4u] { continue; }
        let slot = commands[record];
        let source_pixel = (y - dy + commands[record + 2u]) * params.references[slot].x + x - dx + commands[record + 1u];
        // Compute every channel from the old alpha state, then publish this patch atomically
        // with respect to the following patch. Each invocation owns one complete pixel.
        for (var channel = 0u; channel < params.canvas.w; channel += 1u) {
            let offset = record + 8u + select(0u, channel - 2u, channel >= 3u) * 3u;
            let mode = commands[offset];
            let alpha_channel = 3u + commands[offset + 1u];
            let flags = commands[offset + 2u];
            let index = channel * params.canvas.z + pixel;
            let old = output[index];
            let incoming = reference(slot, channel, source_pixel);
            var old_alpha = 1.0;
            var new_alpha = 1.0;
            if mode >= 4u && (channel >= 3u || params.flags.x != 0u) {
                old_alpha = output[alpha_channel * params.canvas.z + pixel];
                new_alpha = reference(slot, alpha_channel, source_pixel);
            }
            var value = blend(old, incoming, old_alpha, new_alpha, mode, flags, channel == alpha_channel);
            // With no alpha channel, both source-over directions reduce to replacement.
            if channel < 3u && params.flags.x == 0u && (mode == 4u || mode == 5u) { value = incoming; }
            scratch[index] = value;
        }
        let color_mode = commands[record + 8u];
        if params.flags.x != 0u && (color_mode == 4u || color_mode == 5u) {
            let alpha_channel = 3u + commands[record + 9u];
            let index = alpha_channel * params.canvas.z + pixel;
            let old = output[index];
            let incoming = reference(slot, alpha_channel, source_pixel);
            scratch[index] = blend(old, incoming, old, incoming, color_mode, commands[record + 10u], true);
        }
        for (var channel = 0u; channel < params.canvas.w; channel += 1u) {
            let index = channel * params.canvas.z + pixel;
            output[index] = scratch[index];
        }
    }
}
