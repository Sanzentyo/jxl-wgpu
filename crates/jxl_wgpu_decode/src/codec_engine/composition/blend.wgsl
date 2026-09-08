struct Params {
    canvas: vec4<u32>, // width, height, plane stride in words, channel count
    intersection: vec4<u32>,
    source: vec4<u32>, // crop source x/y, foreground width, foreground plane stride
    dispatch: vec4<u32>,
    references: array<vec4<u32>, 4>, // width, height, plane stride, present
};
struct Channel {
    operation: vec4<u32>, // mode, background slot, alpha plane, clamp/association flags
    alpha: vec4<u32>, // alpha background slot, reserved
};
@group(0) @binding(0) var<storage, read> foreground: array<f32>;
@group(0) @binding(1) var<storage, read> reference0: array<f32>;
@group(0) @binding(2) var<storage, read> reference1: array<f32>;
@group(0) @binding(3) var<storage, read> reference2: array<f32>;
@group(0) @binding(4) var<storage, read> reference3: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<storage, read> channels: array<Channel>;
@group(0) @binding(7) var<uniform> params: Params;

fn background(slot: u32, channel: u32, x: u32, y: u32) -> f32 {
    let geometry = params.references[slot];
    if geometry.w == 0u { return 0.0; }
    let index = channel * geometry.z + y * geometry.x + x;
    switch slot {
        case 0u: { return reference0[index]; }
        case 1u: { return reference1[index]; }
        case 2u: { return reference2[index]; }
        default: { return reference3[index]; }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let sample = id.y * params.dispatch.x + id.x;
    let pixels = params.canvas.x * params.canvas.y;
    if sample >= pixels * params.canvas.w { return; }
    let channel = sample / pixels;
    let pixel = sample % pixels;
    let x = pixel % params.canvas.x;
    let y = pixel / params.canvas.x;
    let operation = channels[channel].operation;
    let base = background(operation.y, channel, x, y);
    var result = base;
    if x >= params.intersection.x && y >= params.intersection.y &&
       x - params.intersection.x < params.intersection.z &&
       y - params.intersection.y < params.intersection.w {
        let sx = x - params.intersection.x + params.source.x;
        let sy = y - params.intersection.y + params.source.y;
        let position = sy * params.source.z + sx;
        let top = foreground[channel * params.source.w + position];
        var alpha = 1.0;
        var base_alpha = 0.0;
        if operation.x == 2u || operation.x == 3u || operation.x == 5u {
            alpha = foreground[operation.z * params.source.w + position];
            if (operation.w & 1u) != 0u { alpha = clamp(alpha, 0.0, 1.0); }
            base_alpha = background(channels[channel].alpha.x, operation.z, x, y);
        }
        switch operation.x {
            case 0u: { result = top; }
            case 1u: { result = base + top; }
            case 2u: {
                if (operation.w & 2u) != 0u { result = top + base * (1.0 - alpha); }
                else {
                    let merged_alpha = 1.0 - (1.0 - alpha) * (1.0 - base_alpha);
                    result = 0.0;
                    if merged_alpha > 0.0 { result = (top * alpha + base * base_alpha * (1.0 - alpha)) / merged_alpha; }
                }
            }
            case 3u: { result = base + top * alpha; }
            case 4u: { result = base * select(top, clamp(top, 0.0, 1.0), (operation.w & 1u) != 0u); }
            case 5u: { result = 1.0 - (1.0 - alpha) * (1.0 - base_alpha); }
            default: {} // AlphaWeightedAdd to its own alpha preserves the background sample.
        }
    }
    output[channel * params.canvas.z + pixel] = result;
}
