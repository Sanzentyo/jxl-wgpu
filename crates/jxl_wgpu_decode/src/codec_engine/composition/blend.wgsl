struct Params {
    canvas: vec4<u32>, // width, height, source row stride in pixels, dispatch width
    intersection: vec4<u32>, // destination x, y, width, height
    source: vec4<u32>, // source x, y, color-reference stride, alpha-reference stride
    blend: vec4<u32>, // color mode, alpha mode, color clamp, alpha clamp
    flags: vec4<u32>, // has alpha, has color reference, has alpha reference, reserved
};
@group(0) @binding(0) var<storage, read> foreground: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> background_color: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> background_alpha: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

fn alpha_over(bottom: f32, top: f32) -> f32 {
    return 1.0 - (1.0 - top) * (1.0 - bottom);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = id.y * params.canvas.w + id.x;
    if pixel >= params.canvas.x * params.canvas.y { return; }
    let x = pixel % params.canvas.x;
    let y = pixel / params.canvas.x;
    var base = vec4<f32>(0.0);
    if params.flags.y != 0u { base = background_color[y * params.source.z + x]; }
    base.a = 0.0;
    if params.flags.z != 0u { base.a = background_alpha[y * params.source.w + x].a; }
    if params.flags.x == 0u { base.a = 1.0; }
    var result = base;
    if x >= params.intersection.x && y >= params.intersection.y &&
       x - params.intersection.x < params.intersection.z &&
       y - params.intersection.y < params.intersection.w {
        let sx = x - params.intersection.x + params.source.x;
        let sy = y - params.intersection.y + params.source.y;
        let top = foreground[sy * params.canvas.z + sx];
        let alpha = select(top.a, clamp(top.a, 0.0, 1.0), params.blend.z != 0u);
        switch params.blend.x {
            case 0u: { result = vec4<f32>(top.rgb, result.a); }
            case 1u: { result = vec4<f32>(base.rgb + top.rgb, result.a); }
            case 2u: {
                result = vec4<f32>(top.rgb, result.a);
                if params.flags.x != 0u {
                    let merged_alpha = alpha_over(base.a, alpha);
                    result = vec4<f32>(0.0, 0.0, 0.0, result.a);
                    if merged_alpha > 0.0 {
                        result = vec4<f32>((top.rgb * alpha + base.rgb * base.a * (1.0 - alpha)) / merged_alpha, result.a);
                    }
                }
            }
            case 3u: {
                result = vec4<f32>(base.rgb + top.rgb * select(1.0, alpha, params.flags.x != 0u), result.a);
            }
            case 4u: {
                result = vec4<f32>(base.rgb * select(top.rgb, clamp(top.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), params.blend.z != 0u), result.a);
            }
            default: {}
        }
        if params.flags.x != 0u {
            let ec_alpha = select(top.a, clamp(top.a, 0.0, 1.0), params.blend.w != 0u);
            switch params.blend.y {
                case 0u: { result.a = top.a; }
                case 1u: { result.a = base.a + top.a; }
                case 2u: { result.a = alpha_over(base.a, ec_alpha); }
                case 3u: { result.a = base.a; }
                case 4u: { result.a = base.a * ec_alpha; }
                default: {}
            }
            // Color source-over also writes its selected alpha channel, as defined by the
            // combined color/alpha blend operation, overriding that channel's individual mode.
            if params.blend.x == 2u { result.a = alpha_over(base.a, alpha); }
        }
    }
    output[pixel] = result;
}
