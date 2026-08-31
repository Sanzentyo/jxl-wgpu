override wg_x: u32 = 16u;
override wg_y: u32 = 16u;

struct Params {
    width: u32,
    height: u32,
    base_stride: u32,
    source_stride: u32,
    output_stride: u32,
    base_alpha_stride: u32,
    source_alpha_stride: u32,
    mode: u32,
    component: u32,
    clamp: u32,
    alpha_associated: u32,
    has_alpha: u32,
};

@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read> source: array<f32>;
@group(0) @binding(2) var<storage, read> base_alpha: array<f32>;
@group(0) @binding(3) var<storage, read> source_alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn maybe_clamp(value: f32) -> f32 {
    if (params.clamp != 0u) {
        return clamp(value, 0.0, 1.0);
    }
    return value;
}

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let base_index = gid.y * params.base_stride + gid.x;
    let source_index = gid.y * params.source_stride + gid.x;
    let output_index = gid.y * params.output_stride + gid.x;
    let base_value = base[base_index];
    let source_value = source[source_index];

    if (params.mode == 0u) {
        output[output_index] = base_value;
        return;
    }
    if (params.mode == 1u) {
        output[output_index] = source_value;
        return;
    }
    if (params.mode == 2u) {
        output[output_index] = base_value + source_value;
        return;
    }
    if (params.mode == 3u) {
        output[output_index] = base_value * maybe_clamp(source_value);
        return;
    }

    if (params.mode == 6u || params.mode == 7u) {
        if (params.component == 1u) {
            output[output_index] = select(base_value, source_value, params.mode == 7u);
            return;
        }
        if (params.has_alpha == 0u) {
            output[output_index] = base_value + source_value;
            return;
        }
        let base_a = base_alpha[gid.y * params.base_alpha_stride + gid.x];
        let source_a = source_alpha[gid.y * params.source_alpha_stride + gid.x];
        if (params.mode == 6u) {
            output[output_index] = base_value + source_value * maybe_clamp(source_a);
        } else {
            output[output_index] = source_value + base_value * maybe_clamp(base_a);
        }
        return;
    }

    let source_above = params.mode == 4u;
    if (params.has_alpha == 0u) {
        output[output_index] = select(base_value, source_value, source_above);
        return;
    }

    let base_a = base_alpha[gid.y * params.base_alpha_stride + gid.x];
    let source_a = source_alpha[gid.y * params.source_alpha_stride + gid.x];
    let top_a = maybe_clamp(select(base_a, source_a, source_above));
    let bottom_a = select(source_a, base_a, source_above);
    let one_minus_top_a = 1.0 - top_a;
    let new_a = 1.0 - one_minus_top_a * (1.0 - bottom_a);
    if (params.component == 1u) {
        output[output_index] = new_a;
        return;
    }

    let top_value = select(base_value, source_value, source_above);
    let bottom_value = select(source_value, base_value, source_above);
    if (params.alpha_associated != 0u) {
        output[output_index] = top_value + bottom_value * one_minus_top_a;
    } else {
        let reciprocal_a = select(0.0, 1.0 / new_a, new_a > 0.0);
        output[output_index] =
            (top_value * top_a + bottom_value * bottom_a * one_minus_top_a) * reciprocal_a;
    }
}
