struct Task {
    coefficient_offset: u32,
    scratch_or_basis_offset: u32,
    matrix_offset: u32,
    quant_index: u32,
    coefficient_origin_x: u32,
    lf_offset: u32,
    channel_mask: u32,
    coefficient_origin_y: u32,
    destination_x_x: u32,
    destination_y_x: u32,
    destination_x_y: u32,
    destination_y_y: u32,
    destination_x_b: u32,
    destination_y_b: u32,
    _pad1: u32,
    _pad2: u32,
};

struct Params {
    task_base: u32,
    task_count: u32,
    transform_width: u32,
    transform_height: u32,
    transform_area: u32,
    lf_width: u32,
    lf_height: u32,
    quant_offset: u32,
    correlation_offset: u32,
    lf_offset: u32,
    output_width_x: u32,
    output_height_x: u32,
    output_stride_x: u32,
    output_width_y: u32,
    output_height_y: u32,
    output_stride_y: u32,
    output_width_b: u32,
    output_height_b: u32,
    output_stride_b: u32,
    transform_kind: u32,
    correlation_width: u32,
    correlation_height: u32,
    _padding: vec2<u32>,
    quant_biases: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> coefficients: array<i32>;
@group(0) @binding(1) var<storage, read> tasks: array<Task>;
@group(0) @binding(2) var<storage, read> resources: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> output_x: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(7) var<storage, read_write> output_b: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

var<workgroup> block: array<f32, 192>;
var<workgroup> temporary: array<f32, 192>;
var<workgroup> auxiliary: array<f32, 192>;

const PI: f32 = 3.14159265358979323846;
const SQRT_2: f32 = 1.41421356237309504880;
const HORNUSS: u32 = 0u;
const DCT2X2: u32 = 1u;
const DCT4X4: u32 = 2u;
const DCT4X8: u32 = 3u;
const DCT8X4: u32 = 4u;
const AFV0: u32 = 5u;

fn idct_basis(position: u32, frequency: u32, length: u32) -> f32 {
    if (frequency == 0u) {
        return 1.0;
    }
    let phase = f32((2u * position + 1u) * frequency) * PI / (2.0 * f32(length));
    return SQRT_2 * cos(phase);
}

fn adjust_quantized(value: i32, small_bias: f32, large_bias: f32) -> f32 {
    let value_f = f32(value);
    if (value > -2 && value < 2) {
        return value_f * small_bias;
    }
    return value_f - large_bias / value_f;
}

fn sample(channel: u32, index: u32) -> f32 {
    return block[channel * 64u + index];
}

fn afv_basis(basis_offset: u32, coefficient: u32, pixel: u32) -> f32 {
    let scalar_index = coefficient * 16u + pixel;
    return resources[basis_offset + scalar_index / 4u][scalar_index % 4u];
}

fn afv(task: Task, channel: u32, x: u32, y: u32, kind: u32) -> f32 {
    let afv_x = kind & 1u;
    let afv_y = kind / 2u;
    let block00 = sample(channel, 0u);
    let block01 = sample(channel, 1u);
    let block10 = sample(channel, 8u);
    let dc0 = (block00 + block10 + block01) * 4.0;
    let dc1 = block00 + block10 - block01;
    let dc2 = block00 - block10;

    let local_y = y % 4u;
    if (y / 4u == afv_y) {
        let local_x = x % 4u;
        if (x / 4u == afv_x) {
            let basis_x = select(local_x, 3u - local_x, afv_x == 1u);
            let basis_y = select(local_y, 3u - local_y, afv_y == 1u);
            let pixel = basis_y * 4u + basis_x;
            var sum = 0.0;
            for (var frequency_y = 0u; frequency_y < 4u; frequency_y = frequency_y + 1u) {
                for (var frequency_x = 0u; frequency_x < 4u; frequency_x = frequency_x + 1u) {
                    var coefficient = dc0;
                    if (frequency_x != 0u || frequency_y != 0u) {
                        coefficient = sample(
                            channel,
                            frequency_y * 16u + frequency_x * 2u,
                        );
                    }
                    sum = fma(
                        coefficient,
                        afv_basis(
                            task.scratch_or_basis_offset,
                            frequency_y * 4u + frequency_x,
                            pixel,
                        ),
                        sum,
                    );
                }
            }
            return sum;
        }

        var sum = 0.0;
        for (var frequency_y = 0u; frequency_y < 4u; frequency_y = frequency_y + 1u) {
            for (var frequency_x = 0u; frequency_x < 4u; frequency_x = frequency_x + 1u) {
                var coefficient = dc1;
                if (frequency_x != 0u || frequency_y != 0u) {
                    coefficient = sample(
                        channel,
                        frequency_x * 16u + frequency_y * 2u + 1u,
                    );
                }
                sum = fma(
                    coefficient,
                    idct_basis(local_x, frequency_x, 4u)
                        * idct_basis(local_y, frequency_y, 4u),
                    sum,
                );
            }
        }
        return sum;
    }

    var sum = 0.0;
    for (var frequency_y = 0u; frequency_y < 4u; frequency_y = frequency_y + 1u) {
        for (var frequency_x = 0u; frequency_x < 8u; frequency_x = frequency_x + 1u) {
            var coefficient = dc2;
            if (frequency_x != 0u || frequency_y != 0u) {
                coefficient = sample(
                    channel,
                    (1u + frequency_y * 2u) * 8u + frequency_x,
                );
            }
            sum = fma(
                coefficient,
                idct_basis(x, frequency_x, 8u) * idct_basis(local_y, frequency_y, 4u),
                sum,
            );
        }
    }
    return sum;
}

fn hornuss(channel: u32, x: u32, y: u32) -> f32 {
    let quadrant_x = x / 4u;
    let quadrant_y = y / 4u;
    let local_x = x % 4u;
    let local_y = y % 4u;
    let c00 = sample(channel, 0u);
    let c01 = sample(channel, 1u);
    let c10 = sample(channel, 8u);
    let c11 = sample(channel, 9u);
    let sign01 = select(1.0, -1.0, quadrant_y == 1u);
    let sign10 = select(1.0, -1.0, quadrant_x == 1u);
    let dc = c00 + sign01 * c01 + sign10 * c10 + sign01 * sign10 * c11;
    var residual_sum = 0.0;
    for (var iy = 0u; iy < 4u; iy = iy + 1u) {
        for (var ix = 0u; ix < 4u; ix = ix + 1u) {
            if (ix != 0u || iy != 0u) {
                residual_sum = residual_sum
                    + sample(channel, (quadrant_y + iy * 2u) * 8u + quadrant_x + ix * 2u);
            }
        }
    }
    let center = dc - residual_sum * 0.0625;
    if (local_x == 1u && local_y == 1u) {
        return center;
    }
    if (local_x == 0u && local_y == 0u) {
        return sample(channel, (quadrant_y + 2u) * 8u + quadrant_x + 2u) + center;
    }
    return sample(
        channel,
        (quadrant_y + local_y * 2u) * 8u + quadrant_x + local_x * 2u,
    ) + center;
}

fn idct4x4_special(channel: u32, x: u32, y: u32) -> f32 {
    let quadrant_x = x / 4u;
    let quadrant_y = y / 4u;
    let local_x = x % 4u;
    let local_y = y % 4u;
    let c00 = sample(channel, 0u);
    let c01 = sample(channel, 1u);
    let c10 = sample(channel, 8u);
    let c11 = sample(channel, 9u);
    let sign01 = select(1.0, -1.0, quadrant_y == 1u);
    let sign10 = select(1.0, -1.0, quadrant_x == 1u);
    let dc = c00 + sign01 * c01 + sign10 * c10 + sign01 * sign10 * c11;
    var sum = 0.0;
    for (var frequency_x = 0u; frequency_x < 4u; frequency_x = frequency_x + 1u) {
        for (var frequency_y = 0u; frequency_y < 4u; frequency_y = frequency_y + 1u) {
            var coefficient = dc;
            if (frequency_x != 0u || frequency_y != 0u) {
                coefficient = sample(
                    channel,
                    (quadrant_y + frequency_x * 2u) * 8u
                        + quadrant_x
                        + frequency_y * 2u,
                );
            }
            sum = fma(
                coefficient,
                idct_basis(local_x, frequency_x, 4u) * idct_basis(local_y, frequency_y, 4u),
                sum,
            );
        }
    }
    return sum;
}

fn idct4x8_special(channel: u32, x: u32, y: u32) -> f32 {
    let half = y / 4u;
    let local_y = y % 4u;
    let first = sample(channel, 0u);
    let second = sample(channel, 8u);
    let dc = select(first + second, first - second, half == 1u);
    var sum = 0.0;
    for (var frequency_y = 0u; frequency_y < 4u; frequency_y = frequency_y + 1u) {
        for (var frequency_x = 0u; frequency_x < 8u; frequency_x = frequency_x + 1u) {
            var coefficient = dc;
            if (frequency_x != 0u || frequency_y != 0u) {
                coefficient = sample(channel, (half + frequency_y * 2u) * 8u + frequency_x);
            }
            sum = fma(
                coefficient,
                idct_basis(x, frequency_x, 8u) * idct_basis(local_y, frequency_y, 4u),
                sum,
            );
        }
    }
    return sum;
}

fn idct8x4_special(channel: u32, x: u32, y: u32) -> f32 {
    let half = x / 4u;
    let local_x = x % 4u;
    let first = sample(channel, 0u);
    let second = sample(channel, 8u);
    let dc = select(first + second, first - second, half == 1u);
    var sum = 0.0;
    for (var frequency_x = 0u; frequency_x < 4u; frequency_x = frequency_x + 1u) {
        for (var frequency_y = 0u; frequency_y < 8u; frequency_y = frequency_y + 1u) {
            var coefficient = dc;
            if (frequency_x != 0u || frequency_y != 0u) {
                coefficient = sample(channel, (half + frequency_x * 2u) * 8u + frequency_y);
            }
            sum = fma(
                coefficient,
                idct_basis(local_x, frequency_x, 4u) * idct_basis(y, frequency_y, 8u),
                sum,
            );
        }
    }
    return sum;
}

fn dct2_source(stage: u32, index: u32) -> f32 {
    if (stage == 0u) {
        return block[index];
    }
    if (stage == 1u) {
        return temporary[index];
    }
    return auxiliary[index];
}

fn dct2_stage(stage: u32, channel: u32, x: u32, y: u32, size: u32) -> f32 {
    let half = size / 2u;
    let source_x = x / 2u;
    let source_y = y / 2u;
    let bit_x = x & 1u;
    let bit_y = y & 1u;
    let base = channel * 64u;
    let c00 = dct2_source(stage, base + source_y * 8u + source_x);
    let c01 = dct2_source(stage, base + source_y * 8u + half + source_x);
    let c10 = dct2_source(stage, base + (source_y + half) * 8u + source_x);
    let c11 = dct2_source(stage, base + (source_y + half) * 8u + half + source_x);
    let sign01 = select(1.0, -1.0, bit_y == 1u);
    let sign10 = select(1.0, -1.0, bit_x == 1u);
    return c00 + sign01 * c01 + sign10 * c10 + sign01 * sign10 * c11;
}

fn write_output(task: Task, channel: u32, x: u32, y: u32, value: f32) {
    if (((task.channel_mask >> channel) & 1u) == 0u) {
        return;
    }
    if (channel == 0u) {
        let destination_x = task.destination_x_x + x;
        let destination_y = task.destination_y_x + y;
        if (destination_x < params.output_width_x && destination_y < params.output_height_x) {
            output_x[destination_y * params.output_stride_x + destination_x] = value;
        }
    } else if (channel == 1u) {
        let destination_x = task.destination_x_y + x;
        let destination_y = task.destination_y_y + y;
        if (destination_x < params.output_width_y && destination_y < params.output_height_y) {
            output_y[destination_y * params.output_stride_y + destination_x] = value;
        }
    } else {
        let destination_x = task.destination_x_b + x;
        let destination_y = task.destination_y_b + y;
        if (destination_x < params.output_width_b && destination_y < params.output_height_b) {
            output_b[destination_y * params.output_stride_b + destination_x] = value;
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if (workgroup_id.y >= params.task_count) {
        return;
    }
    let task = tasks[params.task_base + workgroup_id.y];
    let matrix = resources[task.matrix_offset + lane].xyz;
    let quant = resources[params.quant_offset + task.quant_index].xyz;
    let x_value = adjust_quantized(
        coefficients[task.coefficient_offset + lane], params.quant_biases.x, params.quant_biases.w
    ) * quant.x * matrix.x;
    let y_value = adjust_quantized(
        coefficients[task.coefficient_offset + 64u + lane],
        params.quant_biases.y,
        params.quant_biases.w,
    ) * quant.y * matrix.y;
    let b_value = adjust_quantized(
        coefficients[task.coefficient_offset + 128u + lane],
        params.quant_biases.z,
        params.quant_biases.w,
    ) * quant.z * matrix.z;
    let frequency_x = lane / 8u;
    let frequency_y = lane % 8u;
    let correlation_cell_x = (task.coefficient_origin_x + frequency_x) / 64u;
    let correlation_cell_y = (task.coefficient_origin_y + frequency_y) / 64u;
    let correlation = resources[
        params.correlation_offset + correlation_cell_y * params.correlation_width
            + correlation_cell_x
    ].xy;
    if (lane == 0u) {
        let lf = resources[params.lf_offset + task.lf_offset].xyz;
        block[0u] = lf.x;
        block[64u] = lf.y;
        block[128u] = lf.z;
    } else {
        block[lane] = fma(correlation.x, y_value, x_value);
        block[64u + lane] = y_value;
        block[128u + lane] = fma(correlation.y, y_value, b_value);
    }
    workgroupBarrier();

    let x = lane % 8u;
    let y = lane / 8u;
    if (params.transform_kind == DCT2X2) {
        for (var channel = 0u; channel < 3u; channel = channel + 1u) {
            // Each hierarchical pass overwrites only its top-left square; all coefficients
            // outside that square remain live input to the next pass.
            temporary[channel * 64u + lane] = block[channel * 64u + lane];
            workgroupBarrier();
            if (x < 2u && y < 2u) {
                temporary[channel * 64u + y * 8u + x] = dct2_stage(0u, channel, x, y, 2u);
            }
            workgroupBarrier();
            auxiliary[channel * 64u + lane] = temporary[channel * 64u + lane];
            workgroupBarrier();
            if (x < 4u && y < 4u) {
                auxiliary[channel * 64u + y * 8u + x] = dct2_stage(1u, channel, x, y, 4u);
            }
            workgroupBarrier();
            let value = dct2_stage(2u, channel, x, y, 8u);
            write_output(task, channel, x, y, value);
            workgroupBarrier();
        }
        return;
    }

    for (var channel = 0u; channel < 3u; channel = channel + 1u) {
        var value = 0.0;
        if (params.transform_kind == HORNUSS) {
            value = hornuss(channel, x, y);
        } else if (params.transform_kind == DCT4X4) {
            value = idct4x4_special(channel, x, y);
        } else if (params.transform_kind == DCT4X8) {
            value = idct4x8_special(channel, x, y);
        } else if (params.transform_kind == DCT8X4) {
            value = idct8x4_special(channel, x, y);
        } else if (params.transform_kind >= AFV0) {
            value = afv(task, channel, x, y, params.transform_kind - AFV0);
        }
        write_output(task, channel, x, y, value);
    }
}
