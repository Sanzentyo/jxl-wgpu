struct Task {
    coefficient_offset: u32,
    destination_x: u32,
    destination_y: u32,
    quant_index: u32,
    matrix_index: u32,
    lf_index: u32,
    coefficient_origin_x: u32,
    coefficient_origin_y: u32,
};

struct Params {
    task_count: u32,
    output_width: u32,
    output_height: u32,
    output_stride_x: u32,
    output_stride_y: u32,
    output_stride_b: u32,
    quant_offset: u32,
    matrix_offset: u32,
    correlation_offset: u32,
    lf_offset: u32,
    correlation_width: u32,
    correlation_height: u32,
    quant_biases: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> coefficients: array<i32>;
@group(0) @binding(1) var<storage, read> tasks: array<Task>;
@group(0) @binding(2) var<storage, read> resources: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_b: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> dequantized: array<f32, 192>;
var<workgroup> horizontal: array<f32, 192>;

// The same eight-point butterfly generated in jxl_transforms/src/idct8.rs. Keeping the
// operation graph aligned with the CPU implementation both avoids transcendental instructions
// and minimizes CPU/GPU rounding drift.
fn idct8(input: array<f32, 8>) -> array<f32, 8> {
    let v8 = input[0] + input[4];
    let v9 = input[0] - input[4];
    let v10 = input[2] + input[6];
    let v11 = input[2] * 1.4142135623730951;
    let v12 = v11 + v10;
    let v13 = v11 - v10;
    let v14 = fma(v12, 0.5411961001461970, v8);
    let v15 = fma(-v12, 0.5411961001461970, v8);
    let v16 = fma(v13, 1.3065629648763764, v9);
    let v17 = fma(-v13, 1.3065629648763764, v9);
    let v18 = input[1] + input[3];
    let v19 = input[3] + input[5];
    let v20 = input[5] + input[7];
    let v21 = input[1] * 1.4142135623730951;
    let v22 = v21 + v19;
    let v23 = v21 - v19;
    let v24 = v18 + v20;
    let v25 = v18 * 1.4142135623730951;
    let v26 = v25 + v24;
    let v27 = v25 - v24;
    let v28 = fma(v26, 0.5411961001461970, v22);
    let v29 = fma(-v26, 0.5411961001461970, v22);
    let v30 = fma(v27, 1.3065629648763764, v23);
    let v31 = fma(-v27, 1.3065629648763764, v23);
    let v32 = fma(v28, 0.5097955791041592, v14);
    let v33 = fma(-v28, 0.5097955791041592, v14);
    let v34 = fma(v30, 0.6013448869350453, v16);
    let v35 = fma(-v30, 0.6013448869350453, v16);
    let v36 = fma(v31, 0.8999762231364156, v17);
    let v37 = fma(-v31, 0.8999762231364156, v17);
    let v38 = fma(v29, 2.5629154477415055, v15);
    let v39 = fma(-v29, 2.5629154477415055, v15);
    return array<f32, 8>(v32, v34, v36, v38, v39, v37, v35, v33);
}

fn adjust_quantized(value: i32, small_bias: f32, large_bias: f32) -> f32 {
    let value_f = f32(value);
    if (value > -2 && value < 2) {
        return value_f * small_bias;
    }
    return value_f - large_bias / value_f;
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let task_index = workgroup_id.x;
    if (task_index >= params.task_count) {
        return;
    }
    let task = tasks[task_index];
    let quant_scale = resources[params.quant_offset + task.quant_index];
    let matrix_scale = resources[
        params.matrix_offset + task.matrix_index * 64u + lane
    ];
    let frequency_x = lane / 8u;
    let frequency_y = lane % 8u;
    let correlation_cell_x = (task.coefficient_origin_x + frequency_x) / 64u;
    let correlation_cell_y = (task.coefficient_origin_y + frequency_y) / 64u;
    let correlation = resources[
        params.correlation_offset + correlation_cell_y * params.correlation_width
            + correlation_cell_x
    ];

    let quantized_x = coefficients[task.coefficient_offset + lane];
    let quantized_y = coefficients[task.coefficient_offset + 64u + lane];
    let quantized_b = coefficients[task.coefficient_offset + 128u + lane];
    let value_x = adjust_quantized(
        quantized_x, params.quant_biases.x, params.quant_biases.w
    ) * quant_scale.x * matrix_scale.x;
    let value_y = adjust_quantized(
        quantized_y, params.quant_biases.y, params.quant_biases.w
    ) * quant_scale.y * matrix_scale.y;
    let value_b = adjust_quantized(
        quantized_b, params.quant_biases.z, params.quant_biases.w
    ) * quant_scale.z * matrix_scale.z;
    if (lane == 0u) {
        let lf = resources[params.lf_offset + task.lf_index];
        dequantized[0u] = lf.x;
        dequantized[64u] = lf.y;
        dequantized[128u] = lf.z;
    } else {
        dequantized[lane] = fma(correlation.x, value_y, value_x);
        dequantized[64u + lane] = value_y;
        dequantized[128u + lane] = fma(correlation.y, value_y, value_b);
    }
    workgroupBarrier();

    // Twenty-four lanes each own one channel/frequency row. A lane computes all eight spatial
    // values so the butterfly is evaluated once per row instead of redundantly per output.
    if (lane < 24u) {
        let channel = lane / 8u;
        let frequency_y = lane % 8u;
        let source = channel * 64u + frequency_y * 8u;
        let spatial = idct8(array<f32, 8>(
            dequantized[source],
            dequantized[source + 1u],
            dequantized[source + 2u],
            dequantized[source + 3u],
            dequantized[source + 4u],
            dequantized[source + 5u],
            dequantized[source + 6u],
            dequantized[source + 7u],
        ));
        for (var x = 0u; x < 8u; x = x + 1u) {
            horizontal[source + x] = spatial[x];
        }
    }
    workgroupBarrier();

    // The same lanes now own one channel/spatial-x coordinate and transform its frequency column.
    if (lane < 24u) {
        let channel = lane / 8u;
        let x = lane % 8u;
        let source = channel * 64u + x;
        let spatial = idct8(array<f32, 8>(
            horizontal[source],
            horizontal[source + 8u],
            horizontal[source + 16u],
            horizontal[source + 24u],
            horizontal[source + 32u],
            horizontal[source + 40u],
            horizontal[source + 48u],
            horizontal[source + 56u],
        ));
        for (var y = 0u; y < 8u; y = y + 1u) {
            // jxl_transforms::idct2d_8_8 exposes this transposed spatial order.
            let destination_x = task.destination_x + y;
            let destination_y = task.destination_y + x;
            if (destination_x < params.output_width && destination_y < params.output_height) {
                if (channel == 0u) {
                    output_x[destination_y * params.output_stride_x + destination_x] = spatial[y];
                } else if (channel == 1u) {
                    output_y[destination_y * params.output_stride_y + destination_x] = spatial[y];
                } else {
                    output_b[destination_y * params.output_stride_b + destination_x] = spatial[y];
                }
            }
        }
    }
}
