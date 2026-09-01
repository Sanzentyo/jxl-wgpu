struct Task {
    coefficient_offset: u32,
    scratch_or_basis_offset: u32,
    matrix_offset: u32,
    quant_index: u32,
    coefficient_origin_x: u32,
    lf_offset_x: u32,
    channel_mask: u32,
    coefficient_origin_y: u32,
    destination_x_x: u32,
    destination_y_x: u32,
    destination_x_y: u32,
    destination_y_y: u32,
    destination_x_b: u32,
    destination_y_b: u32,
    lf_offset_y: u32,
    lf_offset_b: u32,
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
    lf_offset_x: u32,
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
    task_word_offset: u32,
    bucket_word_offset: u32,
    lf_stride_x: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    quant_biases: vec4<f32>,
    lf_channel_layout: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> coefficients: array<i32>;
@group(0) @binding(1) var<storage, read> task_words: array<u32>;
@group(0) @binding(2) var<storage, read> resources: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dequantized: array<f32>;
@group(0) @binding(4) var<storage, read_write> horizontal: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_x: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(7) var<storage, read_write> output_b: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

const PI: f32 = 3.14159265358979323846;
const SQRT_2: f32 = 1.41421356237309504880;

fn idct_basis(position: u32, frequency: u32, length: u32) -> f32 {
    if (frequency == 0u) {
        return 1.0;
    }
    let phase = f32((2u * position + 1u) * frequency) * PI / (2.0 * f32(length));
    return SQRT_2 * cos(phase);
}

fn reinterpret_scale(frequency: u32, length: u32) -> f32 {
    let value = f32(frequency) * PI / f32(length);
    return cos(value / 16.0) * cos(value / 8.0) * cos(value / 4.0) * f32(length);
}

// The JPEG XL transform buffer is frequency-X-major for square/tall transforms and
// frequency-Y-major for wide transforms. This is the order consumed by jxl_transforms.
fn transform_index(frequency_x: u32, frequency_y: u32) -> u32 {
    if (params.transform_height < params.transform_width) {
        return frequency_y * params.transform_width + frequency_x;
    }
    return frequency_x * params.transform_height + frequency_y;
}

fn adjust_quantized(value: i32, small_bias: f32, large_bias: f32) -> f32 {
    let value_f = f32(value);
    if (value > -2 && value < 2) {
        return value_f * small_bias;
    }
    return value_f - large_bias / value_f;
}

fn hf_correlation(task: Task, frequency_x: u32, frequency_y: u32) -> vec2<f32> {
    let cell_x = (task.coefficient_origin_x + frequency_x) / 64u;
    let cell_y = (task.coefficient_origin_y + frequency_y) / 64u;
    return resources[params.correlation_offset + cell_y * params.correlation_width + cell_x].xy;
}

fn lf_coefficient(task: Task, frequency_x: u32, frequency_y: u32) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    let lf_strides = vec3<u32>(
        select(params.lf_width, params.lf_stride_x, params.lf_stride_x != 0u),
        select(params.lf_width, params.lf_channel_layout.z, params.lf_channel_layout.z != 0u),
        select(params.lf_width, params.lf_channel_layout.w, params.lf_channel_layout.w != 0u),
    );
    let lf_bases = vec3<u32>(
        params.lf_offset_x + task.lf_offset_x,
        params.lf_channel_layout.x + task.lf_offset_y,
        params.lf_channel_layout.y + task.lf_offset_b,
    );
    for (var spatial_y = 0u; spatial_y < params.lf_height; spatial_y = spatial_y + 1u) {
        let basis_y = idct_basis(spatial_y, frequency_y, params.lf_height);
        for (var spatial_x = 0u; spatial_x < params.lf_width; spatial_x = spatial_x + 1u) {
            let basis_x = idct_basis(spatial_x, frequency_x, params.lf_width);
            let lf = vec3<f32>(
                resources[lf_bases.x + spatial_y * lf_strides.x + spatial_x].x,
                resources[lf_bases.y + spatial_y * lf_strides.y + spatial_x].y,
                resources[lf_bases.z + spatial_y * lf_strides.z + spatial_x].z,
            );
            sum = fma(lf, vec3<f32>(basis_x * basis_y), sum);
        }
    }
    return sum / vec3<f32>(
        reinterpret_scale(frequency_x, params.lf_width)
            * reinterpret_scale(frequency_y, params.lf_height)
    );
}

fn load_task(index: u32) -> Task {
    let base = params.task_word_offset + index * 16u;
    return Task(
        task_words[base],
        task_words[base + 1u],
        task_words[base + 2u],
        task_words[base + 3u],
        task_words[base + 4u],
        task_words[base + 5u],
        task_words[base + 6u],
        task_words[base + 7u],
        task_words[base + 8u],
        task_words[base + 9u],
        task_words[base + 10u],
        task_words[base + 11u],
        task_words[base + 12u],
        task_words[base + 13u],
        task_words[base + 14u],
        task_words[base + 15u],
    );
}

fn task_for_workgroup(workgroup_y: u32) -> Task {
    var task_base = params.task_base;
    if (params.bucket_word_offset != 0xffffffffu) {
        task_base = task_words[params.bucket_word_offset + params.transform_kind * 4u + 1u];
    }
    return load_task(task_base + workgroup_y);
}

@compute @workgroup_size(64, 1, 1)
fn dequantize(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if (workgroup_id.y >= params.task_count) {
        return;
    }
    let natural_index = workgroup_id.x * 64u + lane;
    if (natural_index >= params.transform_area) {
        return;
    }
    let task = task_for_workgroup(workgroup_id.y);
    let frequency_y = natural_index / params.transform_width;
    let frequency_x = natural_index % params.transform_width;
    let packed_index = transform_index(frequency_x, frequency_y);
    var values: vec3<f32>;
    if (frequency_x < params.lf_width && frequency_y < params.lf_height) {
        values = lf_coefficient(task, frequency_x, frequency_y);
    } else {
        let matrix = resources[task.matrix_offset + packed_index].xyz;
        let quant = resources[params.quant_offset + task.quant_index].xyz;
        let coefficient_base = task.coefficient_offset + packed_index;
        let x = adjust_quantized(
            coefficients[coefficient_base], params.quant_biases.x, params.quant_biases.w
        ) * quant.x * matrix.x;
        let y = adjust_quantized(
            coefficients[coefficient_base + params.transform_area],
            params.quant_biases.y,
            params.quant_biases.w,
        ) * quant.y * matrix.y;
        let b = adjust_quantized(
            coefficients[coefficient_base + 2u * params.transform_area],
            params.quant_biases.z,
            params.quant_biases.w,
        ) * quant.z * matrix.z;
        let correlation = hf_correlation(task, frequency_x, frequency_y);
        values = vec3<f32>(fma(correlation.x, y, x), y, fma(correlation.y, y, b));
    }
    dequantized[task.scratch_or_basis_offset + natural_index] = values.x;
    dequantized[task.scratch_or_basis_offset + params.transform_area + natural_index] = values.y;
    dequantized[task.scratch_or_basis_offset + 2u * params.transform_area + natural_index] = values.z;
}

@compute @workgroup_size(64, 1, 1)
fn horizontal_idct(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if (workgroup_id.y >= params.task_count) {
        return;
    }
    let item = workgroup_id.x * 64u + lane;
    if (item >= 3u * params.transform_area) {
        return;
    }
    let task = task_for_workgroup(workgroup_id.y);
    let channel = item / params.transform_area;
    if ((task.channel_mask >> channel) & 1u) == 0u {
        return;
    }
    let natural_index = item % params.transform_area;
    let frequency_y = natural_index / params.transform_width;
    let spatial_x = natural_index % params.transform_width;
    let channel_offset = task.scratch_or_basis_offset + channel * params.transform_area;
    var sum = 0.0;
    for (var frequency_x = 0u; frequency_x < params.transform_width; frequency_x = frequency_x + 1u) {
        sum = fma(
            dequantized[channel_offset + frequency_y * params.transform_width + frequency_x],
            idct_basis(spatial_x, frequency_x, params.transform_width),
            sum,
        );
    }
    horizontal[channel_offset + natural_index] = sum;
}

@compute @workgroup_size(64, 1, 1)
fn vertical_idct(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if (workgroup_id.y >= params.task_count) {
        return;
    }
    let item = workgroup_id.x * 64u + lane;
    if (item >= 3u * params.transform_area) {
        return;
    }
    let task = task_for_workgroup(workgroup_id.y);
    let channel = item / params.transform_area;
    if ((task.channel_mask >> channel) & 1u) == 0u {
        return;
    }
    let natural_index = item % params.transform_area;
    let spatial_y = natural_index / params.transform_width;
    let spatial_x = natural_index % params.transform_width;
    let channel_offset = task.scratch_or_basis_offset + channel * params.transform_area;
    var sum = 0.0;
    for (var frequency_y = 0u; frequency_y < params.transform_height; frequency_y = frequency_y + 1u) {
        sum = fma(
            horizontal[channel_offset + frequency_y * params.transform_width + spatial_x],
            idct_basis(spatial_y, frequency_y, params.transform_height),
            sum,
        );
    }

    if (channel == 0u) {
        let x = task.destination_x_x + spatial_x;
        let y = task.destination_y_x + spatial_y;
        if (x < params.output_width_x && y < params.output_height_x) {
            output_x[y * params.output_stride_x + x] = sum;
        }
    } else if (channel == 1u) {
        let x = task.destination_x_y + spatial_x;
        let y = task.destination_y_y + spatial_y;
        if (x < params.output_width_y && y < params.output_height_y) {
            output_y[y * params.output_stride_y + x] = sum;
        }
    } else {
        let x = task.destination_x_b + spatial_x;
        let y = task.destination_y_b + spatial_y;
        if (x < params.output_width_b && y < params.output_height_b) {
            output_b[y * params.output_stride_b + x] = sum;
        }
    }
}
