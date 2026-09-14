// Raw JPEG XL forward transforms. All source-dependent arithmetic stays on GPU.
// Regular transforms use two storage-separated passes; special transforms use
// the strategy-only basis. A final pass derives LF from the complete transform.

struct Params {
    width: u32,
    height: u32,
    area: u32,
    lf_width: u32,
    lf_height: u32,
    lf_area: u32,
    task_count: u32,
    workgroups_x: u32,
    strides: vec4<u32>,
    basis_offsets: vec4<u32>,
}

struct Task {
    origins: array<u32, 3>,
    coefficient_offset: u32,
    lf_offset: u32,
}
@group(0) @binding(8) var<storage, read> tasks: array<Task>;

@group(0) @binding(0) var<storage, read> source_x: array<f32>;
@group(0) @binding(1) var<storage, read> source_y: array<f32>;
@group(0) @binding(2) var<storage, read> source_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> horizontal: array<f32>;
@group(0) @binding(4) var<storage, read_write> coefficients: array<f32>;
@group(0) @binding(5) var<storage, read_write> low_frequency: array<f32>;
@group(0) @binding(6) var<storage, read> basis: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

override wg_x: u32 = 64u;

fn index(group: vec3<u32>, lane: u32) -> u32 {
    return (group.y * params.workgroups_x + group.x) * wg_x + lane;
}

fn load_pixel(task: Task, x: u32, y: u32) -> vec3<f32> {
    let address = vec3<u32>(task.origins[0], task.origins[1], task.origins[2]) + y * params.strides.xyz + vec3<u32>(x);
    return vec3<f32>(source_x[address.x], source_y[address.y], source_b[address.z]);
}

fn wire_index(fx: u32, fy: u32) -> u32 {
    if params.height < params.width { return fy * params.width + fx; }
    return fx * params.height + fy;
}

fn load_coefficient(task: Task, fx: u32, fy: u32) -> vec3<f32> {
    let offset = task.coefficient_offset + wire_index(fx, fy);
    return vec3<f32>(coefficients[offset], coefficients[params.area + offset], coefficients[2u * params.area + offset]);
}

@compute @workgroup_size(wg_x, 1, 1)
fn horizontal_dct(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let batch_item = index(group, lane);
    if batch_item >= params.area * params.task_count { return; }
    let task_index = batch_item / params.area;
    let task = tasks[task_index];
    let item = batch_item % params.area;
    let horizontal_base = task_index * 3u * params.area;
    let fx = item % params.width;
    let y = item / params.width;
    var sum = vec3<f32>(0.0);
    for (var x = 0u; x < params.width; x += 1u) {
        sum += load_pixel(task, x, y) * basis[params.basis_offsets.x + fx * params.width + x];
    }
    horizontal[horizontal_base + item] = sum.x;
    horizontal[horizontal_base + params.area + item] = sum.y;
    horizontal[horizontal_base + 2u * params.area + item] = sum.z;
}

@compute @workgroup_size(wg_x, 1, 1)
fn vertical_dct(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let batch_item = index(group, lane);
    if batch_item >= params.area * params.task_count { return; }
    let task_index = batch_item / params.area;
    let task = tasks[task_index];
    let item = batch_item % params.area;
    let horizontal_base = task_index * 3u * params.area;
    let fx = item % params.width;
    let fy = item / params.width;
    var sum = vec3<f32>(0.0);
    for (var y = 0u; y < params.height; y += 1u) {
        let offset = horizontal_base + y * params.width + fx;
        let value = vec3<f32>(horizontal[offset], horizontal[params.area + offset], horizontal[2u * params.area + offset]);
        sum += value * basis[params.basis_offsets.y + fy * params.height + y];
    }
    let offset = task.coefficient_offset + wire_index(fx, fy);
    coefficients[offset] = sum.x;
    coefficients[params.area + offset] = sum.y;
    coefficients[2u * params.area + offset] = sum.z;
}

@compute @workgroup_size(wg_x, 1, 1)
fn special_transform(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let batch_item = index(group, lane);
    if batch_item >= 64u * params.task_count { return; }
    let task = tasks[batch_item / 64u];
    let item = batch_item % 64u;
    var sum = vec3<f32>(0.0);
    for (var pixel = 0u; pixel < 64u; pixel += 1u) {
        sum += load_pixel(task, pixel % 8u, pixel / 8u) * basis[item * 64u + pixel];
    }
    coefficients[task.coefficient_offset + item] = sum.x;
    coefficients[task.coefficient_offset + 64u + item] = sum.y;
    coefficients[task.coefficient_offset + 128u + item] = sum.z;
}

@compute @workgroup_size(wg_x, 1, 1)
fn extract_lf(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let batch_item = index(group, lane);
    if batch_item >= params.lf_area * params.task_count { return; }
    let task = tasks[batch_item / params.lf_area];
    let item = batch_item % params.lf_area;
    let x = item % params.lf_width;
    let y = item / params.lf_width;
    var sum = vec3<f32>(0.0);
    if params.lf_area == 1u {
        low_frequency[task.lf_offset + 0] = coefficients[task.coefficient_offset + 0];
        low_frequency[task.lf_offset + 1] = coefficients[task.coefficient_offset + params.area];
        low_frequency[task.lf_offset + 2] = coefficients[task.coefficient_offset + 2u * params.area];
        return;
    }
    for (var fy = 0u; fy < params.lf_height; fy += 1u) {
        for (var fx = 0u; fx < params.lf_width; fx += 1u) {
            let weight = basis[params.basis_offsets.z + fx * params.lf_width + x]
                * basis[params.basis_offsets.w + fy * params.lf_height + y];
            sum += load_coefficient(task, fx, fy) * weight;
        }
    }
    low_frequency[task.lf_offset + item] = sum.x;
    low_frequency[task.lf_offset + params.lf_area + item] = sum.y;
    low_frequency[task.lf_offset + 2u * params.lf_area + item] = sum.z;
}
