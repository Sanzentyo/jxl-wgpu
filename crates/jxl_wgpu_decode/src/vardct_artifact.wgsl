struct Params {
    dimensions: vec4<u32>,
    capacities: vec4<u32>,
    image: vec4<u32>,
    artifact_offsets: vec4<u32>,
    metadata_offsets: vec4<u32>,
    source_offsets: vec4<u32>,
    channel_geometry: array<vec4<u32>, 3>,
    matrix_offsets: array<vec4<u32>, 7>,
    dequant_scales: vec4<f32>,
    correlation_params: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> raw_metadata: array<i32>;
@group(0) @binding(1) var<storage, read_write> artifact: array<u32>;
@group(0) @binding(2) var<storage, read_write> occupancy: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> resources: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

const STRATEGY_COUNT: u32 = 27u;
const STATUS_ERROR: u32 = 0u;
const STATUS_TASK_COUNT: u32 = 4u;
const STATUS_COEFFICIENT_WORDS: u32 = 5u;
const STATUS_COVERED_BLOCKS: u32 = 6u;
const STATUS_CONSUMED_ENTRIES: u32 = 7u;
const STATUS_BACKEND_REQUIREMENTS: u32 = 8u;
const STATUS_STRATEGY_MASK: u32 = 9u;
const BACKEND_REQUIREMENT_FREQUENCY_CFL_GRID: u32 = 1u;
const ERROR_INVALID_STRATEGY: u32 = 1u;
const ERROR_NON_POSITIVE_HF_MUL: u32 = 2u;
const ERROR_BLOCK_INFO_EXHAUSTED: u32 = 3u;
const ERROR_TRANSFORM_OUTSIDE_LF_GROUP: u32 = 4u;
const ERROR_PASS_GROUP_CROSSING: u32 = 5u;
const ERROR_TRANSFORM_OVERLAP: u32 = 6u;
const ERROR_TASK_CAPACITY: u32 = 7u;
const ERROR_COEFFICIENT_CAPACITY: u32 = 8u;

fn status_offset() -> u32 {
    return params.artifact_offsets.x;
}

fn fail(code: u32, x: u32, y: u32, value: u32) {
    artifact[status_offset() + STATUS_ERROR] = code;
    artifact[status_offset() + 1u] = x;
    artifact[status_offset() + 2u] = y;
    artifact[status_offset() + 3u] = value;
}

fn block_extent(strategy: u32) -> vec2<u32> {
    switch strategy {
        case 4u { return vec2<u32>(2u, 2u); }
        case 5u { return vec2<u32>(4u, 4u); }
        case 6u { return vec2<u32>(1u, 2u); }
        case 7u { return vec2<u32>(2u, 1u); }
        case 8u { return vec2<u32>(1u, 4u); }
        case 9u { return vec2<u32>(4u, 1u); }
        case 10u { return vec2<u32>(2u, 4u); }
        case 11u { return vec2<u32>(4u, 2u); }
        case 18u { return vec2<u32>(8u, 8u); }
        case 19u { return vec2<u32>(4u, 8u); }
        case 20u { return vec2<u32>(8u, 4u); }
        case 21u { return vec2<u32>(16u, 16u); }
        case 22u { return vec2<u32>(8u, 16u); }
        case 23u { return vec2<u32>(16u, 8u); }
        case 24u { return vec2<u32>(32u, 32u); }
        case 25u { return vec2<u32>(16u, 32u); }
        case 26u { return vec2<u32>(32u, 16u); }
        default { return vec2<u32>(1u, 1u); }
    }
}

fn order_id(strategy: u32) -> u32 {
    switch strategy {
        case 0u { return 0u; }
        case 1u, 2u, 3u, 12u, 13u, 14u, 15u, 16u, 17u { return 1u; }
        case 4u { return 2u; }
        case 5u { return 3u; }
        case 6u, 7u { return 4u; }
        case 8u, 9u { return 5u; }
        case 10u, 11u { return 6u; }
        case 18u { return 7u; }
        case 19u, 20u { return 8u; }
        case 21u { return 9u; }
        case 22u, 23u { return 10u; }
        case 24u { return 11u; }
        default { return 12u; }
    }
}

fn needs_transpose(strategy: u32) -> bool {
    switch strategy {
        case 0u, 4u, 5u, 6u, 8u, 10u, 18u, 19u, 21u, 22u, 24u, 25u {
            return true;
        }
        default { return false; }
    }
}

fn is_special(strategy: u32) -> bool {
    return (strategy >= 1u && strategy <= 3u) || (strategy >= 12u && strategy <= 17u);
}

fn matrix_offset(strategy: u32) -> u32 {
    return params.matrix_offsets[strategy / 4u][strategy % 4u];
}

fn occupied(index: u32) -> bool {
    let word = index / 32u;
    let mask = 1u << (index % 32u);
    return (atomicLoad(&occupancy[word]) & mask) != 0u;
}

fn mark_occupied(index: u32) {
    let word = index / 32u;
    let mask = 1u << (index % 32u);
    atomicOr(&occupancy[word], mask);
}

fn clear_workspace() {
    let block_count = params.dimensions.x * params.dimensions.y;
    let occupancy_words = (block_count + 31u) / 32u;
    for (var index = 0u; index < occupancy_words; index = index + 1u) {
        atomicStore(&occupancy[index], 0u);
    }
    for (var index = 0u; index < 16u; index = index + 1u) {
        artifact[status_offset() + index] = 0u;
    }
    for (var index = 0u; index < block_count; index = index + 1u) {
        artifact[params.metadata_offsets.y + index] = 0u;
    }
}

fn validate_transform(
    x: u32,
    y: u32,
    extent: vec2<u32>,
    strategy: u32,
) -> bool {
    if (extent.x > params.dimensions.x - x || extent.y > params.dimensions.y - y) {
        fail(ERROR_TRANSFORM_OUTSIDE_LF_GROUP, x, y, strategy);
        return false;
    }
    let group_dim = params.dimensions.w;
    if (extent.x > group_dim - x % group_dim || extent.y > group_dim - y % group_dim) {
        fail(ERROR_PASS_GROUP_CROSSING, x, y, strategy);
        return false;
    }
    for (var dy = 0u; dy < extent.y; dy = dy + 1u) {
        for (var dx = 0u; dx < extent.x; dx = dx + 1u) {
            let index = (y + dy) * params.dimensions.x + x + dx;
            if (occupied(index)) {
                fail(ERROR_TRANSFORM_OVERLAP, x + dx, y + dy, strategy);
                return false;
            }
        }
    }
    return true;
}

fn occupy_transform(x: u32, y: u32, extent: vec2<u32>) {
    for (var dy = 0u; dy < extent.y; dy = dy + 1u) {
        for (var dx = 0u; dx < extent.x; dx = dx + 1u) {
            mark_occupied((y + dy) * params.dimensions.x + x + dx);
        }
    }
}

fn write_bucket(strategy: u32, task_offset: u32, task_count: u32) {
    let base = params.artifact_offsets.y + strategy * 4u;
    artifact[base] = strategy;
    artifact[base + 1u] = task_offset;
    artifact[base + 2u] = task_count;
    artifact[base + 3u] = params.artifact_offsets.w + strategy * 9u;

    let area = block_extent(strategy).x * block_extent(strategy).y * 64u;
    let special = is_special(strategy);
    let indirect = params.artifact_offsets.w + strategy * 9u;
    artifact[indirect] = select((area + 63u) / 64u, 1u, special);
    artifact[indirect + 1u] = task_count;
    artifact[indirect + 2u] = 1u;
    artifact[indirect + 3u] = select((3u * area + 63u) / 64u, 0u, special);
    artifact[indirect + 4u] = select(task_count, 0u, special);
    artifact[indirect + 5u] = select(1u, 0u, special);
    artifact[indirect + 6u] = select((3u * area + 63u) / 64u, 0u, special);
    artifact[indirect + 7u] = select(task_count, 0u, special);
    artifact[indirect + 8u] = select(1u, 0u, special);
}

fn channel_is_active(channel: u32, x: u32, y: u32) -> bool {
    let geometry = params.channel_geometry[channel];
    let shifted_x = x >> geometry.x;
    let shifted_y = y >> geometry.y;
    if ((shifted_x << geometry.x) != x || (shifted_y << geometry.y) != y) {
        return false;
    }
    if (geometry.x == 0u && geometry.y == 0u) {
        return true;
    }
    let shifted_raster = shifted_y * params.dimensions.x + shifted_x;
    let current_raster = y * params.dimensions.x + x;
    return shifted_raster == current_raster
        || artifact[params.metadata_offsets.y + shifted_raster] != 0u;
}

fn shifted_destination(channel: u32, x: u32, y: u32) -> vec2<u32> {
    let geometry = params.channel_geometry[channel];
    return vec2<u32>(
        (params.image.y >> geometry.x) + (x >> geometry.x) * 8u,
        (params.image.z >> geometry.y) + (y >> geometry.y) * 8u,
    );
}

fn shifted_lf_offset(channel: u32, x: u32, y: u32) -> u32 {
    let geometry = params.channel_geometry[channel];
    let block_x = params.image.y / 8u + x;
    let block_y = params.image.z / 8u + y;
    return (block_y >> geometry.y) * geometry.w + (block_x >> geometry.x);
}

fn write_task(
    task_index: u32,
    strategy: u32,
    raster_index: u32,
    x: u32,
    y: u32,
    extent: vec2<u32>,
    hf_mul: u32,
    coefficient_offset: u32,
) {
    let area = extent.x * extent.y * 64u;
    let task = params.artifact_offsets.z + task_index * 16u;
    let destination_x = shifted_destination(0u, x, y);
    let destination_y = shifted_destination(1u, x, y);
    let destination_b = shifted_destination(2u, x, y);
    var channel_mask = 0u;
    for (var channel = 0u; channel < 3u; channel += 1u) {
        channel_mask |= select(0u, 1u << channel, channel_is_active(channel, x, y));
    }
    artifact[task] = coefficient_offset;
    artifact[task + 1u] = select(coefficient_offset, params.image.w, strategy >= 14u && strategy <= 17u);
    artifact[task + 2u] = matrix_offset(strategy);
    artifact[task + 3u] = task_index;
    artifact[task + 4u] = params.image.y + x * 8u;
    artifact[task + 5u] = shifted_lf_offset(0u, x, y);
    artifact[task + 6u] = channel_mask;
    artifact[task + 7u] = params.image.z + y * 8u;
    artifact[task + 8u] = destination_x.x;
    artifact[task + 9u] = destination_x.y;
    artifact[task + 10u] = destination_y.x;
    artifact[task + 11u] = destination_y.y;
    artifact[task + 12u] = destination_b.x;
    artifact[task + 13u] = destination_b.y;
    artifact[task + 14u] = shifted_lf_offset(1u, x, y);
    artifact[task + 15u] = shifted_lf_offset(2u, x, y);

    let task_metadata = params.metadata_offsets.x + task_index * 12u;
    artifact[task_metadata] = strategy;
    artifact[task_metadata + 1u] = raster_index;
    artifact[task_metadata + 2u] = x;
    artifact[task_metadata + 3u] = y;
    artifact[task_metadata + 4u] = extent.x;
    artifact[task_metadata + 5u] = extent.y;
    artifact[task_metadata + 6u] = hf_mul;
    artifact[task_metadata + 7u] = (y / params.dimensions.w) * params.capacities.z
        + x / params.dimensions.w;
    artifact[task_metadata + 8u] = coefficient_offset;
    artifact[task_metadata + 9u] = 3u * area;
    artifact[task_metadata + 10u] = order_id(strategy);
    artifact[task_metadata + 11u] = select(0u, 1u, needs_transpose(strategy))
        | select(0u, 2u, is_special(strategy)) | (channel_mask << 8u);
    artifact[params.metadata_offsets.y + raster_index] = task_index + 1u;
    let quant_scale = 65536.0 / (f32(params.source_offsets.z) * f32(hf_mul));
    resources[params.metadata_offsets.z + task_index] = vec4<f32>(
        params.dequant_scales.xyz * quant_scale,
        0.0,
    );
}

@compute @workgroup_size(1, 1, 1)
fn lower_hf_metadata() {
    clear_workspace();
    let correlation_count = params.capacities.w * ((params.dimensions.y + 7u) / 8u);
    for (var index = 0u; index < correlation_count; index += 1u) {
        let correlation_x = index % params.capacities.w;
        let correlation_y = index / params.capacities.w;
        let destination = params.metadata_offsets.w
            + correlation_y * params.source_offsets.w + correlation_x;
        resources[destination] = vec4<f32>(
            fma(
                f32(raw_metadata[index]),
                params.correlation_params.z,
                params.correlation_params.x,
            ),
            fma(
                f32(raw_metadata[correlation_count + index]),
                params.correlation_params.z,
                params.correlation_params.y,
            ),
            0.0,
            0.0,
        );
    }
    var counts: array<u32, 27>;
    var consumed = 0u;
    var covered = 0u;
    var coefficient_words = 0u;
    var backend_requirements = 0u;

    for (var y = 0u; y < params.dimensions.y; y = y + 1u) {
        for (var x = 0u; x < params.dimensions.x; x = x + 1u) {
            let raster_index = y * params.dimensions.x + x;
            if (occupied(raster_index)) {
                continue;
            }
            if (consumed >= params.dimensions.z) {
                fail(ERROR_BLOCK_INFO_EXHAUSTED, x, y, consumed);
                return;
            }
            let raw_strategy = raw_metadata[params.source_offsets.x + consumed];
            if (raw_strategy < 0 || raw_strategy >= i32(STRATEGY_COUNT)) {
                fail(ERROR_INVALID_STRATEGY, x, y, bitcast<u32>(raw_strategy));
                return;
            }
            let raw_hf_mul = raw_metadata[params.source_offsets.y + consumed];
            if (raw_hf_mul < 0 || raw_hf_mul == 0x7fffffffi) {
                fail(ERROR_NON_POSITIVE_HF_MUL, x, y, bitcast<u32>(raw_hf_mul));
                return;
            }
            let strategy = u32(raw_strategy);
            let extent = block_extent(strategy);
            if (!validate_transform(x, y, extent, strategy)) {
                return;
            }
            occupy_transform(x, y, extent);
            let area = extent.x * extent.y;
            let pixel_x = params.image.y + x * 8u;
            let pixel_y = params.image.z + y * 8u;
            let pixel_right = params.image.y + (x + extent.x) * 8u - 1u;
            let pixel_bottom = params.image.z + (y + extent.y) * 8u - 1u;
            if (pixel_x / 64u != pixel_right / 64u
                || pixel_y / 64u != pixel_bottom / 64u) {
                backend_requirements |= BACKEND_REQUIREMENT_FREQUENCY_CFL_GRID;
            }
            counts[strategy] = counts[strategy] + 1u;
            covered = covered + area;
            coefficient_words = coefficient_words + area * 192u;
            consumed = consumed + 1u;
        }
    }
    if (consumed > params.capacities.x) {
        fail(ERROR_TASK_CAPACITY, 0u, 0u, consumed);
        return;
    }
    if (coefficient_words > params.capacities.y) {
        fail(ERROR_COEFFICIENT_CAPACITY, 0u, 0u, coefficient_words);
        return;
    }

    var bucket_cursor = 0u;
    var strategy_mask = 0u;
    var bucket_offsets: array<u32, 27>;
    for (var strategy = 0u; strategy < STRATEGY_COUNT; strategy = strategy + 1u) {
        bucket_offsets[strategy] = bucket_cursor;
        write_bucket(strategy, bucket_cursor, counts[strategy]);
        if (counts[strategy] != 0u) {
            strategy_mask |= 1u << strategy;
        }
        bucket_cursor = bucket_cursor + counts[strategy];
    }

    let block_count = params.dimensions.x * params.dimensions.y;
    let occupancy_words = (block_count + 31u) / 32u;
    for (var index = 0u; index < occupancy_words; index = index + 1u) {
        atomicStore(&occupancy[index], 0u);
    }
    var cursors: array<u32, 27>;
    consumed = 0u;
    coefficient_words = 0u;
    for (var y = 0u; y < params.dimensions.y; y = y + 1u) {
        for (var x = 0u; x < params.dimensions.x; x = x + 1u) {
            let raster_index = y * params.dimensions.x + x;
            if (occupied(raster_index)) {
                continue;
            }
            let strategy = u32(raw_metadata[params.source_offsets.x + consumed]);
            let hf_mul = u32(raw_metadata[params.source_offsets.y + consumed] + 1);
            let extent = block_extent(strategy);
            let task_index = bucket_offsets[strategy] + cursors[strategy];
            write_task(
                task_index,
                strategy,
                raster_index,
                x,
                y,
                extent,
                hf_mul,
                coefficient_words,
            );
            cursors[strategy] = cursors[strategy] + 1u;
            coefficient_words = coefficient_words + extent.x * extent.y * 192u;
            consumed = consumed + 1u;
            occupy_transform(x, y, extent);
        }
    }

    artifact[status_offset() + STATUS_TASK_COUNT] = bucket_cursor;
    artifact[status_offset() + STATUS_COEFFICIENT_WORDS] = coefficient_words;
    artifact[status_offset() + STATUS_COVERED_BLOCKS] = covered;
    artifact[status_offset() + STATUS_CONSUMED_ENTRIES] = consumed;
    artifact[status_offset() + STATUS_BACKEND_REQUIREMENTS] = backend_requirements;
    artifact[status_offset() + STATUS_STRATEGY_MASK] = strategy_mask;
}
