// GPU JPEG scan encoding from validated resident JXL coefficients. Status gates every stage.
// All entropy-sized prefix sums stay on GPU using hierarchical workgroup scans.
struct Params {
    blocks: u32,
    coefficient_words: u32,
    raw_words: u32,
    output_words: u32,
    padding_bits: u32,
    has_padding: u32,
    progressive_base: u32,
    scan_parameters: u32,
    dispatch_width: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}
struct Task {
    coefficient: u32,
    previous_dc: u32,
    dc_table: u32,
    ac_table: u32,
    extra_zeros: u32,
    marker_after: u32,
    segment_first_block: u32,
    preceding_markers: u32,
    reset_before: u32,
    _reserved0: u32,
    _reserved1: u32,
    _reserved2: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> coefficients: array<i32>;
// A code is (length << 16) | canonical_value; padding bits follow the 2048 code words.
@group(0) @binding(2) var<storage, read> tables: array<u32>;
@group(0) @binding(3) var<storage, read> tasks: array<Task>;
@group(0) @binding(4) var<storage, read_write> work: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> raw: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> output: array<atomic<u32>>;
struct ScanParams {
    source: u32,
    destination: u32,
    count: u32,
    scratch: u32,
}
@group(0) @binding(7) var<uniform> scan: ScanParams;
@group(0) @binding(8) var<storage, read_write> padding_state: array<atomic<u32>>;
var<workgroup> scan_values: array<u32, 64>;
const ZIGZAG = array<u32, 64>(
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,
    12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63);
fn linear_index(id: vec3<u32>) -> u32 { return id.x + id.y * params.dispatch_width * 64u; }
fn group_index(id: vec3<u32>) -> u32 { return id.x + id.y * params.dispatch_width; }
fn byte_base() -> u32 { return 4u + params.blocks * 9u; }
var<private> cursor: u32;
var<private> emitting: bool;
fn fail(code: u32) { atomicMax(&work[0], code); }
fn put(length: u32, value: u32) {
    if (length == 0u) { return; }
    if (length > 16u || value >= (1u << length)) { fail(1u); return; }
    if (emitting) {
        if (cursor > params.raw_words * 32u || length > params.raw_words * 32u - cursor) {
            fail(2u); return;
        }
        let first = min(length, 32u - (cursor & 31u));
        atomicOr(&raw[cursor >> 5u], (value >> (length - first)) << (32u - (cursor & 31u) - first));
        if (first < length) {
            atomicOr(&raw[(cursor >> 5u) + 1u], value << (32u - (length - first)));
        }
    }
    cursor += length;
}
fn symbol(base: u32, value: u32) {
    if (base > arrayLength(&tables) || value >= arrayLength(&tables) - base) { fail(3u); return; }
    let packed = tables[base + value];
    let length = packed >> 16u;
    if (length == 0u) { fail(4u); return; }
    put(length, packed & 65535u);
}
fn magnitude_bits(value: i32) -> u32 { return 32u - countLeadingZeros(u32(abs(value))); }
fn amplitude(value: i32, length: u32) {
    if (length == 0u) { return; }
    let encoded = select(value, value - 1, value < 0);
    put(length, u32(encoded) & ((1u << length) - 1u));
}
// Progressive block coding reads original-order coefficients restored on the GPU.
// Per-block body/trailer classification forms EOB runs with a parallel prefix maximum.
// Only run heads emit the EOB symbol; every member owns its own correction bits.
fn spectral_start() -> u32 { return params.scan_parameters & 255u; }
fn spectral_end() -> u32 { return (params.scan_parameters >> 8u) & 255u; }
fn approximation_high() -> u32 { return (params.scan_parameters >> 16u) & 255u; }
fn approximation_low() -> u32 { return params.scan_parameters >> 24u; }
fn progressive_slot(plane: u32, index: u32) -> u32 {
    return params.progressive_base + plane * params.blocks + index;
}

@compute @workgroup_size(64)
fn classify_blocks(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = linear_index(id);
    if (i >= params.blocks) { return; }
    let start = spectral_start();
    let end = spectral_end();
    let high = approximation_high();
    let low = approximation_low();
    if (start > end || end > 63u || high > 13u || low > 13u
        || (high != 0u && high != low + 1u)
        || (start == 0u && end != 0u && !(end == 63u && high == 0u && low == 0u))) {
        fail(13u); return;
    }
    if (params.blocks > arrayLength(&tasks)) { fail(5u); return; }
    let task = tasks[i];
    if (task.coefficient > params.coefficient_words || 64u > params.coefficient_words - task.coefficient
        || params.coefficient_words > arrayLength(&coefficients)
        || task.dc_table >= 1024u || task.ac_table < 1024u || task.ac_table > 1792u
        || (task.dc_table & 255u) != 0u || (task.ac_table & 255u) != 0u
        || arrayLength(&tables) < 2048u || task.reset_before > 1u) { fail(5u); return; }
    if (task.previous_dc != 0xffffffffu) {
        if (task.previous_dc >= params.coefficient_words) { fail(5u); return; }
        let previous = coefficients[task.previous_dc];
        if (previous < -2047 || previous > 2047) { fail(6u); return; }
    }
    for (var k = 0u; k < 64u; k++) {
        let value = coefficients[task.coefficient + k];
        if (value < -2047 || value > 2047) { fail(6u); return; }
    }
    if (end == 0u) {
        if (task.extra_zeros != 0u) { fail(7u); return; }
        atomicStore(&work[progressive_slot(0u, i)], 1u);
        return;
    }
    let first = max(start, 1u);
    var last = first - 1u;
    for (var k = first; k <= end; k++) {
        let magnitude = u32(abs(coefficients[task.coefficient + ZIGZAG[k]])) >> low;
        if ((high == 0u && magnitude != 0u) || (high != 0u && magnitude == 1u)) { last = k; }
    }
    var trailing = end - last;
    if (high == 0u) {
        if (task.extra_zeros > trailing / 16u) { fail(7u); return; }
        trailing -= task.extra_zeros * 16u;
    } else if (task.extra_zeros != 0u) {
        // Refinement extra-ZRL source preservation is not yet qualified.
        fail(14u); return;
    }
    let body = last >= first || task.extra_zeros != 0u;
    let flags = select(0u, 1u, body) | select(0u, 2u, trailing != 0u);
    atomicStore(&work[progressive_slot(0u, i)], flags);
}

@compute @workgroup_size(64)
fn seed_run_heads(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = linear_index(id);
    if (i >= params.blocks || atomicLoad(&work[0]) != 0u) { return; }
    let flags = atomicLoad(&work[progressive_slot(0u, i)]);
    var previous_trailing = false;
    if (i != 0u) { previous_trailing = (atomicLoad(&work[progressive_slot(0u, i - 1u)]) & 2u) != 0u; }
    let begins = i == 0u || !previous_trailing || (flags & 1u) != 0u || (flags & 2u) == 0u
        || tasks[i].reset_before != 0u || tasks[i].segment_first_block == i || spectral_start() == 0u;
    atomicStore(&work[progressive_slot(1u, i)], select(0u, i + 1u, begins));
}

@compute @workgroup_size(64)
fn attach_run_lengths(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = linear_index(id);
    if (i >= params.blocks || atomicLoad(&work[0]) != 0u) { return; }
    if ((atomicLoad(&work[progressive_slot(0u, i)]) & 2u) == 0u) { return; }
    let encoded_head = atomicLoad(&work[progressive_slot(2u, i)]);
    if (encoded_head == 0u || encoded_head > i + 1u) { fail(15u); return; }
    let first = encoded_head - 1u;
    let head = first + ((i - first) / 32767u) * 32767u;
    atomicStore(&work[progressive_slot(2u, i)], head);
    atomicMax(&work[progressive_slot(3u, head)], i - head + 1u);
}

fn eob_header(index: u32, table: u32) {
    if ((atomicLoad(&work[progressive_slot(0u, index)]) & 2u) == 0u) { return; }
    if (atomicLoad(&work[progressive_slot(2u, index)]) != index) { return; }
    let count = atomicLoad(&work[progressive_slot(3u, index)]);
    if (count == 0u || count > 32767u) { fail(15u); return; }
    let width = 31u - countLeadingZeros(count);
    symbol(table, width * 16u);
    put(width, count - (1u << width));
}

fn first_block(index: u32) {
    let task = tasks[index];
    let low = approximation_low();
    let start = spectral_start();
    let end = spectral_end();
    if (start == 0u) {
        let dc = coefficients[task.coefficient] >> low;
        var previous = 0;
        if (task.previous_dc != 0xffffffffu) { previous = coefficients[task.previous_dc] >> low; }
        let difference = dc - previous;
        let length = magnitude_bits(difference);
        symbol(task.dc_table, length);
        amplitude(difference, length);
    }
    if (end == 0u) { return; }
    var zeros = 0u;
    for (var k = max(start, 1u); k <= end; k++) {
        let source = coefficients[task.coefficient + ZIGZAG[k]];
        let magnitude = u32(abs(source)) >> low;
        if (magnitude == 0u) { zeros++; continue; }
        while (zeros >= 16u) { symbol(task.ac_table, 240u); zeros -= 16u; }
        let length = 32u - countLeadingZeros(magnitude);
        symbol(task.ac_table, zeros * 16u + length);
        amplitude(select(i32(magnitude), -i32(magnitude), source < 0), length);
        zeros = 0u;
    }
    for (var k = 0u; k < task.extra_zeros; k++) { symbol(task.ac_table, 240u); zeros -= 16u; }
    eob_header(index, task.ac_table);
}

fn refinement_block(index: u32) {
    let task = tasks[index];
    let low = approximation_low();
    let start = spectral_start();
    let end = spectral_end();
    if (start == 0u) { put(1u, u32((coefficients[task.coefficient] >> low) & 1)); }
    if (end == 0u) { return; }
    var last_new = 0u;
    var magnitudes: array<u32, 64>;
    for (var k = max(start, 1u); k <= end; k++) {
        let magnitude = u32(abs(coefficients[task.coefficient + ZIGZAG[k]])) >> low;
        magnitudes[k] = magnitude;
        if (magnitude == 1u) { last_new = k; }
    }
    var zeros = 0u;
    var correction_count = 0u;
    var correction: array<u32, 64>;
    for (var k = max(start, 1u); k <= end; k++) {
        let magnitude = magnitudes[k];
        if (magnitude == 0u) { zeros++; continue; }
        while (zeros >= 16u && k <= last_new) {
            symbol(task.ac_table, 240u);
            zeros -= 16u;
            for (var c = 0u; c < correction_count; c++) { put(1u, correction[c]); }
            correction_count = 0u;
        }
        if (magnitude > 1u) {
            correction[correction_count] = magnitude & 1u;
            correction_count++;
            continue;
        }
        symbol(task.ac_table, zeros * 16u + 1u);
        put(1u, select(0u, 1u, coefficients[task.coefficient + ZIGZAG[k]] > 0));
        for (var c = 0u; c < correction_count; c++) { put(1u, correction[c]); }
        correction_count = 0u;
        zeros = 0u;
    }
    eob_header(index, task.ac_table);
    for (var c = 0u; c < correction_count; c++) { put(1u, correction[c]); }
}

fn encode_block(index: u32) {
    if (atomicLoad(&work[0]) != 0u) { return; }
    if (approximation_high() == 0u) { first_block(index); }
    else { refinement_block(index); }
}

// Inclusive prefix maximum: parent carries use the preceding group's inclusive maximum.
@compute @workgroup_size(64)
fn max_scan_groups(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let i = linear_index(id);
    if (group_index(group) >= (scan.count + 63u) / 64u) { return; }
    let lane = local.x;
    var value = 0u;
    if (i < scan.count) { value = atomicLoad(&work[scan.source + i]); }
    scan_values[lane] = value;
    workgroupBarrier();
    for (var distance = 1u; distance < 64u; distance <<= 1u) {
        var previous = 0u;
        if (lane >= distance) { previous = scan_values[lane - distance]; }
        workgroupBarrier();
        scan_values[lane] = max(scan_values[lane], previous);
        workgroupBarrier();
    }
    if (i < scan.count) { atomicStore(&work[scan.destination + i], scan_values[lane]); }
    if (lane == 63u) { atomicStore(&work[scan.scratch + group_index(group)], scan_values[63]); }
}

@compute @workgroup_size(64)
fn max_scan_carries(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = linear_index(id);
    if (i >= scan.count || i < 64u) { return; }
    let value = atomicLoad(&work[scan.destination + i]);
    let carry = atomicLoad(&work[scan.source + i / 64u - 1u]);
    atomicStore(&work[scan.destination + i], max(value, carry));
}

@compute @workgroup_size(64)
fn count_blocks(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = linear_index(id);
    if (index >= params.blocks) { return; }
    cursor = 0u; emitting = false;
    encode_block(index);
    atomicStore(&work[4u + index], cursor);
}
@compute @workgroup_size(64)
fn prepare_block_sizes(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = linear_index(id);
    if (i >= params.blocks || atomicLoad(&work[0]) != 0u) { return; }
    let task = tasks[i];
    let marker = task.marker_after;
    if (marker != 0u && (marker < 208u || marker > 215u || i + 1u == params.blocks)) { fail(8u); return; }
    if (task.segment_first_block > i || task.preceding_markers > i) { fail(12u); return; }
    let count = atomicLoad(&work[4u + i]);
    var padding = 0u;
    if (marker != 0u || i + 1u == params.blocks) {
        let start = atomicLoad(&work[4u + params.blocks * 3u + task.segment_first_block]);
        let prefix = atomicLoad(&work[4u + params.blocks * 3u + i]);
        if (prefix < start || count > 0xffffffffu - prefix) { fail(12u); return; }
        let segment_bits = prefix - start + count;
        padding = (8u - (segment_bits & 7u)) & 7u;
    }
    let extra = padding + select(0u, 16u, marker != 0u);
    if (count > 0xffffffffu - extra) { fail(12u); return; }
    atomicStore(&work[4u + params.blocks * 2u + i], padding);
    atomicStore(&work[4u + params.blocks * 4u + i], count + extra);
}
@compute @workgroup_size(64)
fn finish_block_offsets(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = linear_index(id);
    if (i >= params.blocks || atomicLoad(&work[0]) != 0u) { return; }
    let prefix = atomicLoad(&work[4u + params.blocks + i]);
    let plain = atomicLoad(&work[4u + params.blocks * 3u + i]);
    let marker_bits = tasks[i].preceding_markers * 16u;
    if (plain > 0xffffffffu - marker_bits || prefix < plain + marker_bits) { fail(12u); return; }
    let padding_start = prefix - plain - marker_bits;
    let padding = atomicLoad(&work[4u + params.blocks * 2u + i]);
    let padding_base = atomicLoad(&padding_state[0]);
    if ((params.has_padding & 1u) != 0u && (padding_base > params.padding_bits
        || padding_start > params.padding_bits - padding_base
        || padding > params.padding_bits - padding_base - padding_start
        || (params.padding_bits + 31u) / 32u > arrayLength(&tables) - 2048u)) { fail(9u); return; }
    if (i + 1u == params.blocks) {
        let count = atomicLoad(&work[4u + params.blocks * 4u + i]);
        if (prefix > params.raw_words * 32u || count > params.raw_words * 32u - prefix) { fail(2u); return; }
        let total = prefix + count;
        if ((total & 7u) != 0u) { fail(12u); return; }
        if ((params.has_padding & 1u) != 0u) {
            let consumed = padding_base + padding_start + padding;
            if ((params.has_padding & 2u) != 0u && consumed != params.padding_bits) { fail(9u); return; }
            atomicStore(&padding_state[1], consumed);
        }
        atomicStore(&work[1], total);
        atomicStore(&work[2], total / 8u);
    }
}
@compute @workgroup_size(64)
fn emit_blocks(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = linear_index(id);
    if (i >= params.blocks || atomicLoad(&work[0]) != 0u) { return; }
    let start = atomicLoad(&work[4u + params.blocks + i]);
    cursor = start; emitting = true;
    encode_block(i);
    if (cursor - start != atomicLoad(&work[4u + i])) { fail(10u); return; }
    let pad_count = atomicLoad(&work[4u + params.blocks * 2u + i]);
    let pad_start = start - atomicLoad(&work[4u + params.blocks * 3u + i]) - tasks[i].preceding_markers * 16u;
    for (var p = 0u; p < pad_count; p++) {
        var bit = 1u;
        if ((params.has_padding & 1u) != 0u) { let at = atomicLoad(&padding_state[0]) + pad_start + p; bit = (tables[2048u + at / 32u] >> (at & 31u)) & 1u; }
        put(1u, bit);
    }
    if (tasks[i].marker_after != 0u) { put(16u, 65280u | tasks[i].marker_after); }
}
fn raw_byte(at: u32) -> u32 { return (atomicLoad(&raw[at >> 2u]) >> (24u - (at & 3u) * 8u)) & 255u; }
fn is_marker(at: u32) -> bool {
    let bit = at * 8u;
    var low = 0u;
    var high = params.blocks;
    while (low < high) {
        let middle = low + (high - low) / 2u;
        if (atomicLoad(&work[4u + params.blocks + middle]) <= bit) { low = middle + 1u; }
        else { high = middle; }
    }
    if (low == 0u) { return false; }
    let i = low - 1u;
    let end = atomicLoad(&work[4u + params.blocks + i]) + atomicLoad(&work[4u + i])
        + atomicLoad(&work[4u + params.blocks * 2u + i]);
    return tasks[i].marker_after != 0u && bit >= end && bit < end + 16u;
}
@compute @workgroup_size(64)
fn count_bytes(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = linear_index(id);
    if (atomicLoad(&work[0]) != 0u || i >= atomicLoad(&work[2])) { return; }
    let count = 1u + select(0u, 1u, raw_byte(i) == 255u && !is_marker(i));
    atomicStore(&work[byte_base() + i], count);
}
@compute @workgroup_size(1)
fn finish_byte_offsets() {
    if (atomicLoad(&work[0]) != 0u) { return; }
    let count = atomicLoad(&work[2]);
    if (count == 0u) { fail(12u); return; }
    let last = count - 1u;
    let prefix = atomicLoad(&work[byte_base() + params.raw_words * 4u + last]);
    let length = atomicLoad(&work[byte_base() + last]);
    if (prefix > params.output_words * 4u || length > params.output_words * 4u - prefix) { fail(11u); return; }
    atomicStore(&work[3], prefix + length);
}
@compute @workgroup_size(64)
fn pack_bytes(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = linear_index(id);
    if (atomicLoad(&work[0]) != 0u || i >= atomicLoad(&work[2])) { return; }
    let at = atomicLoad(&work[byte_base() + params.raw_words * 4u + i]);
    // The output is zero-initialized; a byte-stuffing zero needs no write.
    atomicOr(&output[at >> 2u], raw_byte(i) << ((at & 3u) * 8u));
}

// The host schedules bounded levels from lengths only. It never reads entropy-sized counts.
@compute @workgroup_size(64)
fn scan_groups(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let i = linear_index(id);
    if (group_index(group) >= (scan.count + 63u) / 64u) { return; }
    let lane = local.x;
    var value = 0u;
    if (i < scan.count) { value = atomicLoad(&work[scan.source + i]); }
    scan_values[lane] = value;
    workgroupBarrier();
    for (var distance = 1u; distance < 64u; distance <<= 1u) {
        var add = 0u;
        if (lane >= distance) { add = scan_values[lane - distance]; }
        workgroupBarrier();
        let current = scan_values[lane];
        if (add > 0xffffffffu - current) { fail(12u); }
        scan_values[lane] = current + add;
        workgroupBarrier();
    }
    if (i < scan.count) { atomicStore(&work[scan.destination + i], scan_values[lane] - value); }
    if (lane == 63u) { atomicStore(&work[scan.scratch + group_index(group)], scan_values[63]); }
}
@compute @workgroup_size(64)
fn add_scan_carries(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = linear_index(id);
    if (i >= scan.count) { return; }
    let value = atomicLoad(&work[scan.destination + i]);
    let carry = atomicLoad(&work[scan.source + i / 64u]);
    if (carry > 0xffffffffu - value) { fail(12u); return; }
    atomicStore(&work[scan.destination + i], value + carry);
}

// Commit only after this scan's counting, encoding, stuffing and guard statuses succeed.
@compute @workgroup_size(1)
fn commit_padding_cursor() {
    if (atomicLoad(&work[0]) == 0u && (params.has_padding & 1u) != 0u) {
        atomicStore(&padding_state[0], atomicLoad(&padding_state[1]));
    }
}
