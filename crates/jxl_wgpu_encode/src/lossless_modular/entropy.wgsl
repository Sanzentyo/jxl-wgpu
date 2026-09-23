@group(0) @binding(0) var<storage, read_write> words: array<u32>;
@group(0) @binding(1) var<storage, read> metadata: array<u32>;

var<private> cursor: u32;
var<private> output: u32;
var<private> state: u32;
var<private> failed: bool;
var<private> symbols: u32;

fn prepend(bits: u32, count: u32) {
    if failed || count == 0u { return; }
    if count > 32u || cursor < count { failed = true; return; }
    cursor -= count;
    let index = output + cursor / 32u;
    let shift = cursor % 32u;
    words[index] |= bits << shift;
    if shift + count > 32u { words[index + 1u] |= bits >> (32u - shift); }
}

fn put_symbol(histogram: u32, symbol: u32) {
    if failed || symbol >= 256u { failed = true; return; }
    let table = metadata[1u] + histogram * 4608u;
    let frequency = metadata[table + symbol];
    if frequency == 0u || frequency > 4096u { failed = true; return; }
    if (state >> 20u) >= frequency {
        prepend(state & 65535u, 16u);
        state >>= 16u;
    }
    let rank = state % frequency;
    let mapped = metadata[table + 512u + metadata[table + 256u + symbol] + rank];
    state = (state / frequency) * 4096u + mapped;
    symbols += 1u;
}

@compute @workgroup_size(1)
fn encode(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= metadata[0u] { return; }
    let job = 4u + id.x * 4u;
    let channels = metadata[job];
    let count = metadata[job + 1u];
    let header = metadata[job + 2u];
    let capacity = metadata[job + 3u] * 32u;
    output = header + 4u;
    cursor = capacity;
    state = 0x130000u;
    failed = false;
    symbols = 0u;
    for (var channel = count; channel > 0u; channel -= 1u) {
        let descriptor = channels + (channel - 1u) * 4u;
        let events = metadata[descriptor];
        let length = words[metadata[descriptor + 1u]];
        let limit = metadata[descriptor + 2u];
        let histogram = metadata[descriptor + 3u];
        if length > limit { failed = true; break; }
        for (var event = length; event > 0u; event -= 1u) {
            let base = events + (event - 1u) * 4u;
            let kind = words[base];
            let token = words[base + 1u];
            let extra_count = words[base + 2u];
            let extra = words[base + 3u];
            if extra_count > 31u || (extra_count < 32u && (extra >> extra_count) != 0u) {
                failed = true; break;
            }
            if kind == 1u && metadata[2u] == 0u {
                put_symbol(0u, 1u);
                prepend(extra, extra_count);
                put_symbol(histogram, 224u + token);
                put_symbol(histogram, 0u);
            } else if kind == 0u || (kind == 2u && metadata[2u] == 1u) || (kind == 3u && metadata[2u] == 1u) {
                prepend(extra, extra_count);
                put_symbol(select(histogram, 0u, kind == 3u), token + select(0u, 224u, kind == 2u));
            } else { failed = true; break; }
        }
        if failed { break; }
    }
    prepend(state, 32u);
    if failed { words[header] = 2u; return; }
    let bit_count = capacity - cursor;
    let shift = cursor % 32u;
    let source = output + cursor / 32u;
    let length = (bit_count + 31u) / 32u;
    for (var index = 0u; index < length; index += 1u) {
        var value = words[source + index] >> shift;
        if shift != 0u && (source + index + 1u) < (output + capacity / 32u) {
            value |= words[source + index + 1u] << (32u - shift);
        }
        if index == length - 1u && bit_count % 32u != 0u { value &= (1u << (bit_count % 32u)) - 1u; }
        words[output + index] = value;
    }
    words[header + 1u] = bit_count;
    words[header + 2u] = symbols;
    words[header + 3u] = 0u;
    words[header] = 1u;
}
