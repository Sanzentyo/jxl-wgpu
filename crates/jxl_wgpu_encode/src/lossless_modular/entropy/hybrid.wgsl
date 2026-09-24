// Shared by profiling and serialization. The checked host plan admits only
// configurations whose complete u32 alphabet fits below the LZ77 threshold.
struct HybridToken { token: u32, count: u32, bits: u32 }

fn hybrid_uint(value: u32, config: u32) -> HybridToken {
    let split = config & 255u;
    let msb = (config >> 8u) & 255u;
    let lsb = config >> 16u;
    let threshold = 1u << split;
    if value < threshold { return HybridToken(value, 0u, 0u); }
    let n = 31u - countLeadingZeros(value);
    let mantissa = value - (1u << n);
    let count = n - msb - lsb;
    let token = threshold + ((n - split) << (msb + lsb))
        + ((mantissa >> (n - msb)) << lsb) + (mantissa & ((1u << lsb) - 1u));
    return HybridToken(token, count, (value >> lsb) & ((1u << count) - 1u));
}

fn canonical_value(token: u32, extra_count: u32, extra: u32) -> u32 {
    return select((1u << extra_count) | extra, 0u, token == 0u);
}

fn canonical_valid(token: u32, extra_count: u32, extra: u32) -> bool {
    return token <= 32u && extra_count == select(token - 1u, 0u, token == 0u)
        && extra_count <= 31u && (extra >> extra_count) == 0u;
}
