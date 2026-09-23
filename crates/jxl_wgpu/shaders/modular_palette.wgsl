// Shared normative implicit entries; nonnegative indices are relative to the explicit table.
const DELTA_PALETTE: array<vec3<i32>, 72> = array<vec3<i32>, 72>(
    vec3<i32>(0, 0, 0), vec3<i32>(4, 4, 4), vec3<i32>(11, 0, 0),
    vec3<i32>(0, 0, -13), vec3<i32>(0, -12, 0), vec3<i32>(-10, -10, -10),
    vec3<i32>(-18, -18, -18), vec3<i32>(-27, -27, -27), vec3<i32>(-18, -18, 0),
    vec3<i32>(0, 0, -32), vec3<i32>(-32, 0, 0), vec3<i32>(-37, -37, -37),
    vec3<i32>(0, -32, -32), vec3<i32>(24, 24, 45), vec3<i32>(50, 50, 50),
    vec3<i32>(-45, -24, -24), vec3<i32>(-24, -45, -45), vec3<i32>(0, -24, -24),
    vec3<i32>(-34, -34, 0), vec3<i32>(-24, 0, -24), vec3<i32>(-45, -45, -24),
    vec3<i32>(64, 64, 64), vec3<i32>(-32, 0, -32), vec3<i32>(0, -32, 0),
    vec3<i32>(-32, 0, 32), vec3<i32>(-24, -45, -24), vec3<i32>(45, 24, 45),
    vec3<i32>(24, -24, -45), vec3<i32>(-45, -24, 24), vec3<i32>(80, 80, 80),
    vec3<i32>(64, 0, 0), vec3<i32>(0, 0, -64), vec3<i32>(0, -64, -64),
    vec3<i32>(-24, -24, 45), vec3<i32>(96, 96, 96), vec3<i32>(64, 64, 0),
    vec3<i32>(45, -24, -24), vec3<i32>(34, -34, 0), vec3<i32>(112, 112, 112),
    vec3<i32>(24, -45, -45), vec3<i32>(45, 45, -24), vec3<i32>(0, -32, 32),
    vec3<i32>(24, -24, 45), vec3<i32>(0, 96, 96), vec3<i32>(45, -24, 24),
    vec3<i32>(24, -45, -24), vec3<i32>(-24, -45, 24), vec3<i32>(0, -64, 0),
    vec3<i32>(96, 0, 0), vec3<i32>(128, 128, 128), vec3<i32>(64, 0, 64),
    vec3<i32>(144, 144, 144), vec3<i32>(96, 96, 0), vec3<i32>(-36, -36, 36),
    vec3<i32>(45, -24, -45), vec3<i32>(45, -45, -24), vec3<i32>(0, 0, -96),
    vec3<i32>(0, 128, 128), vec3<i32>(0, 96, 0), vec3<i32>(45, 24, -45),
    vec3<i32>(-128, 0, 0), vec3<i32>(24, -45, 24), vec3<i32>(-45, 24, -45),
    vec3<i32>(64, 0, -64), vec3<i32>(64, -64, -64), vec3<i32>(96, 0, 96),
    vec3<i32>(45, -45, 24), vec3<i32>(24, 45, -45), vec3<i32>(64, 64, -64),
    vec3<i32>(128, 128, 0), vec3<i32>(0, 0, -128), vec3<i32>(-24, 45, -45),
);

fn mp_implicit_palette_value(index: i32, channel: u32, bit_depth: u32) -> i32 {
    if index < 0i {
        if channel >= 3u {
            return 0i;
        }
        let normalized = (0u - (bitcast<u32>(index) + 1u)) % 143u;
        var value = DELTA_PALETTE[(normalized + 1u) >> 1u][channel];
        if (normalized & 1u) == 0u {
            value = bitcast<i32>(0u - bitcast<u32>(value));
        }
        if bit_depth > 8u {
            value = bitcast<i32>(bitcast<u32>(value) << min(bit_depth, 24u) - 8u);
        }
        return value;
    }
    if channel >= 3u {
        return 0i;
    }
    let maximum = select((1u << bit_depth) - 1u, 0xffffffffu, bit_depth == 32u);
    var implicit_index = u32(index);
    if implicit_index < 64u {
        let digit = (implicit_index >> (2u * channel)) % 4u;
        let scaled = mi_shr(mi_mul_u32(vec2<u32>(maximum, 0u), digit), 2u).x;
        return bitcast<i32>(scaled + (1u << (max(bit_depth, 3u) - 3u)));
    }
    implicit_index -= 64u;
    if channel == 1u {
        implicit_index /= 5u;
    } else if channel == 2u {
        implicit_index /= 25u;
    }
    return bitcast<i32>(mi_shr(
        mi_mul_u32(vec2<u32>(maximum, 0u), implicit_index % 5u), 2u).x);
}
