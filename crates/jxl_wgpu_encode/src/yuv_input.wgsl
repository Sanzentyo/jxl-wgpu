struct Params {
    width: u32, height: u32, kind: u32, order: u32,
    bits: u32, storage_bytes: u32, big_endian: u32, matrix: u32,
    limited: u32, subsample_x: u32, subsample_y: u32, linear: u32,
    siting_x: f32, siting_y: f32, transfer: u32, reserved: u32,
    planes: array<vec4<u32>, 3>,
}
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn byte_at(address: u32) -> u32 {
    return (source[address >> 2u] >> ((address & 3u) * 8u)) & 255u;
}

fn code_at(address: u32) -> f32 {
    var word = byte_at(address);
    if params.storage_bytes == 2u {
        let second = byte_at(address + 1u);
        word = select(word | (second << 8u), (word << 8u) | second, params.big_endian != 0u);
    }
    return f32(word >> (params.storage_bytes * 8u - params.bits));
}

fn luma(x: u32, y: u32) -> f32 {
    let row = params.planes[0].x + y * params.planes[0].y;
    var code: f32;
    if params.kind == 2u {
        code = code_at(row + (x / 2u) * 4u + (x % 2u) * 2u + params.order);
    } else {
        code = code_at(row + x * params.storage_bytes);
    }
    if params.limited != 0u {
        return (code / f32(1u << (params.bits - 8u)) - 16.0) / 219.0;
    }
    return code / f32((1u << params.bits) - 1u);
}

fn chroma(x: i32, y: i32) -> vec2<f32> {
    let w = (params.width + params.subsample_x - 1u) / params.subsample_x;
    let h = (params.height + params.subsample_y - 1u) / params.subsample_y;
    let cx = u32(clamp(x, 0, i32(w) - 1));
    let cy = u32(clamp(y, 0, i32(h) - 1));
    var code: vec2<f32>;
    if params.kind == 2u {
        let pair = params.planes[0].x + cy * params.planes[0].y + cx * 4u;
        let first = 1u - params.order;
        code = vec2<f32>(code_at(pair + first), code_at(pair + first + 2u));
    } else if params.kind == 1u {
        let pair = params.planes[1].x + cy * params.planes[1].y + cx * 2u * params.storage_bytes;
        code = vec2<f32>(code_at(pair), code_at(pair + params.storage_bytes));
        if params.order != 0u { code = code.yx; }
    } else {
        code = vec2<f32>(
            code_at(params.planes[1].x + cy * params.planes[1].y + cx * params.storage_bytes),
            code_at(params.planes[2].x + cy * params.planes[2].y + cx * params.storage_bytes),
        );
    }
    if params.limited != 0u {
        return (code / f32(1u << (params.bits - 8u)) - vec2<f32>(128.0)) / 224.0;
    }
    return (code - vec2<f32>(f32(1u << (params.bits - 1u)))) / f32((1u << params.bits) - 1u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height { return; }
    // Integer quotient/remainder retains chroma phases even beyond f32's exact integer range.
    let divisor = vec2<u32>(params.subsample_x, params.subsample_y);
    let fraction = (vec2<f32>(id.xy % divisor) - vec2<f32>(params.siting_x, params.siting_y)) / vec2<f32>(divisor);
    let lower = floor(fraction);
    let base = vec2<i32>(id.xy / divisor) + vec2<i32>(lower);
    let weight = fraction - lower;
    let c = mix(
        mix(chroma(base.x, base.y), chroma(base.x + 1, base.y), weight.x),
        mix(chroma(base.x, base.y + 1), chroma(base.x + 1, base.y + 1), weight.x), weight.y,
    );
    var kr = 0.299;
    var kb = 0.114;
    if params.matrix == 1u { kr = 0.2126; kb = 0.0722; }
    if params.matrix >= 2u { kr = 0.2627; kb = 0.0593; }
    let y = luma(id.x, id.y);
    var rgb: vec3<f32>;
    if params.matrix == 3u {
        // BT.2020 constant luminance requires its declared transfer and linear output.
        let b = transfer_to_linear(y + c.x * select(1.9404, 1.5816, c.x > 0.0), 5u, 1.0);
        let r = transfer_to_linear(y + c.y * select(1.7184, 0.9936, c.y > 0.0), 5u, 1.0);
        let ly = transfer_to_linear(y, 5u, 1.0);
        rgb = vec3<f32>(r, (ly - kr * r - kb * b) / (1.0 - kr - kb), b);
    } else {
        let r = y + 2.0 * (1.0 - kr) * c.y;
        let b = y + 2.0 * (1.0 - kb) * c.x;
        rgb = vec3<f32>(r, (y - kr * r - kb * b) / (1.0 - kr - kb), b);
        if params.linear != 0u {
            rgb = vec3<f32>(
                transfer_to_linear(rgb.r, params.transfer, 1.0),
                transfer_to_linear(rgb.g, params.transfer, 1.0),
                transfer_to_linear(rgb.b, params.transfer, 1.0),
            );
        }
    }
    let output = (id.y * params.width + id.x) * 3u;
    destination[output] = rgb.r;
    destination[output + 1u] = rgb.g;
    destination[output + 2u] = rgb.b;
}
