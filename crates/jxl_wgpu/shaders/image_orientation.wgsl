// Zero-based Exif orientation codes, matching OutputOrientation::to_exif_value() - 1.
// Coordinates must be inside the nonempty source/output extent before calling these helpers.
fn image_output_coordinate(source: vec2<u32>, extent: vec2<u32>, orientation: u32) -> vec2<u32> {
    let x = source.x;
    let y = source.y;
    switch orientation {
        case 1u: { return vec2<u32>(extent.x - 1u - x, y); }
        case 2u: { return extent - vec2<u32>(1u) - source; }
        case 3u: { return vec2<u32>(x, extent.y - 1u - y); }
        case 4u: { return vec2<u32>(y, x); }
        case 5u: { return vec2<u32>(extent.y - 1u - y, x); }
        case 6u: { return vec2<u32>(extent.y - 1u - y, extent.x - 1u - x); }
        case 7u: { return vec2<u32>(y, extent.x - 1u - x); }
        default: { return source; }
    }
}

fn image_source_coordinate(destination: vec2<u32>, extent: vec2<u32>, orientation: u32) -> vec2<u32> {
    let x = destination.x;
    let y = destination.y;
    switch orientation {
        case 1u: { return vec2<u32>(extent.x - 1u - x, y); }
        case 2u: { return extent - vec2<u32>(1u) - destination; }
        case 3u: { return vec2<u32>(x, extent.y - 1u - y); }
        case 4u: { return vec2<u32>(y, x); }
        case 5u: { return vec2<u32>(y, extent.y - 1u - x); }
        case 6u: { return vec2<u32>(extent.x - 1u - y, extent.y - 1u - x); }
        case 7u: { return vec2<u32>(extent.x - 1u - y, x); }
        default: { return destination; }
    }
}
