override surface_plane_words: u32;
override surface_color_channels: u32;
override surface_pixels: u32;
override dispatch_width: u32;
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(3) var<storage, read_write> rendered_color: array<u32>;
fn surface_value(word: u32) -> f32 { return bitcast<f32>(source[word]); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let position = id.x + id.y * dispatch_width;
    if position >= surface_pixels { return; }
    var rgb = vec3<u32>(source[position]);
    if surface_color_channels == 3u {
        rgb.y = source[surface_plane_words + position];
        rgb.z = source[2u * surface_plane_words + position];
    }
    let rendered = present_rgb_words(rgb, position);
    // A Gray ICC connection consumes only the first rendered device component.
    rendered_color[position] = rendered.x;
    if surface_color_channels == 3u {
        rendered_color[surface_plane_words + position] = rendered.y;
        rendered_color[2u * surface_plane_words + position] = rendered.z;
    }
}
