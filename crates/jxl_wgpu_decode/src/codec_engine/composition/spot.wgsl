struct SpotColor {
    source: vec4<u32>, // absolute plane word offset, reserved
    rgba: vec4<f32>, // declared ink RGB and solidity
};
@group(0) @binding(7) var<storage, read> spot_colors: array<SpotColor>;

fn present_rgb(rgb: vec3<f32>, position: u32) -> vec3<f32> {
    var result = rgb;
    for (var ink = 0u; ink < arrayLength(&spot_colors); ink++) {
        let spot = spot_colors[ink];
        let coverage = spot.rgba.w * surface_value(spot.source.x + position);
        result = coverage * spot.rgba.xyz + (1.0 - coverage) * result;
    }
    return result;
}
