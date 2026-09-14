use super::*;

#[cfg(not(target_arch = "wasm32"))]
mod lifetime;

#[test]
fn gain_map_shader_has_the_exact_uniform_layout_and_portable_bindings() {
    let shader = format!(
        "{}\n{}",
        jxl_wgpu::IMAGE_OUTPUT_SHADER,
        include_str!("../render.wgsl")
    );
    let module = naga::front::wgsl::parse_str(&shader).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(naga::proc::GlobalCtx {
            types: &module.types,
            constants: &module.constants,
            overrides: &module.overrides,
            global_expressions: &module.global_expressions,
        })
        .unwrap();
    for (_, variable) in module.global_variables.iter() {
        if variable.name.as_deref() == Some("gain_params") {
            assert_eq!(layouter[variable.ty].size as usize, size_of::<Params>());
            let naga::TypeInner::Struct { members, .. } = &module.types[variable.ty].inner else {
                panic!()
            };
            assert_eq!(
                members.iter().map(|m| m.offset).collect::<Vec<_>>(),
                [0, 16, 32, 80, 96, 112, 128, 144]
            );
        }
    }
    assert_eq!(size_of::<Params>(), 160);
}
