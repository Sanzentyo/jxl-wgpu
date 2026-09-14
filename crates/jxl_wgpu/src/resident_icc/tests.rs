use super::*;

#[test]
fn profile_binding_capabilities_fail_before_pipeline_creation() {
    for (limits, resource) in [
        (
            wgpu::Limits {
                max_storage_buffers_per_shader_stage: 2,
                ..Default::default()
            },
            "storage bindings",
        ),
        (
            wgpu::Limits {
                max_uniform_buffers_per_shader_stage: 0,
                ..Default::default()
            },
            "uniform bindings",
        ),
        (
            wgpu::Limits {
                max_bind_groups: 0,
                ..Default::default()
            },
            "bind groups",
        ),
        (
            wgpu::Limits {
                max_bindings_per_bind_group: 3,
                ..Default::default()
            },
            "binding slots",
        ),
        (
            wgpu::Limits {
                max_uniform_buffer_binding_size: 271,
                ..Default::default()
            },
            "uniform binding bytes",
        ),
    ] {
        assert!(
            matches!(validate_capabilities(&limits), Err(ResidentIccError::Limit { resource: actual, .. }) if actual == resource)
        );
    }
}

#[test]
fn icc_shader_and_dispatch_abi_are_webgpu_portable() {
    let module = naga::front::wgsl::parse_str(include_str!("../../shaders/icc.wgsl")).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let (_, params) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("Params"))
        .unwrap();
    let naga::TypeInner::Struct { members, span } = &params.inner else {
        panic!("uniform is not a struct");
    };
    assert_eq!(*span, std::mem::size_of::<DispatchParams>() as u32);
    assert_eq!(
        members
            .iter()
            .map(|member| member.offset)
            .collect::<Vec<_>>(),
        vec![0, 16, 80, 144, 208]
    );
}
