use super::{corpus, reject_post_transform_reference};
use jxl_wgpu::WgpuBackend;

#[test]
fn icc_xyb_post_transform_alpha_references_are_invalid_for_every_blend_mode() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let directory = corpus::directory()
        .parent()
        .unwrap()
        .join("embedded_icc_xyb/alpha");
    let mut checked = 0;
    for case in corpus::cases().filter(|case| case.xyb) {
        let base = case.name().strip_suffix("_xyb").unwrap().to_owned();
        for association in ["associated", "straight"] {
            for suffix in ["m0", "m1", "m2", "m2_alpha_ref1", "m3", "m4"] {
                let name = format!("{base}_{association}_{suffix}");
                let data = std::fs::read(directory.join(format!("{name}.jxl"))).unwrap();
                reject_post_transform_reference(&backend, &data, &name);
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 48);
}
