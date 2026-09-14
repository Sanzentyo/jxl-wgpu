use super::{corpus, reject_post_transform_reference};
use jxl_wgpu::WgpuBackend;

#[test]
fn icc_xyb_post_transform_animation_references_are_invalid() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let directory = corpus::directory()
        .parent()
        .unwrap()
        .join("embedded_icc_xyb/animation");
    for case in corpus::cases().filter(|case| case.xyb) {
        let name = case.name().strip_suffix("_xyb").unwrap().to_owned();
        let data = std::fs::read(directory.join(format!("{name}.jxl"))).unwrap();
        reject_post_transform_reference(&backend, &data, &name);
    }
}
