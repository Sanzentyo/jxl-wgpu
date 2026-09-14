use super::*;
use jxl_gpu_protocol::icc::IccStage;
use jxl_wgpu::ResidentIccSampleEncoding::{Complement, Direct};

#[test]
fn complemented_image_samples_leave_icc_black_metadata_and_buffer_guards_unchanged() {
    let backend = backend().expect("CMYK sample conformance requires a GPU adapter");
    let identity = profile("mpe/identity");
    let source = profile("black/lut16_lab_4");
    let target = profile("lut/lut16_lab_4");
    let transforms = [
        IccTransform::new(&source, &identity, IccRenderingIntent::Perceptual).unwrap(),
        IccTransform::new(&identity, &target, IccRenderingIntent::Relative).unwrap(),
    ];
    assert!(
        transforms[0]
            .program()
            .stages()
            .iter()
            .any(|stage| matches!(stage, IccStage::BlackPointConnection(_)))
    );
    let extent = Extent2d::new(17, 9);
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Tile16x16,
    ] {
        let pipeline = ResidentIccPipeline::with_variant(backend.device(), variant).unwrap();
        for transform in &transforms {
            let program = ResidentIccProgram::new(backend.device(), transform).unwrap();
            // Dyadic values make the storage round trip exact; the existing independent
            // LUT corpora separately check the numerical color program itself.
            let input: Vec<_> = (0..extent.area().unwrap() * program.input_channels())
                .map(|i| ((i * 7) % 17) as f32 / 16.0)
                .collect();
            let expected = run_program(&backend, &pipeline, &program, extent, &input, 5);
            for load in [Direct, Complement] {
                for store in [Direct, Complement] {
                    let input: Vec<_> = input
                        .iter()
                        .map(|&v| if load == Complement { 1.0 - v } else { v })
                        .collect();
                    let actual = run_encoded_program(
                        &backend,
                        &pipeline,
                        &program,
                        extent,
                        &input,
                        7,
                        [load, store],
                    );
                    for (actual, &expected) in actual.iter().zip(&expected) {
                        let expected = if store == Complement {
                            1.0 - expected
                        } else {
                            expected
                        };
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{variant:?} load {load:?} store {store:?}"
                        );
                    }
                }
            }
        }
    }
}
