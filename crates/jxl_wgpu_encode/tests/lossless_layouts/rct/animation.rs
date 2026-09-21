use super::*;

#[test]
fn rct_animations_keep_original_words_and_reference_composition() {
    let rig = Rig::new();
    for local in [false, true] {
        for operation in 0..7 {
            let value = operation + 7 * ((operation + u32::from(local)) % 6);
            let config = config(value, local, operation as usize % 2);
            let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config);
            let kind = if operation % 2 == 0 {
                SampleKind::Float
            } else {
                SampleKind::Unsigned
            };
            groups::animation::check_animation_words(
                &rig,
                &encoder,
                config.group_size,
                LosslessModularFormat::Rgba,
                kind,
                if kind == SampleKind::Float { 32 } else { 31 },
            );
            groups::animation::check_cropped_frames(&rig, &encoder, config.group_size);
        }
    }
}
