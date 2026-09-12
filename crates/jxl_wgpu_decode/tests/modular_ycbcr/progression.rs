use super::*;
use std::num::NonZeroUsize;

#[test]
fn subsampled_passes_match_native_prefixes_and_keep_held_images_immutable() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for case in modular_ycbcr::cases()
        .into_iter()
        .filter(|case| case.passes > 1)
    {
        let (bytes, final_reference) = reference(&case.name);
        let snapshots = samples(&format!("{}.progressive", case.name));
        let pixels = (case.size[0] * case.size[1]) as usize;
        assert_eq!(snapshots.len(), (case.passes as usize + 1) * pixels * 4);
        for channel in [None, Some(0), Some(1), Some(2)] {
            let request = if let Some(channel) = channel {
                GpuOutputRequest::numeric(
                    PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                    mapping(case.bit_depth),
                )
                .unwrap()
                .with_color_channel(channel)
                .unwrap()
            } else {
                color_request()
            }
            .with_progressive_output(true)
            .with_max_frame_slots(NonZeroUsize::new(1).unwrap());
            let mut prior = None;
            for limit in [None, NonZeroU64::new(40)] {
                let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                if let Some(limit) = limit {
                    engine = engine.with_stream_window_limit(limit);
                }
                let decoder = GpuDecoder::new(engine);
                let mut session = planes::open_fragmented(&decoder, &bytes, request.clone());
                let mut held = Vec::new();
                let mut outputs = Vec::new();
                while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                    let completed = update
                        .progression()
                        .and_then(|progress| progress.completed_passes())
                        .map_or(case.passes as usize, usize::from);
                    assert_eq!(
                        completed,
                        case.first_output_pass as usize + held.len(),
                        "{} progression",
                        case.name
                    );
                    let expected = &snapshots[completed * pixels * 4..][..pixels * 4];
                    let actual = planes::read(&backend, &update.output().outputs[0]);
                    let expected: Vec<_> = if let Some(channel) = channel {
                        expected
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|p| p[channel as usize])
                            .collect()
                    } else {
                        expected.to_vec()
                    };
                    require_samples(&case.name, &actual, expected.into_iter());
                    outputs.push(actual);
                    held.push(update);
                }
                assert_eq!(
                    held.len(),
                    (case.passes - case.first_output_pass + 1) as usize
                );
                assert!(held.last().unwrap().is_complete());
                if let Some(prior) = &prior {
                    assert_eq!(&outputs, prior);
                }
                prior = Some(outputs.clone());
                drop(session);
                for (update, expected) in held.iter().zip(&outputs) {
                    assert_eq!(
                        planes::read(&backend, &update.output().outputs[0]),
                        *expected
                    );
                }
                drop(held);
                admission::drain(&backend, 0);

                if case.first_output_pass == case.passes {
                    continue;
                }
                // Retaining the first update must not prevent final-only draining or alter it.
                let mut session = planes::open_fragmented(&decoder, &bytes, request.clone());
                let first = session.next_update().unwrap().unwrap();
                let final_image = session.next_frame().unwrap().unwrap();
                let actual = planes::read(&backend, &final_image.output().outputs[0]);
                assert_eq!(actual, *outputs.last().unwrap());
                if channel.is_none() {
                    require_samples(&case.name, &actual, final_reference.iter().copied());
                }
                assert_eq!(
                    planes::read(&backend, &first.output().outputs[0]),
                    outputs[0]
                );
                drop((first, final_image, session));
                admission::drain(&backend, 0);

                // Cancelling the producer after a published pass must preserve the held image.
                let mut session = planes::open_fragmented(&decoder, &bytes, request.clone());
                let first = session.next_update().unwrap().unwrap();
                drop(session);
                assert_eq!(
                    planes::read(&backend, &first.output().outputs[0]),
                    outputs[0]
                );
                drop(first);
                admission::drain(&backend, 0);
            }
        }
    }
}
