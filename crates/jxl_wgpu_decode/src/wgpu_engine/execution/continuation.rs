//! Resume only as far as the next requested image boundary.

use std::collections::VecDeque;

use super::super::progression::{IntermediateFrame, encode_intermediate};
use super::batches::BatchEncoder;
use super::*;

pub(in super::super) struct DecodeExecution {
    backend: WgpuBackend,
    source: Arc<DecodeSource>,
    pipelines: SubmitPipelines,
    stream: wgpu::Buffer,
    binding: wgpu::BindGroup,
    global_binding: Option<wgpu::BindGroup>,
    stream_upload: Vec<u8>,
    next_global: usize,
    next_group: usize,
    completed_groups: usize,
    images: VecDeque<Arc<IntermediateFrame>>,
    progressive: bool,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::{
        GpuPendingFrame, GpuSubmissionSession, NumericSampleMapping, WgpuSubmissionEngine,
    };
    use jxl_gpu_formats::{Channel, SampleKind};

    #[test]
    fn failed_recording_retires_submitted_windows_before_pool_reuse() {
        let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default()))
        else {
            return;
        };
        let encoded = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/gpu_gray8_lossless.jxl"
        ));
        let parsed = jxl_gpu_bitstream::parse(encoded, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let bytes: Arc<[u8]> = parsed.codestream().into();
        let engine = WgpuSubmissionEngine::new(backend.clone())
            .with_stream_window_limit(NonZeroU64::new(40).unwrap());
        let request = GpuOutputRequest::numeric(
            PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
            NumericSampleMapping::NormalizedGray8,
        )
        .unwrap();
        let prepare = || {
            engine
                .open_with_inventory_data(
                    Arc::new(
                        crate::GpuCodestream::from_shared(
                            Arc::clone(&bytes),
                            0..bytes.len(),
                            false,
                        )
                        .unwrap(),
                    ),
                    &request,
                    &inventory,
                )
                .unwrap()
                .session
        };
        let mut invalid = prepare();
        let source = Arc::get_mut(invalid.source.as_mut().unwrap()).unwrap();
        assert_eq!(source.dispatch_layout.global_streams.batch_count(), 0);
        assert!(source.dispatch_layout.streams.batch_count() > 1);
        let first_end = source.dispatch_layout.streams.batch(0).unwrap().segments()[0].input_end;
        assert!(
            source.dispatch_layout.streams.batch(1).unwrap().segments()[0].input_end > first_end
        );
        // Deliberately violate a prepared-source invariant: the first batch submits, while the
        // following upload fails on the host. Public input validation cannot construct this state.
        source.codestream = Arc::new(
            crate::GpuCodestream::from_shared(Arc::clone(&bytes), 0..first_end, false).unwrap(),
        );
        assert!(invalid.submit_next().is_err());
        let pool = Arc::clone(&invalid.buffers);
        drop(invalid);
        backend
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        // Another Device::poll can return after the worker takes a callback but before that
        // callback drops its leases. Wait for the registered worker job to retire as well.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while backend.submission_poller().in_flight() != 0 && std::time::Instant::now() < deadline {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
        assert_eq!(backend.submission_poller().in_flight(), 0);
        assert_eq!(engine.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(pool.stats().leased_bytes, 0);
        let mut valid = prepare();
        let output = valid.submit_next().unwrap().unwrap().wait().unwrap();
        drop((output, valid));
        assert_eq!(engine.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(pool.stats().leased_bytes, 0);
    }
}

impl DecodeExecution {
    pub(in super::super) fn has_global_stream(&self) -> bool {
        self.source.profile.global_stream.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        backend: WgpuBackend,
        source: Arc<DecodeSource>,
        pipelines: SubmitPipelines,
        stream: wgpu::Buffer,
        binding: wgpu::BindGroup,
        global_binding: Option<wgpu::BindGroup>,
        images: VecDeque<Arc<IntermediateFrame>>,
    ) -> Result<Self> {
        let upload_len = usize::try_from(source.dispatch_layout.stream_bytes)
            .map_err(|_| Error::backend("bounded stream upload exceeds host address space"))?;
        Ok(Self {
            backend,
            source,
            pipelines,
            stream,
            binding,
            global_binding,
            stream_upload: vec![0; upload_len],
            next_global: 0,
            next_group: 0,
            completed_groups: 0,
            progressive: !images.is_empty(),
            images,
        })
    }

    pub(in super::super) fn resume(
        &mut self,
        lifetime: &Arc<DecodeJobLifetime>,
        completion: &Arc<MapCompletion>,
    ) -> Result<Option<Arc<IntermediateFrame>>> {
        let poll = self
            .backend
            .submission_poller()
            .try_reserve()
            .map_err(Error::PollBackpressure)?;
        self.submit_next(lifetime, completion, poll)
    }

    pub(super) fn submit_next(
        &mut self,
        lifetime: &Arc<DecodeJobLifetime>,
        completion: &Arc<MapCompletion>,
        poll: SubmissionPollPermit,
    ) -> Result<Option<Arc<IntermediateFrame>>> {
        let image = self.images.front().cloned();
        let mapping = image.as_ref().map_or(completion, |image| &image.completion);
        let mut last_submission = None;
        let result = self.submit_phase(lifetime, completion, image.as_ref(), &mut last_submission);
        if result.is_err() && last_submission.is_some() {
            // A recording error can occur after an earlier bounded batch was submitted. Keep
            // pooled buffers and byte permits alive until that work retires, even on cancellation.
            // Mapping a copy of the active status buffer also works on WebGPU, whose resources
            // cannot be captured by Queue::on_submitted_work_done's Send-only callback.
            let retained = Arc::clone(lifetime);
            let mut fence =
                self.backend
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("jxl-wgpu failed Modular recording retirement"),
                    });
            fence.copy_buffer_to_buffer(
                lifetime._status.buffer(),
                0,
                lifetime.status_staging.buffer(),
                0,
                self.source.dispatch_layout.status_bytes,
            );
            fence.map_buffer_on_submit(
                lifetime.status_staging.buffer(),
                wgpu::MapMode::Read,
                ..,
                move |result| {
                    if result.is_ok() {
                        retained.status_mapped.store(true, Ordering::Release);
                    }
                    drop(retained);
                },
            );
            last_submission = Some(self.backend.queue().submit([fence.finish()]));
        }
        if let Some(submission) = last_submission {
            let callback = Arc::clone(mapping);
            if let Err(error) =
                poll.register(submission, move |error| callback.complete(Err(error)))
            {
                mapping.complete(Err(format!("GPU poll registration failed: {error}")));
            }
        }
        result?;
        if image.is_some() {
            self.images.pop_front();
        }
        Ok(image)
    }

    fn submit_phase(
        &mut self,
        lifetime: &Arc<DecodeJobLifetime>,
        completion: &Arc<MapCompletion>,
        image: Option<&Arc<IntermediateFrame>>,
        last_submission: &mut Option<wgpu::SubmissionIndex>,
    ) -> Result<()> {
        let source = &self.source;
        let device = self.backend.device();
        let group_end = image.map_or(source.profile.entropy_groups.len(), |image| {
            image.boundary.group_end
        });
        let mut batches = BatchEncoder {
            backend: &self.backend,
            source,
            pipelines: &self.pipelines,
            lifetime,
            stream: &self.stream,
            binding: &self.binding,
            global_binding: self.global_binding.as_ref(),
            stream_upload: &mut self.stream_upload,
        };
        let mut final_recorded = false;
        loop {
            let global = self.next_global < source.dispatch_layout.global_streams.batch_count();
            if !global && self.completed_groups >= group_end {
                break;
            }
            let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu bounded Modular continuation"),
            });
            let mut uniforms = if global {
                batches.global(self.next_global, &mut commands)?;
                self.next_global += 1;
                Vec::new()
            } else {
                let batch = source
                    .dispatch_layout
                    .streams
                    .batch(self.next_group)
                    .ok_or(Error::EngineContract(
                        "Modular continuation lost a group batch",
                    ))?;
                if batch.first_group() + batch.group_count() > group_end {
                    return Err(Error::EngineContract(
                        "Modular batch crosses its image boundary",
                    ));
                }
                let uniforms = batches.group(self.next_group, &mut commands)?;
                self.completed_groups += batch
                    .segments()
                    .iter()
                    .filter(|segment| segment.flags & GroupStreamSegment::FINAL != 0)
                    .count();
                self.next_group += 1;
                uniforms
            };
            if !self.progressive
                && self.next_global == source.dispatch_layout.global_streams.batch_count()
                && self.completed_groups == group_end
            {
                uniforms.extend(encode_frame_completion(
                    device,
                    source,
                    &self.pipelines,
                    lifetime,
                    completion,
                    &mut commands,
                )?);
                final_recorded = true;
            }
            *last_submission = Some(self.backend.queue().submit([commands.finish()]));
            drop(uniforms);
        }
        if !final_recorded {
            let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu Modular image boundary"),
            });
            let uniforms = match image {
                Some(image) => encode_intermediate(
                    device,
                    &mut commands,
                    source,
                    &self.pipelines,
                    lifetime,
                    image,
                )?,
                None => encode_frame_completion(
                    device,
                    source,
                    &self.pipelines,
                    lifetime,
                    completion,
                    &mut commands,
                )?,
            };
            *last_submission = Some(self.backend.queue().submit([commands.finish()]));
            drop(uniforms);
        }
        Ok(())
    }
}
