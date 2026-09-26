use super::*;
use std::sync::Weak;
use std::task::Wake;

struct WakeProbe {
    lifetime: Weak<EncodeJobLifetime>,
    context: WgpuContext,
    pool: Arc<EncoderBufferPool>,
    observations: Mutex<Vec<(usize, u64, u64)>>,
}

impl Wake for WakeProbe {
    fn wake(self: Arc<Self>) {
        self.observations.lock().unwrap().push((
            self.lifetime.strong_count(),
            self.context.memory_stats().reserved_bytes,
            self.pool.stats().leased_buffer_sets,
        ));
    }
}

#[test]
fn map_notification_releases_callback_ownership_before_waking_the_consumer() {
    let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        .expect("actual GPU required for mapping lifetime evidence");
    let context = WgpuContext::from_backend(&backend);
    let pool = EncoderBufferPool::new(0);
    const BYTES: u64 = 768;
    for retained in [false, true] {
        for success in [false, true] {
            let lease = pool.checkout(context.device(), 256, 256, false);
            let readback = Arc::clone(&lease.buffers().readback);
            let lifetime = Arc::new(EncodeJobLifetime {
                buffer_lease: lease,
                _source_buffers: Vec::new(),
                _memory_permit: context.memory_budget().try_reserve(BYTES).unwrap(),
                mapped: AtomicBool::new(false),
            });
            let job_lifetime = retained.then(|| Arc::clone(&lifetime));
            let probe = Arc::new(WakeProbe {
                lifetime: Arc::downgrade(&lifetime),
                context: context.clone(),
                pool: Arc::clone(&pool),
                observations: Mutex::new(Vec::new()),
            });
            let waker = Waker::from(Arc::clone(&probe));
            let cx = Context::from_waker(&waker);
            let completion = Arc::new(MapCompletion::default());
            assert!(completion.poll(&cx).is_none());
            if success {
                let callback = Arc::clone(&completion);
                readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        callback.complete_mapping(
                            lifetime,
                            result.map_err(BackendError::ArtifactMapping),
                        );
                    });
                context
                    .device()
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
            } else {
                completion
                    .complete_mapping(lifetime, Err(BackendError::Invariant("test map failure")));
            }
            // The probe runs inside wake(), before the callback returns. This catches the
            // notification/release race deterministically without a sleep or retry loop.
            assert_eq!(
                *probe.observations.lock().unwrap(),
                [(
                    usize::from(retained),
                    if retained { BYTES } else { 0 },
                    u64::from(retained)
                )]
            );
            assert_eq!(completion.poll(&cx).unwrap().is_ok(), success);
            if let Some(job) = job_lifetime {
                if success {
                    assert!(job.mapped.load(Ordering::Acquire));
                    assert_eq!(readback.slice(..).get_mapped_range().unwrap().len(), 256);
                }
                drop(job);
            }
            assert_eq!(context.memory_stats().reserved_bytes, 0);
            assert_eq!(pool.stats().leased_buffer_sets, 0);
        }
    }
}
