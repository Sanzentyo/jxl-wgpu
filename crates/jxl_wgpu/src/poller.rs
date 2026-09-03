//! Bounded native driver for runtime-neutral GPU completion futures.

use std::fmt;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Maximum number of native submissions admitted to the shared poll worker.
///
/// The bound includes permits which have not yet been attached to a submission, requests queued
/// behind the worker, and the request currently being polled. Callers must reserve a permit before
/// submitting GPU work, so saturation is reported without leaving unpolled work behind.
pub const SUBMISSION_POLLER_CAPACITY: usize = 256;

/// Failure to reserve or register a native submission with the bounded poll worker.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
pub enum SubmissionPollerError {
    #[error("GPU poll admission is full (capacity {capacity})")]
    Full { capacity: usize },
    #[error("GPU poll worker has stopped")]
    Stopped,
}

#[cfg(not(target_arch = "wasm32"))]
type ErrorCallback = Box<dyn FnOnce(String) + Send + 'static>;

#[cfg(not(target_arch = "wasm32"))]
struct PollRequest {
    submission: wgpu::SubmissionIndex,
    on_error: ErrorCallback,
    _slot: PollSlot,
}

#[cfg(not(target_arch = "wasm32"))]
struct AdmissionState {
    in_flight: AtomicUsize,
    total_admitted: AtomicU64,
    worker_running: AtomicBool,
}

#[cfg(not(target_arch = "wasm32"))]
impl AdmissionState {
    fn new() -> Self {
        Self {
            in_flight: AtomicUsize::new(0),
            total_admitted: AtomicU64::new(0),
            worker_running: AtomicBool::new(true),
        }
    }

    fn try_acquire(self: &Arc<Self>) -> Result<PollSlot, SubmissionPollerError> {
        if !self.worker_running.load(Ordering::Acquire) {
            return Err(SubmissionPollerError::Stopped);
        }
        self.in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < SUBMISSION_POLLER_CAPACITY).then_some(current + 1)
            })
            .map_err(|_| SubmissionPollerError::Full {
                capacity: SUBMISSION_POLLER_CAPACITY,
            })?;

        self.total_admitted.fetch_add(1, Ordering::Relaxed);
        let slot = PollSlot {
            admission: Arc::clone(self),
        };
        if self.worker_running.load(Ordering::Acquire) {
            Ok(slot)
        } else {
            drop(slot);
            Err(SubmissionPollerError::Stopped)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct PollSlot {
    admission: Arc<AdmissionState>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for PollSlot {
    fn drop(&mut self) {
        let previous = self.admission.in_flight.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "submission poll admission underflow");
    }
}

/// Reservation for exactly one native submission-poll job.
///
/// Acquire it with [`SubmissionPoller::try_reserve`] before calling `Queue::submit`, then consume
/// it with [`Self::register`]. Dropping an unused permit releases its slot. A registered request
/// owns the slot until polling succeeds, fails, or the worker drops the request.
#[must_use = "a poll permit must be registered before the corresponding GPU submission"]
pub struct SubmissionPollPermit {
    #[cfg(not(target_arch = "wasm32"))]
    sender: std::sync::mpsc::SyncSender<PollRequest>,
    #[cfg(not(target_arch = "wasm32"))]
    slot: Option<PollSlot>,
}

impl fmt::Debug for SubmissionPollPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SubmissionPollPermit")
            .field("capacity", &SUBMISSION_POLLER_CAPACITY)
            .finish_non_exhaustive()
    }
}

impl SubmissionPollPermit {
    /// Attaches this pre-reserved slot to a submitted queue index.
    ///
    /// The callback is invoked only if `Device::poll` fails. Successful map/work-done callbacks
    /// are dispatched by wgpu while the shared worker polls. This operation is non-blocking. Once
    /// a permit has been reserved, the bounded queue has room for its request; `Stopped` remains
    /// possible if the worker terminates unexpectedly between reservation and registration.
    pub fn register<F>(
        self,
        submission: wgpu::SubmissionIndex,
        on_error: F,
    ) -> Result<(), SubmissionPollerError>
    where
        F: FnOnce(String) + Send + 'static,
    {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut permit = self;
            let request = PollRequest {
                submission,
                on_error: Box::new(on_error),
                _slot: permit
                    .slot
                    .take()
                    .expect("a submission poll permit can only be registered once"),
            };
            permit
                .sender
                .try_send(request)
                .map_err(|error| match error {
                    // Admission counts unused permits, queued requests, and the active request. With
                    // all registration going through `try_reserve`, a full channel here is impossible:
                    // this request itself already occupies one of the bounded admission slots.
                    std::sync::mpsc::TrySendError::Full(_) => SubmissionPollerError::Full {
                        capacity: SUBMISSION_POLLER_CAPACITY,
                    },
                    std::sync::mpsc::TrySendError::Disconnected(_) => {
                        SubmissionPollerError::Stopped
                    }
                })
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (self, submission, on_error);
            Ok(())
        }
    }
}

/// One bounded native `Device::poll` worker shared by all clones of a backend.
///
/// Browser WebGPU invokes callbacks from its event loop, so this type and its permits are
/// zero-sized and registration is a no-op on `wasm32`. Native registration never spawns a
/// per-submission thread.
#[derive(Clone)]
pub struct SubmissionPoller {
    #[cfg(not(target_arch = "wasm32"))]
    sender: std::sync::mpsc::SyncSender<PollRequest>,
    #[cfg(not(target_arch = "wasm32"))]
    admission: Arc<AdmissionState>,
}

impl fmt::Debug for SubmissionPoller {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SubmissionPoller")
            .field("capacity", &SUBMISSION_POLLER_CAPACITY)
            .field("in_flight", &self.in_flight())
            .finish_non_exhaustive()
    }
}

impl SubmissionPoller {
    /// Starts one native poll worker for `device`.
    ///
    /// # Errors
    ///
    /// Returns the operating-system thread creation error on native targets. Browser WebGPU does
    /// not create a thread and therefore always succeeds.
    pub fn new(device: wgpu::Device) -> std::io::Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (sender, receiver) =
                std::sync::mpsc::sync_channel::<PollRequest>(SUBMISSION_POLLER_CAPACITY);
            let admission = Arc::new(AdmissionState::new());
            let worker_admission = Arc::clone(&admission);
            std::thread::Builder::new()
                .name("jxl-wgpu-poll".into())
                .spawn(move || {
                    let _worker_guard = WorkerGuard(worker_admission);
                    native_poll_loop(device, &receiver);
                })?;
            Ok(Self { sender, admission })
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = device;
            Ok(Self {})
        }
    }

    /// Tries to reserve capacity for one submission before GPU work is submitted.
    ///
    /// This call never blocks. On native targets, at most [`SUBMISSION_POLLER_CAPACITY`] permits,
    /// queued requests, and active requests exist in aggregate. Browser callbacks need no native
    /// poll job, so reservation always succeeds on `wasm32`.
    pub fn try_reserve(&self) -> Result<SubmissionPollPermit, SubmissionPollerError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let slot = self.admission.try_acquire()?;
            Ok(SubmissionPollPermit {
                sender: self.sender.clone(),
                slot: Some(slot),
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            Ok(SubmissionPollPermit {})
        }
    }

    /// Number of permits, queued requests, and active requests currently admitted on native.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.admission.in_flight.load(Ordering::Acquire)
        }
        #[cfg(target_arch = "wasm32")]
        {
            0
        }
    }

    /// Cumulative count of submissions admitted across all permits.
    #[must_use]
    pub fn submission_count(&self) -> u64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.admission.total_admitted.load(Ordering::Acquire)
        }
        #[cfg(target_arch = "wasm32")]
        {
            0
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct WorkerGuard(Arc<AdmissionState>);

#[cfg(not(target_arch = "wasm32"))]
impl Drop for WorkerGuard {
    fn drop(&mut self) {
        self.0.worker_running.store(false, Ordering::Release);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_poll_loop(device: wgpu::Device, receiver: &std::sync::mpsc::Receiver<PollRequest>) {
    while let Ok(request) = receiver.recv() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.poll(wgpu::PollType::Wait {
                submission_index: Some(request.submission.clone()),
                timeout: None,
            })
        }));
        let error = match result {
            Ok(Ok(_)) => continue,
            Ok(Err(error)) => format!("GPU polling failed: {error}"),
            Err(_) => "GPU polling callback panicked".to_string(),
        };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (request.on_error)(error);
        }));
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn admission_is_exactly_bounded_and_abandoned_slots_are_reusable() {
        let admission = Arc::new(AdmissionState::new());
        let mut slots = (0..SUBMISSION_POLLER_CAPACITY)
            .map(|_| admission.try_acquire().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            admission.in_flight.load(Ordering::Acquire),
            SUBMISSION_POLLER_CAPACITY
        );
        assert!(matches!(
            admission.try_acquire(),
            Err(SubmissionPollerError::Full {
                capacity: SUBMISSION_POLLER_CAPACITY
            })
        ));

        drop(slots.pop());
        let replacement = admission.try_acquire().unwrap();
        assert_eq!(
            admission.in_flight.load(Ordering::Acquire),
            SUBMISSION_POLLER_CAPACITY
        );
        drop(replacement);
        drop(slots);
        assert_eq!(admission.in_flight.load(Ordering::Acquire), 0);
    }

    #[test]
    fn stopped_worker_rejects_new_admission_and_releases_existing_slots() {
        let admission = Arc::new(AdmissionState::new());
        let slot = admission.try_acquire().unwrap();
        admission.worker_running.store(false, Ordering::Release);
        assert!(matches!(
            admission.try_acquire(),
            Err(SubmissionPollerError::Stopped)
        ));
        drop(slot);
        assert_eq!(admission.in_flight.load(Ordering::Acquire), 0);
    }

    #[test]
    fn every_pre_reserved_job_registers_and_releases_its_slot() {
        let backend = match pollster::block_on(crate::WgpuBackend::request_default(
            crate::WgpuBackendConfig {
                enable_timestamps: false,
                ..crate::WgpuBackendConfig::default()
            },
        )) {
            Ok(backend) => backend,
            Err(crate::Error::NoAdapter) => return,
            Err(error) => panic!("failed to create test adapter: {error}"),
        };
        let poller = SubmissionPoller::new(backend.device().clone()).unwrap();
        let permits = (0..SUBMISSION_POLLER_CAPACITY)
            .map(|_| poller.try_reserve().unwrap())
            .collect::<Vec<_>>();
        assert!(matches!(
            poller.try_reserve(),
            Err(SubmissionPollerError::Full {
                capacity: SUBMISSION_POLLER_CAPACITY
            })
        ));

        let mut last_submission = None;
        for permit in permits {
            let commands =
                backend
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("submission poll admission test"),
                    });
            let submission = backend.queue().submit([commands.finish()]);
            permit.register(submission.clone(), |_| {}).unwrap();
            last_submission = Some(submission);
        }

        backend
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: last_submission,
                timeout: None,
            })
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while poller.in_flight() != 0 && std::time::Instant::now() < deadline {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
        assert_eq!(poller.in_flight(), 0);
    }
}
