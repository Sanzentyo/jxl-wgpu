use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

/// Default upper bound for idle Modular8 encoder buffers retained for reuse.
pub const DEFAULT_ENCODER_BUFFER_POOL_BYTES: u64 = 32 * 1024 * 1024;

/// Secondary object-count bound for tiny-buffer workloads.
pub const MAX_ENCODER_BUFFER_POOL_IDLE_SETS: usize = 256;

/// Point-in-time counters for the encoder-owned GPU buffer pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderBufferPoolStats {
    /// Maximum bytes that may be retained while idle. Zero disables retention.
    pub limit_bytes: u64,
    /// Bytes held by all currently idle buffer sets.
    pub idle_bytes: u64,
    /// Complete parameter/artifact/readback sets ready for reuse.
    pub idle_buffer_sets: u64,
    /// Hard object-count bound applied in addition to `limit_bytes`.
    pub max_idle_buffer_sets: u64,
    /// Individual buffers ready for reuse. Each set contains three buffers.
    pub idle_buffers: u64,
    /// Sets checked out by live GPU submissions.
    pub leased_buffer_sets: u64,
    /// Checkouts satisfied without creating any GPU buffers.
    pub reuse_hits: u64,
    /// Checkouts that created a new three-buffer set.
    pub allocation_misses: u64,
    /// Complete sets discarded because of the byte limit or an explicit clear.
    pub evicted_buffer_sets: u64,
    /// Individual buffers discarded by eviction.
    pub evicted_buffers: u64,
    /// Total allocation bytes discarded by eviction.
    pub evicted_bytes: u64,
}

#[derive(Debug)]
pub(crate) struct EncoderBufferSet {
    pub(crate) parameters: Arc<wgpu::Buffer>,
    pub(crate) artifact: Arc<wgpu::Buffer>,
    pub(crate) readback: Arc<wgpu::Buffer>,
    parameter_bytes: u64,
    artifact_bytes: u64,
    allocation_bytes: u64,
}

impl EncoderBufferSet {
    fn new(device: &wgpu::Device, parameter_bytes: u64, artifact_bytes: u64) -> Self {
        let allocation_bytes = parameter_bytes
            .checked_add(
                artifact_bytes
                    .checked_mul(2)
                    .expect("checked dispatch plan bounds two artifact buffers"),
            )
            .expect("checked dispatch plan bounds total pooled bytes");
        Self {
            parameters: Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jxl-wgpu lossless modular8 pooled group parameters"),
                size: parameter_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            artifact: Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jxl-wgpu lossless modular8 pooled GPU artifacts"),
                size: artifact_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            readback: Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jxl-wgpu lossless modular8 pooled artifact readback"),
                size: artifact_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })),
            parameter_bytes,
            artifact_bytes,
            allocation_bytes,
        }
    }
}

#[derive(Debug)]
struct IdleBufferSet {
    buffers: EncoderBufferSet,
    generation: u64,
}

#[derive(Debug)]
struct PoolState {
    limit_bytes: u64,
    idle_bytes: u64,
    leased_buffer_sets: u64,
    reuse_hits: u64,
    allocation_misses: u64,
    evicted_buffer_sets: u64,
    evicted_bytes: u64,
    generation: u64,
    idle: VecDeque<IdleBufferSet>,
}

impl PoolState {
    fn record_eviction(&mut self, buffers: &EncoderBufferSet) {
        self.evicted_buffer_sets = self.evicted_buffer_sets.saturating_add(1);
        self.evicted_bytes = self.evicted_bytes.saturating_add(buffers.allocation_bytes);
    }

    fn evict_oldest(&mut self) -> Option<EncoderBufferSet> {
        let idle = self.idle.pop_front()?;
        self.idle_bytes = self
            .idle_bytes
            .saturating_sub(idle.buffers.allocation_bytes);
        self.record_eviction(&idle.buffers);
        Some(idle.buffers)
    }

    fn take_excess(&mut self) -> Vec<EncoderBufferSet> {
        let mut evicted = Vec::new();
        while self.idle_bytes > self.limit_bytes
            || self.idle.len() > MAX_ENCODER_BUFFER_POOL_IDLE_SETS
        {
            let Some(buffers) = self.evict_oldest() else {
                break;
            };
            evicted.push(buffers);
        }
        evicted
    }
}

#[derive(Debug)]
pub(crate) struct EncoderBufferPool {
    state: Mutex<PoolState>,
}

impl EncoderBufferPool {
    pub(crate) fn new(limit_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(PoolState {
                limit_bytes,
                idle_bytes: 0,
                leased_buffer_sets: 0,
                reuse_hits: 0,
                allocation_misses: 0,
                evicted_buffer_sets: 0,
                evicted_bytes: 0,
                generation: 0,
                idle: VecDeque::new(),
            }),
        })
    }

    pub(crate) fn checkout(
        self: &Arc<Self>,
        device: &wgpu::Device,
        parameter_bytes: u64,
        artifact_bytes: u64,
    ) -> EncoderBufferLease {
        let generation = {
            let mut state = self.lock_state();
            let generation = state.generation;
            let exact_match = state.idle.iter().position(|idle| {
                idle.generation == generation
                    && idle.buffers.parameter_bytes == parameter_bytes
                    && idle.buffers.artifact_bytes == artifact_bytes
            });
            if let Some(idle) = exact_match.and_then(|index| state.idle.remove(index)) {
                state.idle_bytes = state
                    .idle_bytes
                    .saturating_sub(idle.buffers.allocation_bytes);
                state.reuse_hits = state.reuse_hits.saturating_add(1);
                state.leased_buffer_sets = state.leased_buffer_sets.saturating_add(1);
                return EncoderBufferLease {
                    pool: Arc::clone(self),
                    buffers: Some(idle.buffers),
                    generation,
                };
            }
            state.allocation_misses = state.allocation_misses.saturating_add(1);
            generation
        };
        // Creating driver objects can be expensive. Keep cold concurrent submissions independent
        // by never holding the pool mutex across `Device::create_buffer`.
        let buffers = EncoderBufferSet::new(device, parameter_bytes, artifact_bytes);
        {
            let mut state = self.lock_state();
            state.leased_buffer_sets = state.leased_buffer_sets.saturating_add(1);
        }
        EncoderBufferLease {
            pool: Arc::clone(self),
            buffers: Some(buffers),
            generation,
        }
    }

    pub(crate) fn stats(&self) -> EncoderBufferPoolStats {
        let state = self.lock_state();
        let idle_buffer_sets = u64::try_from(state.idle.len()).unwrap_or(u64::MAX);
        EncoderBufferPoolStats {
            limit_bytes: state.limit_bytes,
            idle_bytes: state.idle_bytes,
            idle_buffer_sets,
            max_idle_buffer_sets: MAX_ENCODER_BUFFER_POOL_IDLE_SETS as u64,
            idle_buffers: idle_buffer_sets.saturating_mul(3),
            leased_buffer_sets: state.leased_buffer_sets,
            reuse_hits: state.reuse_hits,
            allocation_misses: state.allocation_misses,
            evicted_buffer_sets: state.evicted_buffer_sets,
            evicted_buffers: state.evicted_buffer_sets.saturating_mul(3),
            evicted_bytes: state.evicted_bytes,
        }
    }

    pub(crate) fn set_limit(&self, limit_bytes: u64) {
        let evicted = {
            let mut state = self.lock_state();
            state.limit_bytes = limit_bytes;
            state.take_excess()
        };
        drop(evicted);
    }

    pub(crate) fn clear(&self) {
        let evicted = {
            let mut state = self.lock_state();
            state.generation = state.generation.wrapping_add(1);
            let mut evicted = Vec::with_capacity(state.idle.len());
            while let Some(buffers) = state.evict_oldest() {
                evicted.push(buffers);
            }
            evicted
        };
        drop(evicted);
    }

    fn return_buffers(&self, buffers: EncoderBufferSet, generation: u64) {
        let evicted = {
            let mut state = self.lock_state();
            state.leased_buffer_sets = state.leased_buffer_sets.saturating_sub(1);
            let invalid_return = generation != state.generation
                || buffers.allocation_bytes > state.limit_bytes
                || state
                    .idle_bytes
                    .checked_add(buffers.allocation_bytes)
                    .is_none();
            if invalid_return {
                state.record_eviction(&buffers);
                vec![buffers]
            } else {
                let mut evicted = Vec::new();
                while state
                    .idle_bytes
                    .checked_add(buffers.allocation_bytes)
                    .is_some_and(|bytes| bytes > state.limit_bytes)
                    || state.idle.len() >= MAX_ENCODER_BUFFER_POOL_IDLE_SETS
                {
                    let Some(oldest) = state.evict_oldest() else {
                        break;
                    };
                    evicted.push(oldest);
                }
                state.idle_bytes += buffers.allocation_bytes;
                state.idle.push_back(IdleBufferSet {
                    buffers,
                    generation,
                });
                evicted
            }
        };
        // Releasing driver objects outside the mutex keeps unrelated submissions non-blocking.
        drop(evicted);
    }

    fn lock_state(&self) -> MutexGuard<'_, PoolState> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

/// Exclusive ownership of one complete buffer set until the GPU map callback finishes.
pub(crate) struct EncoderBufferLease {
    pool: Arc<EncoderBufferPool>,
    buffers: Option<EncoderBufferSet>,
    generation: u64,
}

impl EncoderBufferLease {
    pub(crate) fn buffers(&self) -> &EncoderBufferSet {
        self.buffers
            .as_ref()
            .expect("pooled buffer lease has not been returned")
    }
}

impl Drop for EncoderBufferLease {
    fn drop(&mut self) {
        if let Some(buffers) = self.buffers.take() {
            self.pool.return_buffers(buffers, self.generation);
        }
    }
}
