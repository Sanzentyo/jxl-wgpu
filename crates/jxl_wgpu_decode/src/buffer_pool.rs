use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

/// Default byte limit for idle buffers retained by the stock decoder.
pub const DEFAULT_DECODE_BUFFER_POOL_BYTES: u64 = 32 * 1024 * 1024;

/// Default object-count limit for idle decoder buffers.
pub const DEFAULT_DECODE_BUFFER_POOL_BUFFERS: usize = 256;

/// Default number of equal-size/equal-usage buffers retained for one exact key.
pub const DEFAULT_DECODE_BUFFER_POOL_BUFFERS_PER_KEY: usize = 32;

/// Hard limits for the stock decoder's idle transient-buffer cache.
///
/// A zero value disables the corresponding form of retention. These limits apply only to idle
/// physical allocations. Active decode bytes remain governed by the shared `MemoryBudget`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuDecodeBufferPoolLimits {
    pub max_idle_bytes: u64,
    pub max_idle_buffers: usize,
    pub max_idle_buffers_per_key: usize,
}

impl Default for WgpuDecodeBufferPoolLimits {
    fn default() -> Self {
        Self {
            max_idle_bytes: DEFAULT_DECODE_BUFFER_POOL_BYTES,
            max_idle_buffers: DEFAULT_DECODE_BUFFER_POOL_BUFFERS,
            max_idle_buffers_per_key: DEFAULT_DECODE_BUFFER_POOL_BUFFERS_PER_KEY,
        }
    }
}

/// Point-in-time counters for the stock decoder's transient-buffer pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WgpuDecodeBufferPoolStats {
    pub limits: WgpuDecodeBufferPoolLimits,
    /// Acquisitions satisfied by an exact size/usage/alignment match.
    pub hits: u64,
    /// Acquisitions that created a new physical buffer.
    pub misses: u64,
    /// Buffers safely returned after GPU completion and mapping teardown.
    pub recycled: u64,
    /// Physical buffers discarded by limits, clearing, or generation invalidation.
    pub evicted: u64,
    pub evicted_bytes: u64,
    pub idle_buffers: u64,
    pub idle_bytes: u64,
    pub leased_buffers: u64,
    pub leased_bytes: u64,
    /// Incremented by every explicit clear. Outstanding leases from older generations are
    /// discarded when they eventually complete.
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct BufferKey {
    size: u64,
    usage_bits: u32,
    alignment: u64,
}

impl BufferKey {
    fn new(size: u64, usage: wgpu::BufferUsages, alignment: u64) -> Self {
        debug_assert!(size > 0);
        debug_assert!(alignment.is_power_of_two());
        debug_assert!(size.is_multiple_of(alignment));
        Self {
            size,
            usage_bits: usage.bits(),
            alignment,
        }
    }
}

struct IdleBuffer {
    key: BufferKey,
    buffer: Arc<wgpu::Buffer>,
}

struct PoolState {
    limits: WgpuDecodeBufferPoolLimits,
    hits: u64,
    misses: u64,
    recycled: u64,
    evicted: u64,
    evicted_bytes: u64,
    idle_bytes: u64,
    leased_buffers: u64,
    leased_bytes: u64,
    generation: u64,
    // Oldest at the front. The default 256-object bound makes exact matching and deterministic
    // global eviction cheaper and simpler than maintaining a second keyed LRU index.
    idle: VecDeque<IdleBuffer>,
}

impl PoolState {
    fn record_eviction(&mut self, key: BufferKey) {
        self.evicted = self.evicted.saturating_add(1);
        self.evicted_bytes = self.evicted_bytes.saturating_add(key.size);
    }

    fn remove_idle(&mut self, index: usize) -> Option<IdleBuffer> {
        let idle = self.idle.remove(index)?;
        self.idle_bytes = self.idle_bytes.saturating_sub(idle.key.size);
        Some(idle)
    }

    fn evict_at(&mut self, index: usize) -> Option<Arc<wgpu::Buffer>> {
        let idle = self.remove_idle(index)?;
        self.record_eviction(idle.key);
        Some(idle.buffer)
    }

    fn trim_to_limits(&mut self) -> Vec<Arc<wgpu::Buffer>> {
        let mut evicted = Vec::new();
        loop {
            let over_global = self.idle_bytes > self.limits.max_idle_bytes
                || self.idle.len() > self.limits.max_idle_buffers;
            let over_key = self.idle.iter().position(|candidate| {
                self.idle
                    .iter()
                    .filter(|idle| idle.key == candidate.key)
                    .count()
                    > self.limits.max_idle_buffers_per_key
            });
            let index = if over_global { Some(0) } else { over_key };
            let Some(index) = index else {
                break;
            };
            let Some(buffer) = self.evict_at(index) else {
                break;
            };
            evicted.push(buffer);
        }
        evicted
    }
}

/// Decoder-local pool whose leases cross the GPU completion boundary explicitly.
pub(crate) struct DecodeBufferPool {
    device: wgpu::Device,
    state: Mutex<PoolState>,
}

impl std::fmt::Debug for DecodeBufferPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecodeBufferPool")
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl DecodeBufferPool {
    pub(crate) fn new(device: wgpu::Device, limits: WgpuDecodeBufferPoolLimits) -> Arc<Self> {
        Arc::new(Self {
            device,
            state: Mutex::new(PoolState {
                limits,
                hits: 0,
                misses: 0,
                recycled: 0,
                evicted: 0,
                evicted_bytes: 0,
                idle_bytes: 0,
                leased_buffers: 0,
                leased_bytes: 0,
                generation: 0,
                idle: VecDeque::new(),
            }),
        })
    }

    pub(crate) fn checkout(
        self: &Arc<Self>,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
        alignment: u64,
    ) -> DecodeBufferLease {
        let key = BufferKey::new(size, usage, alignment);
        let (generation, cached) = {
            let mut state = self.lock_state();
            let generation = state.generation;
            let cached = state
                .idle
                .iter()
                .position(|idle| idle.key == key)
                .and_then(|index| state.remove_idle(index));
            if cached.is_some() {
                state.hits = state.hits.saturating_add(1);
            } else {
                state.misses = state.misses.saturating_add(1);
            }
            state.leased_buffers = state.leased_buffers.saturating_add(1);
            state.leased_bytes = state.leased_bytes.saturating_add(key.size);
            (generation, cached)
        };

        // Driver allocation may be slow, so cold concurrent submissions do not hold the mutex.
        let buffer = cached.map_or_else(
            || {
                Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                }))
            },
            |idle| idle.buffer,
        );
        DecodeBufferLease {
            pool: Arc::clone(self),
            buffer: Some(buffer),
            key,
            generation,
        }
    }

    pub(crate) fn limits(&self) -> WgpuDecodeBufferPoolLimits {
        self.lock_state().limits
    }

    pub(crate) fn set_limits(&self, limits: WgpuDecodeBufferPoolLimits) {
        let evicted = {
            let mut state = self.lock_state();
            state.limits = limits;
            state.trim_to_limits()
        };
        // Backend allocation destruction never runs while the state mutex is held.
        drop(evicted);
    }

    pub(crate) fn clear(&self) -> u64 {
        let (generation, evicted) = {
            let mut state = self.lock_state();
            state.generation = state.generation.wrapping_add(1);
            let generation = state.generation;
            let mut evicted = Vec::with_capacity(state.idle.len());
            while let Some(buffer) = state.evict_at(0) {
                evicted.push(buffer);
            }
            (generation, evicted)
        };
        drop(evicted);
        generation
    }

    pub(crate) fn stats(&self) -> WgpuDecodeBufferPoolStats {
        let state = self.lock_state();
        WgpuDecodeBufferPoolStats {
            limits: state.limits,
            hits: state.hits,
            misses: state.misses,
            recycled: state.recycled,
            evicted: state.evicted,
            evicted_bytes: state.evicted_bytes,
            idle_buffers: u64::try_from(state.idle.len()).unwrap_or(u64::MAX),
            idle_bytes: state.idle_bytes,
            leased_buffers: state.leased_buffers,
            leased_bytes: state.leased_bytes,
            generation: state.generation,
        }
    }

    fn return_buffer(&self, buffer: Arc<wgpu::Buffer>, key: BufferKey, generation: u64) {
        let evicted = {
            let mut state = self.lock_state();
            state.leased_buffers = state.leased_buffers.saturating_sub(1);
            state.leased_bytes = state.leased_bytes.saturating_sub(key.size);

            let same_key = state.idle.iter().filter(|idle| idle.key == key).count();
            let cacheable = generation == state.generation
                && key.size <= state.limits.max_idle_bytes
                && state.limits.max_idle_buffers > 0
                && state.limits.max_idle_buffers_per_key > 0
                && same_key < state.limits.max_idle_buffers_per_key;
            if !cacheable {
                state.record_eviction(key);
                vec![buffer]
            } else {
                let mut evicted = Vec::new();
                while state.idle.len() >= state.limits.max_idle_buffers
                    || state
                        .idle_bytes
                        .checked_add(key.size)
                        .is_none_or(|bytes| bytes > state.limits.max_idle_bytes)
                {
                    let Some(oldest) = state.evict_at(0) else {
                        break;
                    };
                    evicted.push(oldest);
                }
                state.idle_bytes += key.size;
                state.idle.push_back(IdleBuffer { key, buffer });
                state.recycled = state.recycled.saturating_add(1);
                evicted
            }
        };
        drop(evicted);
    }

    fn lock_state(&self) -> MutexGuard<'_, PoolState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Exclusive transient allocation retained until a stock decode job's completion callback and
/// consumer-side status validation have both released the job lifetime.
pub(crate) struct DecodeBufferLease {
    pool: Arc<DecodeBufferPool>,
    buffer: Option<Arc<wgpu::Buffer>>,
    key: BufferKey,
    generation: u64,
}

impl DecodeBufferLease {
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        self.buffer
            .as_deref()
            .expect("decoder buffer lease is returned exactly once")
    }
}

impl Drop for DecodeBufferLease {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_buffer(buffer, self.key, self.generation);
        }
    }
}
