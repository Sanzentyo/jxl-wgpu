// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Bounded, exact-match reuse for internal GPU buffers.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

/// Snapshot of the accelerator-wide internal buffer pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WgpuBufferPoolStats {
    /// Acquisitions satisfied by an exactly matching cached allocation.
    pub hits: u64,
    /// Acquisitions that created a new `wgpu::Buffer`.
    pub misses: u64,
    /// Buffers accepted back into the cache.
    pub recycled: u64,
    /// Cached buffers released to remain within the byte limit.
    pub evicted: u64,
    /// Buffers not cached because they were aliased, too large, or caching was disabled.
    pub rejected: u64,
    /// Number of buffers currently retained by the cache.
    pub cached_buffers: u64,
    /// Bytes currently retained by the cache.
    pub cached_bytes: u64,
    /// Cached buffers whose contents are known to be all zero.
    pub zeroed_buffers: u64,
}

struct CacheEntry {
    buffer: Arc<wgpu::Buffer>,
    size: u64,
    usage: wgpu::BufferUsages,
    zeroed: bool,
}

#[derive(Default)]
struct PoolState {
    // Oldest at the front. The pool is deliberately small and exact-match lookup is O(n), which
    // keeps eviction deterministic without maintaining a second global LRU index.
    available: VecDeque<CacheEntry>,
    cached_bytes: u64,
}

/// Accelerator-wide cache. A lease must be returned only at a proven GPU ownership boundary.
pub(crate) struct BufferPool {
    device: wgpu::Device,
    max_cached_bytes: u64,
    state: Mutex<PoolState>,
    hits: AtomicU64,
    misses: AtomicU64,
    recycled: AtomicU64,
    evicted: AtomicU64,
    rejected: AtomicU64,
}

impl fmt::Debug for BufferPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BufferPool")
            .field("max_cached_bytes", &self.max_cached_bytes)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl BufferPool {
    pub(crate) fn new(device: wgpu::Device, max_cached_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            device,
            max_cached_bytes,
            state: Mutex::new(PoolState::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            recycled: AtomicU64::new(0),
            evicted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
        })
    }

    pub(crate) fn acquire(
        self: &Arc<Self>,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> PooledBuffer {
        self.acquire_with_contents(label, size, usage, false)
    }

    /// Acquires only a newly allocated or cached-known-zero buffer.
    pub(crate) fn acquire_zeroed(
        self: &Arc<Self>,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> PooledBuffer {
        self.acquire_with_contents(label, size, usage, true)
    }

    fn acquire_with_contents(
        self: &Arc<Self>,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
        require_zeroed: bool,
    ) -> PooledBuffer {
        let cached = {
            let mut state = self.state();
            let position = state.available.iter().position(|entry| {
                entry.size == size && entry.usage == usage && (!require_zeroed || entry.zeroed)
            });
            position.and_then(|position| {
                let entry = state.available.remove(position)?;
                state.cached_bytes = state.cached_bytes.saturating_sub(entry.size);
                Some(entry)
            })
        };

        let (buffer, was_reused) = if let Some(entry) = cached {
            self.hits.fetch_add(1, Ordering::Relaxed);
            (entry.buffer, true)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            (
                Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                })),
                false,
            )
        };

        PooledBuffer {
            buffer: Some(buffer),
            size,
            usage,
            owner: Arc::downgrade(self),
            return_zeroed: false,
            was_reused,
        }
    }

    /// Releases every idle cached allocation. In-flight leases are unaffected.
    pub(crate) fn clear(&self) {
        let evicted = {
            let mut state = self.state();
            state.cached_bytes = 0;
            std::mem::take(&mut state.available)
        };
        self.evicted.fetch_add(
            u64::try_from(evicted.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        drop(evicted);
    }

    pub(crate) fn stats(&self) -> WgpuBufferPoolStats {
        let state = self.state();
        WgpuBufferPoolStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            recycled: self.recycled.load(Ordering::Relaxed),
            evicted: self.evicted.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            cached_buffers: u64::try_from(state.available.len()).unwrap_or(u64::MAX),
            cached_bytes: state.cached_bytes,
            zeroed_buffers: u64::try_from(
                state.available.iter().filter(|entry| entry.zeroed).count(),
            )
            .unwrap_or(u64::MAX),
        }
    }

    fn recycle(&self, mut lease: PooledBuffer) -> bool {
        let Some(buffer) = lease.buffer.take() else {
            return false;
        };
        // An internal lease is cacheable only when no caller-visible or pending readback Arc can
        // still reference it. This also keeps GPU-only public output buffers permanently outside
        // the pool even if an integration mistake attempts to return one.
        if self.max_cached_bytes == 0
            || lease.size > self.max_cached_bytes
            || Arc::strong_count(&buffer) != 1
        {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let mut evicted = Vec::new();
        let mut state = self.state();
        while state
            .cached_bytes
            .checked_add(lease.size)
            .is_none_or(|bytes| bytes > self.max_cached_bytes)
        {
            let Some(entry) = state.available.pop_front() else {
                self.rejected.fetch_add(1, Ordering::Relaxed);
                return false;
            };
            state.cached_bytes = state.cached_bytes.saturating_sub(entry.size);
            evicted.push(entry);
        }
        state.cached_bytes += lease.size;
        state.available.push_back(CacheEntry {
            buffer,
            size: lease.size,
            usage: lease.usage,
            zeroed: lease.return_zeroed,
        });
        drop(state);

        self.evicted.fetch_add(
            u64::try_from(evicted.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        // Avoid dropping backend allocations while holding the pool mutex.
        drop(evicted);
        self.recycled.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn state(&self) -> MutexGuard<'_, PoolState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Exclusive host-side lease for an internal pool allocation.
///
/// Dropping a lease frees the buffer normally. Call [`Self::recycle`] only after all commands that
/// reference it have been submitted to the owning queue, or after a direct mapping was unmapped.
pub(crate) struct PooledBuffer {
    buffer: Option<Arc<wgpu::Buffer>>,
    size: u64,
    usage: wgpu::BufferUsages,
    owner: Weak<BufferPool>,
    return_zeroed: bool,
    was_reused: bool,
}

impl fmt::Debug for PooledBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PooledBuffer")
            .field("size", &self.size)
            .field("usage", &self.usage)
            .field("leased", &self.buffer.is_some())
            .field("was_reused", &self.was_reused)
            .field("return_zeroed", &self.return_zeroed)
            .finish_non_exhaustive()
    }
}

impl PooledBuffer {
    pub(crate) fn buffer(&self) -> &Arc<wgpu::Buffer> {
        self.buffer
            .as_ref()
            .expect("a pooled buffer lease is consumed exactly once")
    }

    pub(crate) fn recycle(self) -> bool {
        let Some(owner) = self.owner.upgrade() else {
            return false;
        };
        owner.recycle(self)
    }

    pub(crate) fn cacheable(&self) -> bool {
        self.owner
            .upgrade()
            .is_some_and(|owner| owner.max_cached_bytes != 0 && self.size <= owner.max_cached_bytes)
    }

    #[cfg(test)]
    pub(crate) const fn was_reused(&self) -> bool {
        self.was_reused
    }

    /// Marks that the owning command buffer clears the complete allocation before submission
    /// finishes. This is metadata only; callers must record the clear themselves.
    pub(crate) fn mark_zeroed_on_recycle(&mut self) {
        self.return_zeroed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jxl-wgpu buffer pool test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()
    }

    #[test]
    fn exact_size_and_usage_are_reused() {
        let Some((device, _queue)) = device() else {
            eprintln!("skipping GPU buffer pool test: no adapter");
            return;
        };
        let pool = BufferPool::new(device, 4096);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let first = pool.acquire("first", 256, usage);
        assert!(!first.was_reused());
        let identity = Arc::as_ptr(first.buffer());
        assert!(first.recycle());

        let second = pool.acquire("second", 256, usage);
        assert!(second.was_reused());
        assert_eq!(Arc::as_ptr(second.buffer()), identity);
        assert_eq!(pool.stats().hits, 1);
        assert!(second.recycle());
    }

    #[test]
    fn aliases_are_never_cached_and_limit_is_hard() {
        let Some((device, _queue)) = device() else {
            eprintln!("skipping GPU buffer pool test: no adapter");
            return;
        };
        let pool = BufferPool::new(device, 384);
        let aliased = pool.acquire("aliased", 128, wgpu::BufferUsages::COPY_DST);
        let external = Arc::clone(aliased.buffer());
        assert!(!aliased.recycle());
        assert_eq!(pool.stats().cached_bytes, 0);
        drop(external);

        let first = pool.acquire("first", 256, wgpu::BufferUsages::COPY_DST);
        assert!(first.recycle());
        let second = pool.acquire("second", 256, wgpu::BufferUsages::COPY_SRC);
        assert!(second.recycle());
        let stats = pool.stats();
        assert_eq!(stats.cached_bytes, 256);
        assert_eq!(stats.cached_buffers, 1);
        assert_eq!(stats.evicted, 1);
        assert_eq!(stats.rejected, 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn tail_clear_prevents_cross_frame_data_leak_when_reused_before_completion() {
        use std::sync::mpsc;

        let Some((device, queue)) = device() else {
            eprintln!("skipping GPU buffer pool test: no adapter");
            return;
        };
        let pool = BufferPool::new(device.clone(), 4096);
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mut first = pool.acquire("first frame resident slot", 64, usage);
        queue.write_buffer(first.buffer(), 0, &[0xa5; 64]);
        let mut first_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("first frame tail clear"),
        });
        first_encoder.clear_buffer(first.buffer(), 0, None);
        first.mark_zeroed_on_recycle();
        queue.submit([first_encoder.finish()]);
        // Deliberately return before GPU completion. Queue ordering, not a host wait, is the safety
        // mechanism used by the production session.
        assert!(first.recycle());

        let second = pool.acquire_zeroed("second frame resident slot", 64, usage);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cross-frame zeroing readback"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut second_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("second frame observe zero"),
        });
        second_encoder.copy_buffer_to_buffer(second.buffer(), 0, &readback, 0, 64);
        let (sender, receiver) = mpsc::sync_channel(1);
        second_encoder.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        let submission = queue.submit([second_encoder.finish()]);
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = readback.slice(..).get_mapped_range().unwrap();
        assert!(mapped.iter().all(|&byte| byte == 0));
        drop(mapped);
        readback.unmap();
        assert!(second.recycle());
    }

    #[test]
    fn dirty_entries_do_not_satisfy_zero_initialized_acquisitions() {
        let Some((device, _queue)) = device() else {
            eprintln!("skipping GPU buffer pool test: no adapter");
            return;
        };
        let pool = BufferPool::new(device, 4096);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let dirty = pool.acquire("dirty", 256, usage);
        let dirty_identity = Arc::as_ptr(dirty.buffer());
        assert!(dirty.recycle());

        let zeroed = pool.acquire_zeroed("zero required", 256, usage);
        assert_ne!(Arc::as_ptr(zeroed.buffer()), dirty_identity);
        assert_eq!(pool.stats().misses, 2);
        assert!(!zeroed.was_reused());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn concurrent_acquisitions_never_lease_the_same_buffer() {
        use std::sync::Barrier;

        let Some((device, _queue)) = device() else {
            eprintln!("skipping GPU buffer pool test: no adapter");
            return;
        };
        let pool = BufferPool::new(device, 4096);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let warm = pool.acquire("warm", 256, usage);
        assert!(warm.recycle());

        let start = Arc::new(Barrier::new(3));
        let acquired = Arc::new(Barrier::new(3));
        let workers = (0..2)
            .map(|index| {
                let pool = Arc::clone(&pool);
                let start = Arc::clone(&start);
                let acquired = Arc::clone(&acquired);
                std::thread::spawn(move || {
                    start.wait();
                    let lease = pool.acquire(&format!("concurrent {index}"), 256, usage);
                    let identity = Arc::as_ptr(lease.buffer()) as usize;
                    acquired.wait();
                    assert!(lease.recycle());
                    identity
                })
            })
            .collect::<Vec<_>>();
        start.wait();
        acquired.wait();
        let identities = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_ne!(identities[0], identities[1]);
        assert_eq!(pool.stats().cached_buffers, 2);
    }
}
