use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Waker};

use jxl_wgpu::{GpuBufferLease, MemoryPermit};

use crate::GpuCodestream;
use crate::buffer_pool::DecodeBufferLease;
use crate::modular_finalize::ModularFinalizeParams;
use crate::profile::StandardModularProfile;
use crate::progressive_dc::ProgressiveDcXybPlanes;

use super::execution::{GroupDispatchLayout, OutputPlan};
pub(super) struct DecodeSource {
    pub(super) codestream: Arc<GpuCodestream>,
    pub(super) profile: StandardModularProfile,
    pub(super) dispatch_layout: GroupDispatchLayout,
    // Immutable within the session. Global and deduplicated local MA/entropy descriptors share
    // one rebased word buffer without sharing mutable GPU transient allocations.
    pub(super) modular_metadata: Arc<[u32]>,
    pub(super) ma_metadata_offsets: Arc<[u32]>,
    pub(super) global_ma_metadata_offset: Option<u32>,
    pub(super) channel_layout_offsets: Arc<[u32]>,
    pub(super) global_channel_layout_offset: Option<u32>,
    pub(super) finalize_params: Arc<[ModularFinalizeParams]>,
    pub(super) output: OutputPlan,
}

pub(super) struct DecodeJobLifetime {
    pub(super) output: GpuBufferLease,
    pub(super) _modular_metadata: DecodeBufferLease,
    pub(super) _reconstructed: DecodeBufferLease,
    pub(super) _frame_arena: Option<DecodeBufferLease>,
    pub(super) _native_f64_dummy_words: Option<DecodeBufferLease>,
    pub(super) _status: DecodeBufferLease,
    pub(super) status_staging: DecodeBufferLease,
    pub(super) status_mapped: AtomicBool,
    pub(super) _params: DecodeBufferLease,
    pub(super) _dispatch_control: DecodeBufferLease,
    pub(super) _transient_permit: MemoryPermit,
    pub(super) progressive_dc_planes: Option<ProgressiveDcXybPlanes>,
    pub(super) _progressive_dc_uniform: Mutex<Option<wgpu::Buffer>>,
}

impl Drop for DecodeJobLifetime {
    fn drop(&mut self) {
        // A successful map remains mapped until explicitly released. This also covers abandoned
        // sessions/Futures: the callback owns the final Arc until mapping has completed, then this
        // drop runs and unmaps before field destruction returns the staging lease to the pool.
        if self.status_mapped.swap(false, Ordering::AcqRel) {
            self.status_staging.buffer().unmap();
        }
    }
}

pub(super) struct DecodeMemoryPermits {
    pub(super) output: MemoryPermit,
    pub(super) transient: MemoryPermit,
}
#[derive(Default)]
pub(super) struct MapCompletion {
    pub(super) state: Mutex<MapState>,
    pub(super) condition: Condvar,
}

#[derive(Default)]
pub(super) struct MapState {
    pub(super) result: Option<std::result::Result<(), String>>,
    pub(super) waker: Option<Waker>,
}

impl MapCompletion {
    pub(super) fn complete(&self, result: std::result::Result<(), String>) {
        let waker = {
            let mut state = lock_unpoisoned(&self.state);
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        self.condition.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    pub(super) fn poll(&self, context: &Context<'_>) -> Option<std::result::Result<(), String>> {
        let mut state = lock_unpoisoned(&self.state);
        if state.result.is_none() {
            state.waker = Some(context.waker().clone());
        }
        state.result.take()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn wait(&self) -> std::result::Result<(), String> {
        let mut state = lock_unpoisoned(&self.state);
        while state.result.is_none() {
            state = self
                .condition
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state
            .result
            .take()
            .expect("mapping result was checked as present")
    }
}

pub(super) fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
