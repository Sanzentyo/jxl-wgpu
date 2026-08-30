// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Thread-safe cache for lazily compiled compute pipelines.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::autotune::KernelVariant;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PipelineKey {
    pub shader: Arc<str>,
    pub entry_point: Arc<str>,
    pub variant: KernelVariant,
    /// Hash of bind-group layout and shader constants not represented by `variant`.
    pub layout_hash: u64,
}

impl PipelineKey {
    pub(crate) fn new(
        shader: impl Into<Arc<str>>,
        entry_point: impl Into<Arc<str>>,
        variant: KernelVariant,
        layout_hash: u64,
    ) -> Self {
        Self {
            shader: shader.into(),
            entry_point: entry_point.into(),
            variant,
            layout_hash,
        }
    }
}

pub(crate) struct PipelineCache<T = wgpu::ComputePipeline> {
    entries: Mutex<HashMap<PipelineKey, Arc<T>>>,
}

impl<T> Default for PipelineCache<T> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}

impl<T> fmt::Debug for PipelineCache<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PipelineCache")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

impl<T> PipelineCache<T> {
    pub(crate) fn get(&self, key: &PipelineKey) -> Option<Arc<T>> {
        self.entries().get(key).cloned()
    }

    /// Returns the existing entry or creates it exactly once while holding the cache lock.
    ///
    /// Pipeline creation is relatively expensive, but it is also rare and keyed. Serializing
    /// creation here avoids compiling the same WGSL pipeline multiple times when frame sessions
    /// start concurrently.
    pub(crate) fn get_or_insert_with<E>(
        &self,
        key: PipelineKey,
        create: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<Arc<T>, E> {
        let mut entries = self.entries();
        if let Some(value) = entries.get(&key) {
            return Ok(value.clone());
        }
        let value = Arc::new(create()?);
        entries.insert(key, value.clone());
        Ok(value)
    }

    pub(crate) fn len(&self) -> usize {
        self.entries().len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries().is_empty()
    }

    pub(crate) fn clear(&self) {
        self.entries().clear();
    }

    fn entries(&self) -> MutexGuard<'_, HashMap<PipelineKey, Arc<T>>> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn key() -> PipelineKey {
        PipelineKey::new("copy.wgsl", "main", KernelVariant::Tile16x16, 7)
    }

    #[test]
    fn creates_each_key_once() {
        let cache = PipelineCache::<u32>::default();
        let calls = AtomicUsize::new(0);
        let first = cache
            .get_or_insert_with(key(), || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok::<_, ()>(42)
            })
            .unwrap();
        let second = cache
            .get_or_insert_with(key(), || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok::<_, ()>(99)
            })
            .unwrap();
        assert_eq!(*first, 42);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn creation_errors_are_not_cached() {
        let cache = PipelineCache::<u32>::default();
        assert_eq!(
            cache.get_or_insert_with(key(), || Err::<u32, _>("compile failed")),
            Err("compile failed")
        );
        assert!(cache.is_empty());
        assert_eq!(
            *cache
                .get_or_insert_with(key(), || Ok::<_, &str>(3))
                .unwrap(),
            3
        );
    }

    #[test]
    fn keys_include_variant_and_layout() {
        let cache = PipelineCache::<u32>::default();
        let first = key();
        let mut second = first.clone();
        second.variant = KernelVariant::Tile8x8;
        cache.get_or_insert_with(first, || Ok::<_, ()>(1)).unwrap();
        cache.get_or_insert_with(second, || Ok::<_, ()>(2)).unwrap();
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }
}
