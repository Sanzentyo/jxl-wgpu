// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Byte-weighted admission for caller-visible, concurrently live GPU allocations.

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A point-in-time view of a [`MemoryBudget`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudgetSnapshot {
    pub limit_bytes: u64,
    pub reserved_bytes: u64,
    pub available_bytes: u64,
}

/// Failure to admit a byte-weighted GPU allocation without blocking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MemoryBudgetError {
    #[error(
        "GPU memory admission needs {requested_bytes} bytes with {reserved_bytes}/{limit_bytes} bytes already reserved"
    )]
    Exhausted {
        requested_bytes: u64,
        reserved_bytes: u64,
        limit_bytes: u64,
    },
    #[error(
        "GPU memory admission overflowed while adding {requested_bytes} bytes to {reserved_bytes} reserved bytes"
    )]
    Overflow {
        requested_bytes: u64,
        reserved_bytes: u64,
    },
}

struct MemoryBudgetInner {
    limit_bytes: u64,
    reserved_bytes: AtomicU64,
}

/// Cloneable, runtime-independent byte budget shared by concurrent GPU submissions.
///
/// Admission is deliberately non-blocking: [`Self::try_reserve`] either returns an owned permit
/// or a typed error. This keeps the primitive usable from synchronous code and from any async
/// runtime without creating a runtime-specific semaphore or waiter queue.
#[derive(Clone)]
pub struct MemoryBudget {
    inner: Arc<MemoryBudgetInner>,
}

impl std::fmt::Debug for MemoryBudget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MemoryBudget")
            .field("snapshot", &self.snapshot())
            .finish()
    }
}

impl MemoryBudget {
    #[must_use]
    pub fn new(limit_bytes: NonZeroU64) -> Self {
        Self {
            inner: Arc::new(MemoryBudgetInner {
                limit_bytes: limit_bytes.get(),
                reserved_bytes: AtomicU64::new(0),
            }),
        }
    }

    /// Attempts to reserve bytes immediately. Clones of the returned permit share one
    /// reservation, which is released only after the final clone is dropped.
    pub fn try_reserve(&self, bytes: u64) -> Result<MemoryPermit, MemoryBudgetError> {
        let mut reserved = self.inner.reserved_bytes.load(Ordering::Acquire);
        loop {
            let next = reserved
                .checked_add(bytes)
                .ok_or(MemoryBudgetError::Overflow {
                    requested_bytes: bytes,
                    reserved_bytes: reserved,
                })?;
            if next > self.inner.limit_bytes {
                return Err(MemoryBudgetError::Exhausted {
                    requested_bytes: bytes,
                    reserved_bytes: reserved,
                    limit_bytes: self.inner.limit_bytes,
                });
            }
            match self.inner.reserved_bytes.compare_exchange_weak(
                reserved,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(MemoryPermit {
                        reservation: Arc::new(MemoryReservation {
                            budget: Arc::clone(&self.inner),
                            bytes,
                        }),
                    });
                }
                Err(actual) => reserved = actual,
            }
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> MemoryBudgetSnapshot {
        let reserved_bytes = self.inner.reserved_bytes.load(Ordering::Acquire);
        MemoryBudgetSnapshot {
            limit_bytes: self.inner.limit_bytes,
            reserved_bytes,
            available_bytes: self.inner.limit_bytes.saturating_sub(reserved_bytes),
        }
    }
}

struct MemoryReservation {
    budget: Arc<MemoryBudgetInner>,
    bytes: u64,
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        let previous = self
            .budget
            .reserved_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
        debug_assert!(previous >= self.bytes, "memory permit accounting underflow");
    }
}

/// Cloneable ownership token for bytes admitted by a [`MemoryBudget`].
///
/// A clone does not reserve the bytes again. The reservation is returned atomically when the last
/// clone is dropped, making this suitable for GPU buffer leases that can cross sync/async API
/// boundaries.
#[derive(Clone)]
pub struct MemoryPermit {
    reservation: Arc<MemoryReservation>,
}

impl MemoryPermit {
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.reservation.bytes
    }
}

impl std::fmt::Debug for MemoryPermit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MemoryPermit")
            .field("bytes", &self.bytes())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use std::thread;

    use super::{MemoryBudget, MemoryBudgetError};

    #[test]
    fn reservations_are_byte_weighted_and_clone_owned() {
        let budget = MemoryBudget::new(NonZeroU64::new(10).unwrap());
        let first = budget.try_reserve(6).unwrap();
        let cloned = first.clone();
        assert_eq!(budget.snapshot().reserved_bytes, 6);
        assert!(matches!(
            budget.try_reserve(5),
            Err(MemoryBudgetError::Exhausted {
                requested_bytes: 5,
                reserved_bytes: 6,
                limit_bytes: 10,
            })
        ));
        drop(first);
        assert_eq!(budget.snapshot().reserved_bytes, 6);
        drop(cloned);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn concurrent_admission_never_exceeds_the_limit() {
        let budget = Arc::new(MemoryBudget::new(NonZeroU64::new(64).unwrap()));
        let threads = (0..32)
            .map(|_| {
                let budget = Arc::clone(&budget);
                thread::spawn(move || budget.try_reserve(16).ok())
            })
            .collect::<Vec<_>>();
        let permits = threads
            .into_iter()
            .filter_map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert!(permits.len() <= 4);
        assert_eq!(budget.snapshot().reserved_bytes, permits.len() as u64 * 16);
        drop(permits);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn arithmetic_overflow_is_typed_and_does_not_mutate_the_budget() {
        let budget = MemoryBudget::new(NonZeroU64::new(u64::MAX).unwrap());
        let permit = budget.try_reserve(u64::MAX).unwrap();
        assert!(matches!(
            budget.try_reserve(1),
            Err(MemoryBudgetError::Overflow {
                requested_bytes: 1,
                reserved_bytes: u64::MAX,
            })
        ));
        assert_eq!(budget.snapshot().reserved_bytes, u64::MAX);
        drop(permit);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
    }
}
