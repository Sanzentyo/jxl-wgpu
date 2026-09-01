//! Byte-weighted admission for compressed input retained by incremental decoders.

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A point-in-time view of shared incremental-input retention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IncrementalInputBudgetSnapshot {
    pub limit_bytes: u64,
    pub reserved_bytes: u64,
    pub available_bytes: u64,
}

/// Failure to retain another compressed-input range without blocking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum IncrementalInputBudgetError {
    #[error(
        "incremental decode input needs {requested_bytes} more bytes with {reserved_bytes}/{limit_bytes} bytes already retained"
    )]
    Exhausted {
        requested_bytes: u64,
        reserved_bytes: u64,
        limit_bytes: u64,
    },
    #[error(
        "incremental decode input accounting overflowed while adding {requested_bytes} bytes to {reserved_bytes} retained bytes"
    )]
    Overflow {
        requested_bytes: u64,
        reserved_bytes: u64,
    },
}

struct IncrementalInputBudgetInner {
    limit_bytes: u64,
    reserved_bytes: AtomicU64,
}

/// Cloneable, runtime-independent budget shared by incremental decoder instances.
///
/// This budget is intentionally separate from the GPU allocation budget. A compressed host input
/// and the GPU upload populated from it are simultaneously live during submission, so charging
/// both to one limit could prevent an otherwise valid submission from ever making progress.
#[derive(Clone)]
pub struct IncrementalInputBudget {
    inner: Arc<IncrementalInputBudgetInner>,
}

impl IncrementalInputBudget {
    #[must_use]
    pub fn new(limit_bytes: NonZeroU64) -> Self {
        Self {
            inner: Arc::new(IncrementalInputBudgetInner {
                limit_bytes: limit_bytes.get(),
                reserved_bytes: AtomicU64::new(0),
            }),
        }
    }

    pub(crate) fn reserve_empty(&self) -> IncrementalInputPermit {
        IncrementalInputPermit {
            reservation: Arc::new(IncrementalInputReservation {
                budget: Arc::clone(&self.inner),
                bytes: AtomicU64::new(0),
            }),
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> IncrementalInputBudgetSnapshot {
        let reserved_bytes = self.inner.reserved_bytes.load(Ordering::Acquire);
        IncrementalInputBudgetSnapshot {
            limit_bytes: self.inner.limit_bytes,
            reserved_bytes,
            available_bytes: self.inner.limit_bytes.saturating_sub(reserved_bytes),
        }
    }
}

impl std::fmt::Debug for IncrementalInputBudget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("IncrementalInputBudget")
            .field("snapshot", &self.snapshot())
            .finish()
    }
}

struct IncrementalInputReservation {
    budget: Arc<IncrementalInputBudgetInner>,
    bytes: AtomicU64,
}

impl Drop for IncrementalInputReservation {
    fn drop(&mut self) {
        let bytes = self.bytes.load(Ordering::Acquire);
        let previous = self
            .budget
            .reserved_bytes
            .fetch_sub(bytes, Ordering::AcqRel);
        debug_assert!(previous >= bytes, "incremental input accounting underflow");
    }
}

/// Shared ownership token for one incrementally retained codestream.
#[derive(Clone)]
pub(crate) struct IncrementalInputPermit {
    reservation: Arc<IncrementalInputReservation>,
}

impl IncrementalInputPermit {
    pub(crate) fn try_grow(
        &self,
        additional_bytes: u64,
    ) -> Result<(), IncrementalInputBudgetError> {
        let mut reserved = self
            .reservation
            .budget
            .reserved_bytes
            .load(Ordering::Acquire);
        loop {
            let next = reserved.checked_add(additional_bytes).ok_or(
                IncrementalInputBudgetError::Overflow {
                    requested_bytes: additional_bytes,
                    reserved_bytes: reserved,
                },
            )?;
            if next > self.reservation.budget.limit_bytes {
                return Err(IncrementalInputBudgetError::Exhausted {
                    requested_bytes: additional_bytes,
                    reserved_bytes: reserved,
                    limit_bytes: self.reservation.budget.limit_bytes,
                });
            }
            match self
                .reservation
                .budget
                .reserved_bytes
                .compare_exchange_weak(reserved, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    self.reservation
                        .bytes
                        .fetch_add(additional_bytes, Ordering::AcqRel);
                    return Ok(());
                }
                Err(actual) => reserved = actual,
            }
        }
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.reservation.bytes.load(Ordering::Acquire)
    }
}

impl std::fmt::Debug for IncrementalInputPermit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("IncrementalInputPermit")
            .field("bytes", &self.bytes())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn one_growable_permit_tracks_exact_bytes_and_clone_lifetime() {
        let budget = IncrementalInputBudget::new(NonZeroU64::new(10).unwrap());
        let permit = budget.reserve_empty();
        permit.try_grow(3).unwrap();
        permit.try_grow(4).unwrap();
        assert_eq!(permit.bytes(), 7);
        assert_eq!(budget.snapshot().reserved_bytes, 7);

        let clone = permit.clone();
        drop(permit);
        assert_eq!(budget.snapshot().reserved_bytes, 7);
        assert!(matches!(
            clone.try_grow(4),
            Err(IncrementalInputBudgetError::Exhausted {
                requested_bytes: 4,
                reserved_bytes: 7,
                limit_bytes: 10,
            })
        ));
        assert_eq!(budget.snapshot().reserved_bytes, 7);
        drop(clone);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn concurrent_stream_admission_never_exceeds_the_shared_limit() {
        let budget = IncrementalInputBudget::new(NonZeroU64::new(64).unwrap());
        let threads = (0..32)
            .map(|_| {
                let budget = budget.clone();
                thread::spawn(move || {
                    let permit = budget.reserve_empty();
                    permit.try_grow(16).ok().map(|()| permit)
                })
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
}
