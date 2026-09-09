//! Byte-weighted admission for compressed input retained by incremental decoders.

use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::{Arc, Mutex};

/// Default bound on independently retained transport ranges across incremental streams.
pub const DEFAULT_INCREMENTAL_INPUT_SPANS: usize = 1 << 20;

/// A point-in-time view of shared incremental-input retention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IncrementalInputBudgetSnapshot {
    pub limit_bytes: u64,
    pub reserved_bytes: u64,
    pub available_bytes: u64,
    pub limit_spans: usize,
    pub reserved_spans: usize,
    pub available_spans: usize,
}

/// Failure to retain another compressed-input range without waiting for capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum IncrementalInputBudgetError {
    #[error(
        "incremental decode input has {reserved_spans}/{limit_spans} transport spans already retained"
    )]
    SpanLimit {
        reserved_spans: usize,
        limit_spans: usize,
    },
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
    limit_spans: usize,
    reserved: Mutex<RetainedInput>,
}

#[derive(Default)]
struct RetainedInput {
    bytes: u64,
    spans: usize,
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
        Self::with_limits(
            limit_bytes,
            NonZeroUsize::new(DEFAULT_INCREMENTAL_INPUT_SPANS).expect("nonzero default span limit"),
        )
    }

    /// Bounds logical compressed bytes and the number of independently retained input ranges.
    #[must_use]
    pub fn with_limits(limit_bytes: NonZeroU64, limit_spans: NonZeroUsize) -> Self {
        Self {
            inner: Arc::new(IncrementalInputBudgetInner {
                limit_bytes: limit_bytes.get(),
                limit_spans: limit_spans.get(),
                reserved: Mutex::new(RetainedInput::default()),
            }),
        }
    }

    pub(crate) fn try_reserve(
        &self,
        bytes: u64,
    ) -> Result<IncrementalInputPermit, IncrementalInputBudgetError> {
        let mut reserved = self
            .inner
            .reserved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let next =
            reserved
                .bytes
                .checked_add(bytes)
                .ok_or(IncrementalInputBudgetError::Overflow {
                    requested_bytes: bytes,
                    reserved_bytes: reserved.bytes,
                })?;
        if next > self.inner.limit_bytes {
            return Err(IncrementalInputBudgetError::Exhausted {
                requested_bytes: bytes,
                reserved_bytes: reserved.bytes,
                limit_bytes: self.inner.limit_bytes,
            });
        }
        if reserved.spans == self.inner.limit_spans {
            return Err(IncrementalInputBudgetError::SpanLimit {
                reserved_spans: reserved.spans,
                limit_spans: self.inner.limit_spans,
            });
        }
        reserved.bytes = next;
        reserved.spans += 1;
        Ok(IncrementalInputPermit {
            reservation: Arc::new(IncrementalInputReservation {
                budget: Arc::clone(&self.inner),
                bytes,
            }),
        })
    }

    #[must_use]
    pub fn snapshot(&self) -> IncrementalInputBudgetSnapshot {
        let reserved = self
            .inner
            .reserved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        IncrementalInputBudgetSnapshot {
            limit_bytes: self.inner.limit_bytes,
            reserved_bytes: reserved.bytes,
            available_bytes: self.inner.limit_bytes - reserved.bytes,
            limit_spans: self.inner.limit_spans,
            reserved_spans: reserved.spans,
            available_spans: self.inner.limit_spans - reserved.spans,
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
    bytes: u64,
}

impl Drop for IncrementalInputReservation {
    fn drop(&mut self) {
        let mut reserved = self
            .budget
            .reserved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reserved.bytes -= self.bytes;
        reserved.spans -= 1;
    }
}

/// Immutable shared ownership token for one admitted transport span.
#[derive(Clone)]
pub(crate) struct IncrementalInputPermit {
    reservation: Arc<IncrementalInputReservation>,
}

impl IncrementalInputPermit {
    pub(crate) fn bytes(&self) -> u64 {
        self.reservation.bytes
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
    fn immutable_span_permits_share_charges_without_retaining_future_input() {
        let budget = IncrementalInputBudget::with_limits(
            NonZeroU64::new(10).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        );
        let first = budget.try_reserve(3).unwrap();
        let preview = first.clone();
        let second = budget.try_reserve(4).unwrap();
        assert_eq!(preview.bytes(), 3);
        assert_eq!(budget.snapshot().reserved_bytes, 7);
        assert_eq!(budget.snapshot().reserved_spans, 2);
        assert!(matches!(
            budget.try_reserve(4),
            Err(IncrementalInputBudgetError::Exhausted { .. })
        ));
        assert!(matches!(
            budget.try_reserve(1),
            Err(IncrementalInputBudgetError::SpanLimit { .. })
        ));
        assert_eq!(budget.snapshot().reserved_bytes, 7);
        drop(first);
        drop(second);
        assert_eq!(budget.snapshot().reserved_bytes, 3);
        assert_eq!(budget.snapshot().reserved_spans, 1);
        let retry = budget.try_reserve(7).unwrap();
        drop(preview);
        assert_eq!(budget.snapshot().reserved_bytes, 7);
        drop(retry);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
        assert_eq!(budget.snapshot().reserved_spans, 0);
    }

    #[test]
    fn concurrent_stream_admission_never_exceeds_the_shared_limit() {
        let budget = IncrementalInputBudget::new(NonZeroU64::new(64).unwrap());
        let threads = (0..32)
            .map(|_| {
                let budget = budget.clone();
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
}
