use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

#[derive(Default)]
struct State {
    in_flight: usize,
    waiters: VecDeque<Waker>,
}

struct Inner {
    limit: usize,
    state: Mutex<State>,
}

/// Output-agnostic bounded GPU in-flight limiter.
///
/// A frame lease retains one permit. Releasing it wakes every registered task so engines can
/// continue submission without polling or a runtime-specific channel.
#[derive(Clone)]
pub struct InFlightLimiter {
    inner: Arc<Inner>,
}

impl InFlightLimiter {
    #[must_use]
    pub fn new(limit: NonZeroUsize) -> Self {
        Self {
            inner: Arc::new(Inner {
                limit: limit.get(),
                state: Mutex::new(State::default()),
            }),
        }
    }

    #[must_use]
    pub fn limit(&self) -> usize {
        self.inner.limit
    }

    #[must_use]
    pub fn in_flight(&self) -> usize {
        lock_unpoisoned(&self.inner.state).in_flight
    }

    #[must_use]
    pub fn try_acquire(&self) -> Option<InFlightPermit> {
        let mut state = lock_unpoisoned(&self.inner.state);
        if state.in_flight >= self.inner.limit {
            return None;
        }
        state.in_flight += 1;
        Some(InFlightPermit {
            inner: Some(Arc::clone(&self.inner)),
        })
    }

    pub fn poll_acquire(&self, context: &mut Context<'_>) -> Poll<InFlightPermit> {
        let mut state = lock_unpoisoned(&self.inner.state);
        if state.in_flight < self.inner.limit {
            state.in_flight += 1;
            state
                .waiters
                .retain(|waiter| !waiter.will_wake(context.waker()));
            return Poll::Ready(InFlightPermit {
                inner: Some(Arc::clone(&self.inner)),
            });
        }
        if !state
            .waiters
            .iter()
            .any(|waiter| waiter.will_wake(context.waker()))
        {
            state.waiters.push_back(context.waker().clone());
        }
        Poll::Pending
    }

    pub const fn acquire(&self) -> Acquire<'_> {
        Acquire { limiter: self }
    }
}

impl fmt::Debug for InFlightLimiter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InFlightLimiter")
            .field("limit", &self.limit())
            .field("in_flight", &self.in_flight())
            .finish()
    }
}

/// One bounded GPU submission/output slot.
pub struct InFlightPermit {
    inner: Option<Arc<Inner>>,
}

impl fmt::Debug for InFlightPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InFlightPermit")
            .field("active", &self.inner.is_some())
            .finish()
    }
}

impl Drop for InFlightPermit {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        let waiters = {
            let mut state = lock_unpoisoned(&inner.state);
            debug_assert!(state.in_flight > 0);
            state.in_flight -= 1;
            std::mem::take(&mut state.waiters)
        };
        for waiter in waiters {
            waiter.wake();
        }
    }
}

#[must_use = "futures do nothing unless polled"]
pub struct Acquire<'limiter> {
    limiter: &'limiter InFlightLimiter,
}

impl Future for Acquire<'_> {
    type Output = InFlightPermit;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.limiter.poll_acquire(context)
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
