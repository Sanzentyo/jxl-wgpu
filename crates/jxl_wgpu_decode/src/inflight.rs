use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

#[derive(Default)]
struct State {
    active_slots: usize,
    next_waiter_id: u64,
    waiters: BTreeMap<u64, Waker>,
    /// `poll_acquire` is used by one sequential decode session. Replacing its latest waker keeps
    /// manual polling bounded even when callers create and abandon futures with distinct wakers.
    direct_waiter: Option<Waker>,
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
    pub fn active_slots(&self) -> usize {
        lock_unpoisoned(&self.inner.state).active_slots
    }

    #[must_use]
    pub fn try_acquire(&self) -> Option<InFlightPermit> {
        let mut state = lock_unpoisoned(&self.inner.state);
        if state.active_slots >= self.inner.limit {
            return None;
        }
        state.active_slots += 1;
        Some(InFlightPermit {
            inner: Some(Arc::clone(&self.inner)),
        })
    }

    pub fn poll_acquire(&self, context: &mut Context<'_>) -> Poll<InFlightPermit> {
        let mut state = lock_unpoisoned(&self.inner.state);
        if state.active_slots < self.inner.limit {
            state.active_slots += 1;
            state.direct_waiter = None;
            return Poll::Ready(InFlightPermit {
                inner: Some(Arc::clone(&self.inner)),
            });
        }
        state.direct_waiter = Some(context.waker().clone());
        Poll::Pending
    }

    pub const fn acquire(&self) -> Acquire<'_> {
        Acquire {
            limiter: self,
            waiter_id: None,
        }
    }

    fn poll_registered(
        &self,
        waiter_id: &mut Option<u64>,
        context: &mut Context<'_>,
    ) -> Poll<InFlightPermit> {
        let mut state = lock_unpoisoned(&self.inner.state);
        if state.active_slots < self.inner.limit {
            state.active_slots += 1;
            if let Some(waiter_id) = waiter_id.take() {
                state.waiters.remove(&waiter_id);
            }
            return Poll::Ready(InFlightPermit {
                inner: Some(Arc::clone(&self.inner)),
            });
        }
        let id = match *waiter_id {
            Some(id) => id,
            None => {
                let id = next_waiter_id(&mut state);
                *waiter_id = Some(id);
                id
            }
        };
        state.waiters.insert(id, context.waker().clone());
        Poll::Pending
    }
}

impl fmt::Debug for InFlightLimiter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InFlightLimiter")
            .field("limit", &self.limit())
            .field("active_slots", &self.active_slots())
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
        let (waiters, direct_waiter) = {
            let mut state = lock_unpoisoned(&inner.state);
            debug_assert!(state.active_slots > 0);
            state.active_slots -= 1;
            (
                std::mem::take(&mut state.waiters),
                state.direct_waiter.take(),
            )
        };
        for waiter in waiters.into_values() {
            waiter.wake();
        }
        if let Some(waiter) = direct_waiter {
            waiter.wake();
        }
    }
}

#[must_use = "futures do nothing unless polled"]
pub struct Acquire<'limiter> {
    limiter: &'limiter InFlightLimiter,
    waiter_id: Option<u64>,
}

impl Future for Acquire<'_> {
    type Output = InFlightPermit;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let acquire = self.get_mut();
        acquire
            .limiter
            .poll_registered(&mut acquire.waiter_id, context)
    }
}

impl Drop for Acquire<'_> {
    fn drop(&mut self) {
        let Some(waiter_id) = self.waiter_id.take() else {
            return;
        };
        lock_unpoisoned(&self.limiter.inner.state)
            .waiters
            .remove(&waiter_id);
    }
}

fn next_waiter_id(state: &mut State) -> u64 {
    loop {
        let candidate = state.next_waiter_id;
        state.next_waiter_id = state.next_waiter_id.wrapping_add(1);
        if !state.waiters.contains_key(&candidate) {
            return candidate;
        }
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Wake, Waker};

    use super::*;

    #[derive(Default)]
    struct CountWake(AtomicUsize);

    impl Wake for CountWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn abandoned_acquire_futures_remove_their_registered_waiters() {
        let limiter = InFlightLimiter::new(NonZeroUsize::new(1).unwrap());
        let permit = limiter.try_acquire().unwrap();

        for _ in 0..10_000 {
            let waker = Waker::from(Arc::new(CountWake::default()));
            let mut context = Context::from_waker(&waker);
            let mut acquire = limiter.acquire();
            assert!(matches!(
                Pin::new(&mut acquire).poll(&mut context),
                Poll::Pending
            ));
            assert_eq!(lock_unpoisoned(&limiter.inner.state).waiters.len(), 1);
            drop(acquire);
            assert!(lock_unpoisoned(&limiter.inner.state).waiters.is_empty());
        }

        drop(permit);
        assert_eq!(limiter.active_slots(), 0);
    }

    #[test]
    fn direct_manual_polling_keeps_only_the_latest_waker() {
        let limiter = InFlightLimiter::new(NonZeroUsize::new(1).unwrap());
        let permit = limiter.try_acquire().unwrap();
        for _ in 0..10_000 {
            let waker = Waker::from(Arc::new(CountWake::default()));
            let mut context = Context::from_waker(&waker);
            assert!(matches!(limiter.poll_acquire(&mut context), Poll::Pending));
            assert!(
                lock_unpoisoned(&limiter.inner.state)
                    .direct_waiter
                    .is_some()
            );
        }
        drop(permit);
        assert!(
            lock_unpoisoned(&limiter.inner.state)
                .direct_waiter
                .is_none()
        );
    }
}
