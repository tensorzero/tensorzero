//! In-memory token pool for pre-borrowing rate limit tokens.
//!
//! This module implements an adaptive pre-borrowing strategy to reduce database contention
//! at scale. Instead of hitting Postgres for every request, we pre-borrow tokens into an
//! in-memory pool and serve requests locally.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::RateLimit;
use super::batching_buffer::{BatchingBuffer, TimerFlushStrategy};
use super::exhaustion_backoff::ExhaustionBackoff;
use super::usage_tracker::PerSecondUsageTracker;

// ============================================================================
// Constants
// ============================================================================

/// Adaptive borrow cap thresholds based on observed utilization.
/// When capacity is tight, we borrow less to reduce token stranding across instances.
const ADAPTIVE_CAP_HIGH_UTIL: f64 = 0.05; // > 80% utilization
const ADAPTIVE_CAP_MED_UTIL: f64 = 0.10; // 50-80% utilization
const ADAPTIVE_CAP_LOW_UTIL: f64 = 0.25; // < 50% utilization

/// In-memory token pool for a single rate limit key.
///
/// This pool tracks:
/// - Available tokens in the local pool
/// - Total borrowed from DB (for return on shutdown)
/// - Total actually consumed (for accurate accounting)
/// - Per-second usage for P99 calculation
/// - Last observed tickets_remaining (for adaptive borrow cap)
/// - Exhaustion backoff state (for rate-limiting DB spam when exhausted)
pub(super) struct TokenPool {
    /// Currently available tokens in the local pool (can go negative temporarily)
    available: AtomicI64,
    /// Total tokens borrowed from DB that haven't been returned
    borrowed_from_db: AtomicU64,
    /// Total tokens actually consumed from this pool
    used_from_pool: AtomicU64,
    /// Last observed tickets_remaining from DB (for adaptive borrow cap).
    /// Updated on every DB response.
    last_tickets_remaining: AtomicU64,
    /// Per-second usage tracker for P99 calculation.
    /// Tracks total tokens consumed per second over a 120-second rolling window.
    per_second_tracker: PerSecondUsageTracker,
    /// Last time we replenished from DB
    last_replenish: Mutex<Instant>,
    /// The rate limit parameters for this pool
    pub(super) limit: Arc<RateLimit>,
    /// Batching buffer for cold start and burst handling.
    /// Groups concurrent requests to reduce DB calls.
    pub(super) batching_buffer: BatchingBuffer,
    /// Exhaustion backoff state.
    /// When DB reports no tokens available, we enter backoff to avoid DB spam.
    pub(super) exhaustion_backoff: ExhaustionBackoff,
}

impl std::fmt::Debug for TokenPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenPool")
            .field("available", &self.available.load(Ordering::Relaxed))
            .field(
                "borrowed_from_db",
                &self.borrowed_from_db.load(Ordering::Relaxed),
            )
            .field(
                "used_from_pool",
                &self.used_from_pool.load(Ordering::Relaxed),
            )
            .field("limit", &self.limit)
            .finish_non_exhaustive()
    }
}

impl TokenPool {
    pub(super) fn new(limit: Arc<RateLimit>, buffer_timeout: std::time::Duration) -> Self {
        let now = Instant::now();
        let capacity = limit.capacity;
        Self {
            available: AtomicI64::new(0),
            borrowed_from_db: AtomicU64::new(0),
            used_from_pool: AtomicU64::new(0),
            // Initialize to capacity (optimistic: assume no utilization until we learn otherwise)
            last_tickets_remaining: AtomicU64::new(capacity),
            per_second_tracker: PerSecondUsageTracker::new(),
            last_replenish: Mutex::new(now),
            limit,
            batching_buffer: BatchingBuffer::new(TimerFlushStrategy::new(buffer_timeout)),
            exhaustion_backoff: ExhaustionBackoff::default(),
        }
    }

    /// Update the last observed tickets_remaining from a DB response.
    /// This is used to compute the adaptive borrow cap based on utilization.
    pub(super) fn update_tickets_remaining(&self, tickets_remaining: u64) {
        self.last_tickets_remaining
            .store(tickets_remaining, Ordering::Release);
    }

    /// Get current available tokens in the pool.
    /// May be useful for debugging or observability.
    #[expect(dead_code)]
    pub(super) fn available(&self) -> i64 {
        self.available.load(Ordering::Acquire)
    }

    /// Try to consume tokens from the local pool.
    /// Returns true if successful, false if insufficient tokens.
    pub(super) fn try_consume(&self, amount: u64) -> bool {
        let amount_i64 = amount as i64;
        loop {
            let current = self.available.load(Ordering::Acquire);
            if current < amount_i64 {
                return false;
            }
            if self
                .available
                .compare_exchange_weak(
                    current,
                    current - amount_i64,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                self.used_from_pool.fetch_add(amount, Ordering::Relaxed);
                return true;
            }
        }
    }

    /// Rollback a consumption (used when multi-pool consumption fails partway)
    pub(super) fn rollback_consume(&self, amount: u64) {
        self.available.fetch_add(amount as i64, Ordering::Release);
        self.used_from_pool.fetch_sub(amount, Ordering::Relaxed);
    }

    /// Record estimated usage to per-second tracker for P99 calculation.
    ///
    /// Called at consume time (before request runs) so P99 reflects demand
    /// when requests arrive, not after they complete.
    ///
    /// Returns the epoch when the usage was recorded, which can be used later
    /// to adjust the record when actual usage is known.
    pub(super) fn record_estimated_usage(&self, estimated_tokens: u64) -> u32 {
        self.per_second_tracker.record_with_epoch(estimated_tokens)
    }

    /// Calculate how many tokens to borrow based on the deficit between target and current pool.
    ///
    /// Uses deficit-based formula:
    /// ```text
    /// target = max(queued_tokens, p99_tokens_per_second)
    /// borrow_amount = target - current_available
    /// ```
    ///
    /// This ensures we only borrow what's needed to reach the target pool level,
    /// avoiding over-borrowing when tokens have been refunded via `adjust_usage()`.
    ///
    /// The borrow amount is capped based on observed utilization:
    /// - < 50% utilization: cap at 25% of capacity (aggressive)
    /// - 50-80% utilization: cap at 10% of capacity (moderate)
    /// - > 80% utilization: cap at 5% of capacity (conservative)
    ///
    /// We always borrow at least enough to satisfy queued waiters (`min_borrow`).
    pub(super) fn calculate_borrow_amount(&self, queued_tokens: u64) -> u64 {
        let capacity = self.limit.capacity;
        let current_available = self.available.load(Ordering::Acquire);
        let tickets_remaining = self.last_tickets_remaining.load(Ordering::Acquire);

        // Compute adaptive cap based on observed utilization
        let utilization = if capacity > 0 {
            1.0 - (tickets_remaining as f64 / capacity as f64)
        } else {
            0.0
        };

        let adaptive_cap = if utilization > 0.8 {
            ADAPTIVE_CAP_HIGH_UTIL // 5%
        } else if utilization > 0.5 {
            ADAPTIVE_CAP_MED_UTIL // 10%
        } else {
            ADAPTIVE_CAP_LOW_UTIL // 25%
        };

        // Minimum borrow: enough to satisfy current waiters
        // (available can be negative, so handle that case)
        let min_borrow = if current_available >= 0 {
            queued_tokens.saturating_sub(current_available as u64)
        } else {
            // Pool is negative - need to cover deficit plus queued
            queued_tokens.saturating_add(-current_available as u64)
        };

        // Target pool level = max(queued, p99)
        let p99 = self.per_second_tracker.p99().unwrap_or(0);
        let target = std::cmp::max(queued_tokens, p99);

        // Ideal borrow = deficit to reach target level
        let ideal_borrow = if current_available >= 0 {
            target.saturating_sub(current_available as u64)
        } else {
            target + (-current_available as u64)
        };

        // Cap speculative borrowing, but always borrow at least min_borrow
        // Never request more than capacity (DB would reject such requests).
        let max_speculative = (capacity as f64 * adaptive_cap) as u64;
        ideal_borrow
            .min(max_speculative)
            .max(min_borrow)
            .min(capacity)
    }

    /// Add tokens to the pool (after successful DB borrow)
    pub(super) fn add_tokens(&self, amount: u64) {
        self.available.fetch_add(amount as i64, Ordering::Release);
        self.borrowed_from_db.fetch_add(amount, Ordering::Release);
        if let Ok(mut last) = self.last_replenish.lock() {
            *last = Instant::now();
        }
    }

    /// Calculate unused tokens (for return on shutdown)
    pub(super) fn unused_tokens(&self) -> u64 {
        let borrowed = self.borrowed_from_db.load(Ordering::Acquire);
        let used = self.used_from_pool.load(Ordering::Acquire);
        borrowed.saturating_sub(used)
    }

    /// Adjust usage accounting based on actual vs estimated usage.
    /// If actual > estimated, we need to account for more usage (both available and used_from_pool).
    /// If actual < estimated, we can recover some usage (restore available and reduce used_from_pool).
    ///
    /// Also adjusts the P99 tracker to reflect actual usage instead of estimated.
    /// The `recorded_epoch` should be the value returned by `record_estimated_usage`.
    pub(super) fn adjust_usage(&self, estimated: u64, actual: u64, recorded_epoch: Option<u32>) {
        if actual > estimated {
            // We used more than estimated - need to consume extra from pool
            let extra = actual - estimated;
            self.available.fetch_sub(extra as i64, Ordering::Release);
            self.used_from_pool.fetch_add(extra, Ordering::Release);
        } else if actual < estimated {
            // We used less than estimated - restore the difference to available
            let refund = estimated - actual;
            self.available.fetch_add(refund as i64, Ordering::Release);
            // Decrease used_from_pool (but don't go negative)
            let _ =
                self.used_from_pool
                    .fetch_update(Ordering::Release, Ordering::Acquire, |current| {
                        Some(current.saturating_sub(refund))
                    });
        }

        // Adjust P99 tracker to reflect actual usage instead of estimated
        if let Some(epoch) = recorded_epoch {
            self.per_second_tracker.adjust(epoch, estimated, actual);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate_limiting::{RateLimitInterval, RateLimitResource};
    use std::time::Duration;

    const TEST_BUFFER_TIMEOUT: Duration = Duration::from_millis(10);

    fn make_test_rate_limit(capacity: u64) -> Arc<RateLimit> {
        Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Minute,
            capacity,
            refill_rate: 10,
        })
    }

    #[test]
    fn test_token_pool_try_consume_success() {
        let pool = TokenPool::new(make_test_rate_limit(1000), TEST_BUFFER_TIMEOUT);
        pool.add_tokens(100);

        assert!(pool.try_consume(50), "Should consume 50 tokens");
        assert_eq!(pool.available.load(Ordering::Relaxed), 50);
        assert_eq!(pool.used_from_pool.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn test_token_pool_try_consume_insufficient() {
        let pool = TokenPool::new(make_test_rate_limit(1000), TEST_BUFFER_TIMEOUT);
        pool.add_tokens(30);

        assert!(
            !pool.try_consume(50),
            "Should not consume 50 tokens when only 30 available"
        );
        assert_eq!(pool.available.load(Ordering::Relaxed), 30);
        assert_eq!(pool.used_from_pool.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_token_pool_rollback() {
        let pool = TokenPool::new(make_test_rate_limit(1000), TEST_BUFFER_TIMEOUT);
        pool.add_tokens(100);
        pool.try_consume(50);

        pool.rollback_consume(50);

        assert_eq!(pool.available.load(Ordering::Relaxed), 100);
        assert_eq!(pool.used_from_pool.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_token_pool_unused_tokens() {
        let pool = TokenPool::new(make_test_rate_limit(1000), TEST_BUFFER_TIMEOUT);
        pool.add_tokens(100);
        pool.try_consume(30);

        assert_eq!(pool.unused_tokens(), 70, "Should have 70 unused tokens");
    }

    #[test]
    fn test_token_pool_adjust_usage() {
        let pool = TokenPool::new(make_test_rate_limit(1000), TEST_BUFFER_TIMEOUT);
        pool.add_tokens(100);
        pool.try_consume(50); // used_from_pool = 50

        // Actual usage was less than estimated
        pool.adjust_usage(50, 30, None);
        assert_eq!(pool.used_from_pool.load(Ordering::Relaxed), 30);

        // Actual usage was more than estimated
        pool.adjust_usage(30, 45, None);
        assert_eq!(pool.used_from_pool.load(Ordering::Relaxed), 45);
    }

    // ========================================================================
    // calculate_borrow_amount tests (deficit-based borrowing)
    // ========================================================================

    #[test]
    fn test_calculate_borrow_amount_cold_start() {
        // Cold start: no P99 data, pool empty -> borrow exactly queued
        let pool = TokenPool::new(make_test_rate_limit(1000), TEST_BUFFER_TIMEOUT);
        let borrow = pool.calculate_borrow_amount(100);
        assert_eq!(
            borrow, 100,
            "Cold start should borrow exactly queued tokens"
        );
    }

    #[test]
    fn test_calculate_borrow_amount_pool_has_tokens() {
        // Pool has 50 tokens, queued 100, no P99 -> borrow deficit (50)
        let pool = TokenPool::new(make_test_rate_limit(1000), TEST_BUFFER_TIMEOUT);
        pool.add_tokens(50);
        let borrow = pool.calculate_borrow_amount(100);
        assert_eq!(
            borrow, 50,
            "Should borrow deficit to satisfy waiters (100 - 50 = 50)"
        );
    }

    #[test]
    fn test_calculate_borrow_amount_pool_sufficient() {
        // Pool has 200 tokens, queued 50, no P99 -> borrow 0
        let pool = TokenPool::new(make_test_rate_limit(1000), TEST_BUFFER_TIMEOUT);
        pool.add_tokens(200);
        let borrow = pool.calculate_borrow_amount(50);
        assert_eq!(borrow, 0, "Should borrow 0 when pool already has enough");
    }

    #[test]
    fn test_calculate_borrow_amount_negative_pool() {
        // Pool is negative (-20), queued 100 -> borrow queued + deficit (120)
        let pool = TokenPool::new(make_test_rate_limit(1000), TEST_BUFFER_TIMEOUT);
        // Simulate negative pool by consuming more than available
        pool.available.store(-20, Ordering::Release);
        let borrow = pool.calculate_borrow_amount(100);
        assert_eq!(
            borrow, 120,
            "Should borrow queued + negative deficit (100 + 20 = 120)"
        );
    }

    #[test]
    fn test_calculate_borrow_amount_with_p99_higher() {
        // Pool has 50 tokens, queued 50, P99 = 150 -> borrow deficit to P99 (100)
        let pool = TokenPool::new(make_test_rate_limit(1000), TEST_BUFFER_TIMEOUT);
        pool.add_tokens(50);
        // Record P99 value once (P99 of a single sample is that sample)
        pool.per_second_tracker.record_with_epoch(150);
        let borrow = pool.calculate_borrow_amount(50);
        // P99 = 150, available = 50, so ideal_borrow = 100
        // min_borrow = max(0, 50 - 50) = 0
        // Result should be ideal_borrow capped by adaptive cap (25% of 1000 = 250)
        assert_eq!(
            borrow, 100,
            "Should borrow deficit to reach P99 level (150 - 50 = 100)"
        );
    }

    #[test]
    fn test_calculate_borrow_amount_capped_by_capacity() {
        // Very small capacity, ensure we don't exceed it
        let pool = TokenPool::new(make_test_rate_limit(10), TEST_BUFFER_TIMEOUT);
        let borrow = pool.calculate_borrow_amount(100);
        assert_eq!(borrow, 10, "Should not borrow more than capacity");
    }

    #[test]
    fn test_calculate_borrow_amount_small_top_up() {
        // Pool has 80 tokens, queued 50, P99 = 100 -> small top-up (20)
        let pool = TokenPool::new(make_test_rate_limit(1000), TEST_BUFFER_TIMEOUT);
        pool.add_tokens(80);
        // Record P99 value once (P99 of a single sample is that sample)
        pool.per_second_tracker.record_with_epoch(100);
        let borrow = pool.calculate_borrow_amount(50);
        // min_borrow = max(0, 50 - 80) = 0
        // ideal_borrow = 100 - 80 = 20
        assert_eq!(
            borrow, 20,
            "Should do small top-up to reach P99 (100 - 80 = 20)"
        );
    }
}
