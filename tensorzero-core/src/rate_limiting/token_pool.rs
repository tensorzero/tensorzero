//! In-memory token pool for pre-borrowing rate limit tokens.
//!
//! This module implements an adaptive pre-borrowing strategy to reduce database contention
//! at scale. Instead of hitting Postgres for every request, we pre-borrow tokens into an
//! in-memory pool and serve requests locally.
//!
//! Key features:
//! - Adaptive borrow sizing based on P99 historical usage (5-minute rolling window)
//! - Maximum borrow cap at 25% of capacity for fairness
//! - Threshold-based replenishment when pool drops below 20%
//! - Graceful shutdown with token return to database

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::Notify;

use super::usage_histogram::RollingUsageTracker;
use super::{ActiveRateLimit, RateLimit, RateLimitResource};

/// Duration for the P99 usage tracking rolling window
const P99_WINDOW_DURATION: Duration = Duration::from_secs(5 * 60); // 5 minutes

/// Replenishment threshold: replenish when available drops below this fraction of borrowed
const REPLENISHMENT_THRESHOLD: f64 = 0.20;

/// Minimum borrow amount to prevent thrashing
const MIN_BORROW_FLOOR: u64 = 1;

/// Adaptive borrow cap thresholds based on observed utilization.
/// When capacity is tight, we borrow less to reduce token stranding across instances.
const ADAPTIVE_CAP_HIGH_UTIL: f64 = 0.05; // > 80% utilization
const ADAPTIVE_CAP_MED_UTIL: f64 = 0.10; // 50-80% utilization
const ADAPTIVE_CAP_LOW_UTIL: f64 = 0.25; // < 50% utilization

/// Number of requests to process in direct mode before switching to pooled mode.
/// This allows the histogram to build up accurate P99 data before pre-borrowing.
/// We use 100 requests as this provides enough samples for statistically meaningful P99.
const WARMUP_REQUEST_COUNT: u64 = 100;

/// Maximum time to stay in warm-up mode before switching to pooled mode.
/// For low-traffic functions, 100 requests could take hours; this timeout ensures
/// we switch to pooled mode within a reasonable time (5 minutes).
const WARMUP_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// Target duration (in seconds) that a single borrow should last.
/// At 1.0, we aim to make at most one database round-trip per second per pool.
const TARGET_BORROW_SECONDS: f64 = 1.0;

/// In-memory token pool for a single rate limit key.
///
/// This pool tracks:
/// - Available tokens in the local pool
/// - Total borrowed from DB (for return on shutdown)
/// - Total actually consumed (for accurate accounting)
/// - Usage distribution for P99 calculation (via bucketed histogram)
/// - Request count for warm-up phase tracking
/// - Last observed tickets_remaining (for adaptive borrow cap)
#[derive(Debug)]
pub(super) struct TokenPool {
    /// Currently available tokens in the local pool (can go negative temporarily)
    available: AtomicI64,
    /// Total tokens borrowed from DB that haven't been returned
    borrowed_from_db: AtomicU64,
    /// Total tokens actually consumed from this pool
    used_from_pool: AtomicU64,
    /// Number of requests processed (for warm-up tracking)
    request_count: AtomicU64,
    /// Time when the pool was created (for warm-up timeout)
    created_at: Instant,
    /// Last observed tickets_remaining from DB (for adaptive borrow cap).
    /// Updated on every DB response (warm-up and replenishment).
    last_tickets_remaining: AtomicU64,
    /// Usage tracker for P99 calculation (lock-free histogram)
    usage_tracker: RollingUsageTracker,
    /// Last time we replenished from DB
    last_replenish: Mutex<Instant>,
    /// The rate limit parameters for this pool
    pub(super) limit: Arc<RateLimit>,
    /// Notify when replenishment is complete (for waiters)
    replenish_notify: Notify,
    /// Whether a replenishment is currently in progress
    pub(super) replenish_in_progress: AtomicBool,
}

impl TokenPool {
    pub(super) fn new(limit: Arc<RateLimit>) -> Self {
        let now = Instant::now();
        let capacity = limit.capacity;
        Self {
            available: AtomicI64::new(0),
            borrowed_from_db: AtomicU64::new(0),
            used_from_pool: AtomicU64::new(0),
            request_count: AtomicU64::new(0),
            created_at: now,
            // Initialize to capacity (optimistic: assume no utilization until we learn otherwise)
            last_tickets_remaining: AtomicU64::new(capacity),
            usage_tracker: RollingUsageTracker::new(P99_WINDOW_DURATION),
            last_replenish: Mutex::new(now),
            limit,
            replenish_notify: Notify::new(),
            replenish_in_progress: AtomicBool::new(false),
        }
    }

    /// Check if the pool is still in warm-up phase.
    ///
    /// During warm-up, requests go directly to the DB while building histogram data.
    /// Warm-up ends when EITHER:
    /// - We've processed WARMUP_REQUEST_COUNT requests (100), OR
    /// - WARMUP_TIMEOUT has elapsed (5 minutes) AND we have at least one request
    ///
    /// The time-based fallback ensures low-traffic functions don't stay in warm-up forever,
    /// but we never exit warm-up with zero data (that would cause degenerate borrowing).
    pub(super) fn is_warming_up(&self) -> bool {
        let request_count = self.request_count.load(Ordering::Acquire);
        if request_count >= WARMUP_REQUEST_COUNT {
            return false;
        }

        // Never exit warm-up with zero data - we'd have no histogram for borrow sizing
        if request_count == 0 {
            return true;
        }

        // Time-based fallback: exit warm-up after WARMUP_TIMEOUT even with few requests
        self.created_at.elapsed() < WARMUP_TIMEOUT
    }

    /// Increment the request count (called for each request during warm-up).
    pub(super) fn increment_request_count(&self) {
        self.request_count.fetch_add(1, Ordering::Release);
    }

    /// Update the last observed tickets_remaining from a DB response.
    /// This is used to compute the adaptive borrow cap based on utilization.
    pub(super) fn update_tickets_remaining(&self, tickets_remaining: u64) {
        self.last_tickets_remaining
            .store(tickets_remaining, Ordering::Release);
    }

    /// Get current available tokens in the pool.
    /// Used for pre-checking if pooled consumption is viable before replenishment.
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

    /// Record usage for P99 tracking - O(1), lock-free.
    pub(super) fn record_usage(&self, tokens: u64, model_inferences: u64) {
        self.usage_tracker.record(tokens, model_inferences);
    }

    /// Calculate P99 usage for the given resource type - O(NUM_BUCKETS).
    fn calculate_p99(&self, resource: RateLimitResource) -> Option<u64> {
        self.usage_tracker.p99(resource)
    }

    /// Check if replenishment is needed based on threshold
    pub(super) fn needs_replenishment(&self) -> bool {
        let available = self.available.load(Ordering::Acquire);
        let borrowed = self.borrowed_from_db.load(Ordering::Acquire);

        if borrowed == 0 {
            return true; // Never borrowed, need initial borrow
        }

        // Replenish when available drops below threshold of borrowed
        (available as f64) < (borrowed as f64) * REPLENISHMENT_THRESHOLD
    }

    /// Calculate how many tokens to borrow based on P99 usage, request rate, and utilization.
    ///
    /// The goal is to borrow enough tokens to last ~1 second of traffic, so we make
    /// at most one database round-trip per second per pool. The formula is:
    ///
    /// ```text
    /// borrow_amount = P99_per_request × request_rate_per_second × TARGET_BORROW_SECONDS
    /// ```
    ///
    /// The borrow amount is capped based on observed utilization:
    /// - < 50% utilization: cap at 25% of capacity (aggressive)
    /// - 50-80% utilization: cap at 10% of capacity (moderate)
    /// - > 80% utilization: cap at 5% of capacity (conservative)
    ///
    /// This adaptive cap reduces token stranding across instances when capacity is tight.
    pub(super) fn calculate_borrow_amount(&self) -> u64 {
        let capacity = self.limit.capacity;
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

        let max_borrow = (capacity as f64 * adaptive_cap) as u64;

        // Get P99 usage per request - should be available after warm-up phase
        let p99_per_request = self
            .calculate_p99(self.limit.resource)
            .unwrap_or(MIN_BORROW_FLOOR);

        // Get request rate per second
        let request_rate = self.usage_tracker.request_rate_per_second();

        // Calculate target: enough tokens for TARGET_BORROW_SECONDS of traffic
        // If rate is very low (< 1 req/s), ensure we borrow at least P99
        let target = if request_rate < 1.0 {
            p99_per_request
        } else {
            (p99_per_request as f64 * request_rate * TARGET_BORROW_SECONDS) as u64
        };

        // Apply caps and floor
        target.max(MIN_BORROW_FLOOR).min(max_borrow)
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
    pub(super) fn adjust_usage(&self, estimated: u64, actual: u64) {
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
        // If equal, no adjustment needed
    }

    /// Wait for replenishment to complete
    pub(super) async fn wait_for_replenishment(&self) {
        self.replenish_notify.notified().await;
    }

    /// Signal that replenishment is complete
    pub(super) fn signal_replenishment_complete(&self) {
        self.replenish_in_progress.store(false, Ordering::Release);
        self.replenish_notify.notify_waiters();
    }
}

/// Manager for all token pools across rate limit keys.
///
/// The TokenPoolManager coordinates:
/// - Creating pools for new rate limit keys
/// - Providing access to individual pools
#[derive(Debug, Default)]
pub(super) struct TokenPoolRegistry {
    /// Per-key token pools (keyed by the string representation of ActiveRateLimitKey)
    pools: DashMap<String, Arc<TokenPool>>,
}

impl TokenPoolRegistry {
    pub(super) fn new() -> Self {
        Self {
            pools: DashMap::new(),
        }
    }

    /// Returns true if there are no pools tracked by this registry.
    pub(super) fn is_empty(&self) -> bool {
        self.pools.is_empty()
    }

    /// Get or create a pool for the given active limit
    pub(super) fn get_or_create(&self, active_limit: &ActiveRateLimit) -> Arc<TokenPool> {
        let key = active_limit.key.0.clone();
        self.pools
            .entry(key)
            .or_insert_with(|| Arc::new(TokenPool::new(active_limit.limit.clone())))
            .clone()
    }

    /// Iterate over all pools (for shutdown)
    pub(super) fn iter(&self) -> impl Iterator<Item = (String, Arc<TokenPool>)> + '_ {
        self.pools
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate_limiting::RateLimitInterval;

    fn make_test_limit(capacity: u64) -> Arc<RateLimit> {
        Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Minute,
            capacity,
            refill_rate: 10,
        })
    }

    #[test]
    fn test_token_pool_try_consume_success() {
        let pool = TokenPool::new(make_test_limit(1000));
        pool.add_tokens(100);

        assert!(pool.try_consume(50), "Should consume 50 tokens");
        assert_eq!(pool.available.load(Ordering::Relaxed), 50);
        assert_eq!(pool.used_from_pool.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn test_token_pool_try_consume_insufficient() {
        let pool = TokenPool::new(make_test_limit(1000));
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
        let pool = TokenPool::new(make_test_limit(1000));
        pool.add_tokens(100);
        pool.try_consume(50);

        pool.rollback_consume(50);

        assert_eq!(pool.available.load(Ordering::Relaxed), 100);
        assert_eq!(pool.used_from_pool.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_token_pool_needs_replenishment() {
        let pool = TokenPool::new(make_test_limit(1000));

        // Initially needs replenishment (never borrowed)
        assert!(
            pool.needs_replenishment(),
            "Should need replenishment initially"
        );

        pool.add_tokens(100);
        assert!(
            !pool.needs_replenishment(),
            "Should not need replenishment after adding tokens"
        );

        // Consume most tokens
        pool.try_consume(85);
        assert!(
            pool.needs_replenishment(),
            "Should need replenishment when below threshold"
        );
    }

    #[test]
    fn test_token_pool_unused_tokens() {
        let pool = TokenPool::new(make_test_limit(1000));
        pool.add_tokens(100);
        pool.try_consume(30);

        assert_eq!(pool.unused_tokens(), 70, "Should have 70 unused tokens");
    }

    #[test]
    fn test_token_pool_adjust_usage() {
        let pool = TokenPool::new(make_test_limit(1000));
        pool.add_tokens(100);
        pool.try_consume(50); // used_from_pool = 50

        // Actual usage was less than estimated
        pool.adjust_usage(50, 30);
        assert_eq!(pool.used_from_pool.load(Ordering::Relaxed), 30);

        // Actual usage was more than estimated
        pool.adjust_usage(30, 45);
        assert_eq!(pool.used_from_pool.load(Ordering::Relaxed), 45);
    }

    #[test]
    fn test_token_pool_registry() {
        let registry = TokenPoolRegistry::new();
        assert!(registry.is_empty());

        let limit = make_test_limit(1000);
        let active_limit = ActiveRateLimit::new(limit.clone(), vec![]).unwrap();

        let pool1 = registry.get_or_create(&active_limit);
        let pool2 = registry.get_or_create(&active_limit);

        assert!(
            Arc::ptr_eq(&pool1, &pool2),
            "Should return same pool for same key"
        );
        assert!(!registry.is_empty());
    }
}
