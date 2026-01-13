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

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::Notify;

use super::{
    ActiveRateLimit, ActiveRateLimitKey, FailedRateLimit, RateLimit, RateLimitResource,
    RateLimitingConfig,
};
use crate::db::{ConsumeTicketsRequest, RateLimitQueries, ReturnTicketsRequest};
use crate::error::{Error, ErrorDetails};

/// Duration for the P99 usage tracking rolling window
const P99_WINDOW_DURATION: Duration = Duration::from_secs(5 * 60); // 5 minutes

/// Replenishment threshold: replenish when available drops below this fraction of borrowed
const REPLENISHMENT_THRESHOLD: f64 = 0.20;

/// Maximum fraction of capacity that can be borrowed at once (fairness cap)
const MAX_BORROW_CAP: f64 = 0.25;

/// Fraction of capacity to borrow on cold start (no usage history)
const COLD_START_BORROW_PCT: f64 = 0.10;

/// A record of usage at a specific point in time
#[derive(Debug, Clone)]
struct UsageRecord {
    timestamp: Instant,
    tokens_used: u64,
    model_inferences_used: u64,
}

/// In-memory token pool for a single rate limit key.
///
/// This pool tracks:
/// - Available tokens in the local pool
/// - Total borrowed from DB (for return on shutdown)
/// - Total actually consumed (for accurate accounting)
/// - Usage history for P99 calculation
#[derive(Debug)]
struct TokenPool {
    /// Currently available tokens in the local pool (can go negative temporarily)
    available: AtomicI64,
    /// Total tokens borrowed from DB that haven't been returned
    borrowed_from_db: AtomicU64,
    /// Total tokens actually consumed from this pool
    used_from_pool: AtomicU64,
    /// Usage history for P99 calculation
    usage_history: Mutex<VecDeque<UsageRecord>>,
    /// Last time we replenished from DB
    last_replenish: Mutex<Instant>,
    /// The rate limit parameters for this pool
    limit: Arc<RateLimit>,
    /// Notify when replenishment is complete (for waiters)
    replenish_notify: Notify,
    /// Whether a replenishment is currently in progress
    replenish_in_progress: AtomicBool,
}

impl TokenPool {
    fn new(limit: Arc<RateLimit>) -> Self {
        Self {
            available: AtomicI64::new(0),
            borrowed_from_db: AtomicU64::new(0),
            used_from_pool: AtomicU64::new(0),
            usage_history: Mutex::new(VecDeque::new()),
            last_replenish: Mutex::new(Instant::now()),
            limit,
            replenish_notify: Notify::new(),
            replenish_in_progress: AtomicBool::new(false),
        }
    }

    /// Try to consume tokens from the local pool.
    /// Returns true if successful, false if insufficient tokens.
    fn try_consume(&self, amount: u64) -> bool {
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

    /// Record usage for P99 tracking
    fn record_usage(&self, tokens: u64, model_inferences: u64) {
        let record = UsageRecord {
            timestamp: Instant::now(),
            tokens_used: tokens,
            model_inferences_used: model_inferences,
        };
        if let Ok(mut history) = self.usage_history.lock() {
            history.push_back(record);
        }
    }

    /// Prune old usage records outside the rolling window
    fn prune_old_usage(&self) {
        let cutoff = Instant::now() - P99_WINDOW_DURATION;
        if let Ok(mut history) = self.usage_history.lock() {
            while let Some(front) = history.front() {
                if front.timestamp < cutoff {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    /// Calculate P99 usage for the given resource type
    fn calculate_p99(&self, resource: RateLimitResource) -> Option<u64> {
        self.prune_old_usage();
        let history = self.usage_history.lock().ok()?;

        if history.is_empty() {
            return None;
        }

        let mut usages: Vec<u64> = history
            .iter()
            .map(|r| match resource {
                RateLimitResource::Token => r.tokens_used,
                RateLimitResource::ModelInference => r.model_inferences_used,
            })
            .collect();

        usages.sort_unstable();

        // Calculate P99 index
        let p99_index = ((usages.len() as f64) * 0.99).ceil() as usize - 1;
        let p99_index = p99_index.min(usages.len() - 1);

        Some(usages[p99_index])
    }

    /// Check if replenishment is needed based on threshold
    fn needs_replenishment(&self) -> bool {
        let available = self.available.load(Ordering::Acquire);
        let borrowed = self.borrowed_from_db.load(Ordering::Acquire);

        if borrowed == 0 {
            return true; // Never borrowed, need initial borrow
        }

        // Replenish when available drops below threshold of borrowed
        (available as f64) < (borrowed as f64) * REPLENISHMENT_THRESHOLD
    }

    /// Calculate how many tokens to borrow based on P99 and caps
    fn calculate_borrow_amount(&self, min_floor: u64) -> u64 {
        let capacity = self.limit.capacity;

        // Calculate max based on fairness cap
        let max_borrow = ((capacity as f64) * MAX_BORROW_CAP) as u64;

        // Get P99 usage if available
        let p99 = self.calculate_p99(self.limit.resource);

        let target = match p99 {
            Some(p99_usage) => {
                // Use P99 as the target, capped at max
                p99_usage.min(max_borrow)
            }
            None => {
                // Cold start: use a conservative percentage of max
                ((max_borrow as f64) * COLD_START_BORROW_PCT) as u64
            }
        };

        // Apply minimum floor
        target.max(min_floor)
    }

    /// Add tokens to the pool (after successful DB borrow)
    fn add_tokens(&self, amount: u64) {
        self.available.fetch_add(amount as i64, Ordering::Release);
        self.borrowed_from_db.fetch_add(amount, Ordering::Release);
        if let Ok(mut last) = self.last_replenish.lock() {
            *last = Instant::now();
        }
    }

    /// Calculate unused tokens (for return on shutdown)
    fn unused_tokens(&self) -> u64 {
        let borrowed = self.borrowed_from_db.load(Ordering::Acquire);
        let used = self.used_from_pool.load(Ordering::Acquire);
        borrowed.saturating_sub(used)
    }
}

/// Manager for all token pools across rate limit keys.
///
/// The TokenPoolManager coordinates:
/// - Creating pools for new rate limit keys
/// - Replenishing pools from the database
/// - Returning unused tokens on shutdown
pub struct TokenPoolManager {
    /// Per-key token pools (keyed by the string representation of ActiveRateLimitKey)
    pools: DashMap<String, Arc<TokenPool>>,
    /// The rate limiting configuration
    config: Arc<RateLimitingConfig>,
}

impl TokenPoolManager {
    /// Create a new TokenPoolManager
    pub fn new(config: Arc<RateLimitingConfig>) -> Self {
        Self {
            pools: DashMap::new(),
            config,
        }
    }

    /// Get or create a pool for the given active limit
    fn get_or_create_pool(&self, active_limit: &ActiveRateLimit) -> Arc<TokenPool> {
        let key = active_limit.key.0.clone();
        self.pools
            .entry(key)
            .or_insert_with(|| Arc::new(TokenPool::new(active_limit.limit.clone())))
            .clone()
    }

    /// Try to consume tokens for a request from the in-memory pool.
    /// Returns Ok(()) if successful, Err if insufficient and replenishment is needed.
    pub(crate) async fn try_consume(
        &self,
        active_limits: &[ActiveRateLimit],
        tokens: u64,
        model_inferences: u64,
    ) -> Result<(), Vec<ActiveRateLimit>> {
        // First pass: check if all pools have sufficient tokens
        let mut pools_to_consume: Vec<(Arc<TokenPool>, u64)> = Vec::new();
        let mut needs_replenish = Vec::new();

        for active_limit in active_limits {
            let pool = self.get_or_create_pool(active_limit);
            let amount = match active_limit.limit.resource {
                RateLimitResource::Token => tokens,
                RateLimitResource::ModelInference => model_inferences,
            };

            if pool.needs_replenishment() {
                needs_replenish.push(active_limit.clone());
            }
            pools_to_consume.push((pool, amount));
        }

        if !needs_replenish.is_empty() {
            return Err(needs_replenish);
        }

        // Second pass: try to consume from all pools atomically
        // If any fails, we need to rollback and replenish
        let mut consumed = Vec::new();
        let mut failed = false;

        for (pool, amount) in &pools_to_consume {
            if pool.try_consume(*amount) {
                consumed.push((pool.clone(), *amount));
            } else {
                failed = true;
                break;
            }
        }

        if failed {
            // Rollback consumed tokens
            for (pool, amount) in consumed {
                pool.available.fetch_add(amount as i64, Ordering::Release);
                pool.used_from_pool.fetch_sub(amount, Ordering::Relaxed);
            }
            return Err(active_limits.to_vec());
        }

        // Record usage for P99 tracking
        for (pool, _) in pools_to_consume {
            pool.record_usage(tokens, model_inferences);
        }

        Ok(())
    }

    /// Replenish tokens from the database for the given active limits.
    pub(crate) async fn replenish(
        &self,
        client: &impl RateLimitQueries,
        active_limits: &[ActiveRateLimit],
    ) -> Result<(), Error> {
        let min_floor = self.config.pool().min_borrow_floor;

        // Build consume requests for each limit that needs replenishment
        let mut requests: Vec<ConsumeTicketsRequest> = Vec::new();
        let mut pool_mappings: Vec<(Arc<TokenPool>, ActiveRateLimit)> = Vec::new();

        for active_limit in active_limits {
            let pool = self.get_or_create_pool(active_limit);

            // Check if another thread is already replenishing
            if pool
                .replenish_in_progress
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                // Wait for the other thread to finish
                pool.replenish_notify.notified().await;
                continue;
            }

            let borrow_amount = pool.calculate_borrow_amount(min_floor);
            let request = ConsumeTicketsRequest {
                key: active_limit.key.clone(),
                requested: borrow_amount,
                capacity: active_limit.limit.capacity,
                refill_amount: active_limit.limit.refill_rate,
                refill_interval: active_limit.limit.interval.to_pg_interval(),
            };
            requests.push(request);
            pool_mappings.push((pool, active_limit.clone()));
        }

        if requests.is_empty() {
            return Ok(());
        }

        // Execute the database request
        let results = client.consume_tickets(&requests).await?;

        // Process results and update pools
        let mut failed_limits = Vec::new();
        for (result, (pool, active_limit)) in results.iter().zip(pool_mappings.iter()) {
            if result.success {
                pool.add_tokens(result.tickets_consumed);
            } else {
                // Track failed limits for error reporting
                failed_limits.push(FailedRateLimit {
                    key: active_limit.key.clone(),
                    requested: result.tickets_consumed,
                    available: result.tickets_remaining,
                    resource: active_limit.limit.resource,
                    scope_key: active_limit.scope_key.clone(),
                });
            }
            // Release the replenishment lock and notify waiters
            pool.replenish_in_progress.store(false, Ordering::Release);
            pool.replenish_notify.notify_waiters();
        }

        // Check if any requests failed
        if !failed_limits.is_empty() {
            return Err(Error::new(ErrorDetails::RateLimitExceeded {
                failed_rate_limits: failed_limits,
            }));
        }

        Ok(())
    }

    /// Return all unused tokens to the database on shutdown.
    pub async fn shutdown(&self, client: &impl RateLimitQueries) -> Result<(), Error> {
        let timeout = self.config.pool().shutdown_timeout();

        let return_future = async {
            let mut return_requests: Vec<ReturnTicketsRequest> = Vec::new();

            for entry in self.pools.iter() {
                let pool = entry.value();
                let unused = pool.unused_tokens();

                if unused > 0 {
                    let request = ReturnTicketsRequest {
                        key: ActiveRateLimitKey::new(entry.key().clone()),
                        returned: unused,
                        capacity: pool.limit.capacity,
                        refill_amount: pool.limit.refill_rate,
                        refill_interval: pool.limit.interval.to_pg_interval(),
                    };
                    return_requests.push(request);
                }
            }

            if return_requests.is_empty() {
                return Ok(());
            }

            tracing::info!(
                "Returning {} unused tokens across {} rate limit keys",
                return_requests.iter().map(|r| r.returned).sum::<u64>(),
                return_requests.len()
            );

            client.return_tickets(return_requests).await?;
            Ok(())
        };

        match tokio::time::timeout(timeout, return_future).await {
            Ok(result) => result,
            Err(_) => {
                tracing::warn!(
                    "Shutdown timeout ({:?}) exceeded while returning rate limit tokens",
                    timeout
                );
                Ok(()) // Don't fail shutdown due to timeout
            }
        }
    }

    /// Get the underlying config
    pub fn config(&self) -> &RateLimitingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_limit(resource: RateLimitResource, capacity: u64) -> Arc<RateLimit> {
        Arc::new(RateLimit {
            resource,
            interval: RateLimitInterval::Minute,
            capacity,
            refill_rate: capacity,
        })
    }

    #[test]
    fn test_token_pool_try_consume_success() {
        let limit = create_test_limit(RateLimitResource::Token, 100);
        let pool = TokenPool::new(limit);

        // Add some tokens
        pool.add_tokens(50);

        // Consume should succeed
        assert!(pool.try_consume(30), "Should be able to consume 30 tokens when 50 are available");
        assert_eq!(pool.available.load(Ordering::Acquire), 20, "Should have 20 tokens remaining");
        assert_eq!(pool.used_from_pool.load(Ordering::Acquire), 30, "Should have used 30 tokens");
    }

    #[test]
    fn test_token_pool_try_consume_insufficient() {
        let limit = create_test_limit(RateLimitResource::Token, 100);
        let pool = TokenPool::new(limit);

        // Add some tokens
        pool.add_tokens(20);

        // Consume should fail
        assert!(!pool.try_consume(30), "Should not be able to consume 30 tokens when only 20 are available");
        assert_eq!(pool.available.load(Ordering::Acquire), 20, "Should still have 20 tokens");
        assert_eq!(pool.used_from_pool.load(Ordering::Acquire), 0, "Should not have used any tokens");
    }

    #[test]
    fn test_token_pool_needs_replenishment() {
        let limit = create_test_limit(RateLimitResource::Token, 100);
        let pool = TokenPool::new(limit);

        // Initially needs replenishment (never borrowed)
        assert!(pool.needs_replenishment(), "Should need replenishment when never borrowed");

        // After borrowing, shouldn't need replenishment
        pool.add_tokens(50);
        assert!(!pool.needs_replenishment(), "Should not need replenishment after adding tokens");

        // After consuming most tokens, should need replenishment
        assert!(pool.try_consume(45), "Should consume 45 tokens"); // Only 5 left, which is 10% of 50
        assert!(pool.needs_replenishment(), "Should need replenishment when below threshold");
    }

    #[test]
    fn test_token_pool_calculate_borrow_amount_cold_start() {
        let limit = create_test_limit(RateLimitResource::Token, 1000);
        let pool = TokenPool::new(limit);

        // Cold start: should borrow COLD_START_BORROW_PCT of MAX_BORROW_CAP of capacity
        // MAX_BORROW_CAP = 0.25, COLD_START_BORROW_PCT = 0.10
        // So: 1000 * 0.25 * 0.10 = 25
        let borrow = pool.calculate_borrow_amount(1);
        assert_eq!(borrow, 25, "Cold start borrow amount should be 25");
    }

    #[test]
    fn test_token_pool_calculate_borrow_amount_with_history() {
        let limit = create_test_limit(RateLimitResource::Token, 1000);
        let pool = TokenPool::new(limit);

        // Add some usage history
        for i in 0..100 {
            pool.record_usage(i, 1);
        }

        // P99 should be around 99
        let borrow = pool.calculate_borrow_amount(1);
        assert!(borrow >= 98 && borrow <= 100, "P99-based borrow should be around 99, got {}", borrow);
    }

    #[test]
    fn test_token_pool_calculate_borrow_amount_respects_cap() {
        let limit = create_test_limit(RateLimitResource::Token, 100);
        let pool = TokenPool::new(limit);

        // Add high usage history
        for _ in 0..100 {
            pool.record_usage(1000, 1); // Much higher than capacity
        }

        // Should be capped at MAX_BORROW_CAP * capacity = 0.25 * 100 = 25
        let borrow = pool.calculate_borrow_amount(1);
        assert_eq!(borrow, 25, "Borrow amount should be capped at 25% of capacity");
    }

    #[test]
    fn test_token_pool_calculate_borrow_amount_respects_floor() {
        let limit = create_test_limit(RateLimitResource::Token, 100);
        let pool = TokenPool::new(limit);

        // Cold start with high floor
        let borrow = pool.calculate_borrow_amount(50);
        assert_eq!(borrow, 50, "Borrow amount should respect the floor of 50");
    }

    #[test]
    fn test_token_pool_unused_tokens() {
        let limit = create_test_limit(RateLimitResource::Token, 100);
        let pool = TokenPool::new(limit);

        pool.add_tokens(50);
        assert!(pool.try_consume(30), "Should consume 30 tokens");

        assert_eq!(pool.unused_tokens(), 20, "Should have 20 unused tokens");
    }
}
