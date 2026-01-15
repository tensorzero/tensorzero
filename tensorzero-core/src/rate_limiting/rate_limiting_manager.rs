//! Rate limiting manager for coordinating rate limiting operations.
//!
//! The `RateLimitingManager` is the primary interface for rate limiting operations.
//! It wraps a `RateLimitingConfig` and provides methods to consume and return tickets.
//!
//! The manager supports two modes:
//! - **Pooled mode (default)**: Pre-borrows tokens into an in-memory pool to reduce database contention
//! - **Direct mode**: Every request hits the database directly

use std::sync::Arc;

use dashmap::DashMap;
use tracing::Span;

use super::batching_buffer::{BatchResult, BatchingBuffer, TimerFlushStrategy};
use super::{
    FailedRateLimit, PoolMode, RateLimitResource, RateLimitResourceUsage, RateLimitedRequest,
    RateLimitingConfig, ScopeInfo, TicketBorrow, TicketBorrows,
};
use crate::db::postgres::PostgresConnectionInfo;
use crate::db::{
    ConsumeTicketsReceipt, ConsumeTicketsRequest, RateLimitQueries, ReturnTicketsRequest,
};
use crate::error::{Error, ErrorDetails};
use crate::rate_limiting::ActiveRateLimit;
use crate::rate_limiting::token_pool::TokenPool;

/// Manager for rate limiting operations.
///
/// This is the primary interface for consuming and returning rate limit tickets.
/// It wraps a `RateLimitingConfig` and coordinates all rate limiting operations.
pub struct RateLimitingManager {
    /// The rate limiting configuration
    config: Arc<RateLimitingConfig>,
    /// The database client for rate limiting operations
    client: Arc<dyn RateLimitQueries>,

    /// Per-key token pools (keyed by the string representation of ActiveRateLimitKey)
    pools: DashMap<String, Arc<TokenPool>>,
}

impl std::fmt::Debug for RateLimitingManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimitingManager")
            .field("config", &self.config)
            .field("client", &"<dyn RateLimitQueries>")
            .field("pools", &self.pools)
            .finish()
    }
}

impl RateLimitingManager {
    /// Create a new RateLimitingManager
    pub fn new(config: Arc<RateLimitingConfig>, client: PostgresConnectionInfo) -> Self {
        Self {
            config,
            client: Arc::new(client),
            pools: DashMap::new(),
        }
    }

    /// Create a new RateLimitingManager with a custom client (for testing).
    #[cfg(test)]
    pub fn new_with_client(
        config: Arc<RateLimitingConfig>,
        client: Arc<dyn RateLimitQueries>,
    ) -> Self {
        Self {
            config,
            client,
            pools: DashMap::new(),
        }
    }

    /// Create a new, dummy RateLimitingManager for unit tests.
    #[cfg(test)]
    pub fn new_dummy() -> Self {
        Self {
            config: Arc::new(RateLimitingConfig::default()),
            client: Arc::new(PostgresConnectionInfo::Disabled),
            pools: DashMap::new(),
        }
    }

    /// Returns true if there are no active rate limits.
    /// Used to optimize shutdown (skip operations if nothing to do).
    pub fn is_empty(&self) -> bool {
        self.config.rules.is_empty()
    }

    /// Get the underlying config
    pub fn config(&self) -> &RateLimitingConfig {
        &self.config
    }

    /// Returns the pool mode configured for this manager.
    fn pool_mode(&self) -> PoolMode {
        self.config.pooling.mode
    }

    /// Get or create a pool for the given active limit
    fn get_or_create_pool(&self, active_limit: &ActiveRateLimit) -> Arc<TokenPool> {
        let key = active_limit.key.0.clone();
        let buffer_timeout = self.config.pooling.buffer_timeout();
        self.pools
            .entry(key)
            .or_insert_with(|| Arc::new(TokenPool::new(active_limit.limit.clone(), buffer_timeout)))
            .clone()
    }

    /// Consume tickets for a rate-limited request.
    #[tracing::instrument(skip_all, fields(otel.name = "rate_limiting_consume_tickets", estimated_usage.tokens, estimated_usage.model_inferences))]
    pub async fn consume_tickets(
        self: &Arc<Self>,
        scope_info: &ScopeInfo,
        rate_limited_request: &impl RateLimitedRequest,
    ) -> Result<TicketBorrows, Error> {
        // We want rate-limiting errors to show up as errors in OpenTelemetry,
        // even though they only get logged as warnings to the console.
        self.consume_tickets_inner(scope_info, rate_limited_request)
            .await
            .inspect_err(|e| e.ensure_otel_span_errored(&Span::current()))
    }

    async fn consume_tickets_inner(
        self: &Arc<Self>,
        scope_info: &ScopeInfo,
        rate_limited_request: &impl RateLimitedRequest,
    ) -> Result<TicketBorrows, Error> {
        let span = Span::current();
        scope_info.apply_otel_span_attributes(&span);
        let limits = self.config.get_active_limits(scope_info);
        if limits.is_empty() {
            return Ok(TicketBorrows::empty(Arc::clone(self)));
        }

        let rate_limited_resources = self.config.get_rate_limited_resources(scope_info);
        let rate_limit_resource_requests =
            rate_limited_request.estimated_resource_usage(&rate_limited_resources)?;

        if let Some(tokens) = rate_limit_resource_requests.tokens {
            span.record("estimated_usage.tokens", tokens as i64);
        }
        if let Some(model_inferences) = rate_limit_resource_requests.model_inferences {
            span.record("estimated_usage.model_inferences", model_inferences as i64);
        }

        let ticket_requests: Result<Vec<ConsumeTicketsRequest>, Error> = limits
            .iter()
            .map(|limit| limit.get_consume_tickets_request(&rate_limit_resource_requests))
            .collect();
        let ticket_requests = ticket_requests?;

        // Choose path based on pool mode
        let receipts = match self.pool_mode() {
            PoolMode::Pooled => {
                // Pooled mode: consume from local pools
                self.consume_from_pools(&limits, &ticket_requests).await?
            }
            PoolMode::Direct => {
                // Direct mode: hit the database for every request
                self.client.consume_tickets(&ticket_requests).await?
            }
        };

        TicketBorrows::new(Arc::clone(self), receipts, limits, ticket_requests)
    }

    /// Consume tickets from the in-memory pools using three-phase approach.
    ///
    /// The three-phase approach provides best-effort fairness for multi-limit acquisition:
    ///
    /// 1. **Phase 1 (Fast Path):** Try to consume from all pools immediately. If all
    ///    succeed, return without any DB call or added latency.
    ///
    /// 2. **Phase 2 (Batching):** If any pool has insufficient tokens, rollback Phase 1
    ///    successes and join batching buffers for failed pools. Wait for all batches
    ///    to complete.
    ///
    /// 3. **Phase 3 (Retry):** After batches complete, retry ALL pools. Rollback partial
    ///    successes if any pool fails to prevent token leakage.
    ///
    /// The batching buffer groups concurrent requests to reduce DB calls. During cold
    /// start or bursts, multiple requests join the same batch window (default 10ms)
    /// and a single DB call serves all of them.
    ///
    /// **Best-effort fairness:** We do not reserve tokens while waiting for batching.
    /// Under sustained contention, a request may be retried in Phase 3 and still fail
    /// if tokens are consumed by other requests. In that case, we return RateLimitExceeded
    /// rather than falling back to direct DB access.
    async fn consume_from_pools(
        &self,
        limits: &[super::ActiveRateLimit],
        ticket_requests: &[ConsumeTicketsRequest],
    ) -> Result<Vec<ConsumeTicketsReceipt>, Error> {
        // Get or create pools for all limits
        let pools: Vec<Arc<super::token_pool::TokenPool>> = limits
            .iter()
            .map(|limit| self.get_or_create_pool(limit))
            .collect();

        // Pre-check: Reject immediately if any pool is in exhaustion backoff.
        // This prevents spamming the database when the rate limit is depleted.
        for (pool, (limit, request)) in pools.iter().zip(limits.iter().zip(ticket_requests.iter()))
        {
            if pool.exhaustion_backoff.should_reject() {
                return Err(Error::new(ErrorDetails::RateLimitExceeded {
                    failed_rate_limits: vec![FailedRateLimit {
                        key: limit.key.clone(),
                        requested: request.requested,
                        available: 0,
                        resource: limit.limit.resource,
                        scope_key: limit.scope_key.clone(),
                    }],
                }));
            }
        }

        // Phase 1: Try to consume from all pools (fast path)
        let mut phase1_results: Vec<bool> = Vec::with_capacity(limits.len());
        for (pool, request) in pools.iter().zip(ticket_requests.iter()) {
            let success = pool.try_consume(request.requested);
            phase1_results.push(success);
        }

        if phase1_results.iter().all(|s| *s) {
            // All pools had sufficient tokens - fast path succeeded.
            // Record estimated usage for P99 tracking and collect epochs for later adjustment.
            let epochs: Vec<u32> = pools
                .iter()
                .zip(ticket_requests.iter())
                .map(|(pool, request)| pool.record_estimated_usage(request.requested))
                .collect();

            // Build synthetic receipts for the successful consumptions.
            let receipts: Vec<ConsumeTicketsReceipt> = ticket_requests
                .iter()
                .zip(epochs.iter())
                .map(|(r, &epoch)| ConsumeTicketsReceipt {
                    key: r.key.clone(),
                    success: true,
                    tickets_remaining: r.capacity.saturating_sub(r.requested),
                    tickets_consumed: r.requested,
                    recorded_epoch: Some(epoch),
                })
                .collect();
            return Ok(receipts);
        }

        // Phase 2: Some pools had insufficient tokens.
        // Roll back Phase 1 successes and join batching for failed pools.
        for (i, (pool, request)) in pools.iter().zip(ticket_requests.iter()).enumerate() {
            if phase1_results[i] {
                pool.rollback_consume(request.requested);
            }
        }

        // Identify failed pools and join their batching buffers
        let failed_indices: Vec<usize> = phase1_results
            .iter()
            .enumerate()
            .filter(|(_, success)| !**success)
            .map(|(i, _)| i)
            .collect();

        // Wait for all failed pools to batch and replenish
        self.wait_for_batches(&pools, limits, ticket_requests, &failed_indices)
            .await?;

        // Phase 3: Retry ALL pools (best-effort fairness)
        // After batching completes, pools should have tokens. Retry all pools
        // and rollback partial successes if any fails.
        let mut phase3_successes: Vec<usize> = Vec::with_capacity(limits.len());

        for (i, (pool, request)) in pools.iter().zip(ticket_requests.iter()).enumerate() {
            if pool.try_consume(request.requested) {
                phase3_successes.push(i);
            } else {
                // Rollback Phase 3 successes
                for &idx in &phase3_successes {
                    pools[idx].rollback_consume(ticket_requests[idx].requested);
                }

                // Return RateLimitExceeded instead of DB fallback
                return Err(Error::new(ErrorDetails::RateLimitExceeded {
                    failed_rate_limits: vec![FailedRateLimit {
                        key: limits[i].key.clone(),
                        requested: request.requested,
                        available: 0,
                        resource: limits[i].limit.resource,
                        scope_key: limits[i].scope_key.clone(),
                    }],
                }));
            }
        }

        // All Phase 3 consumptions succeeded.
        // Record estimated usage for P99 tracking and collect epochs for later adjustment.
        let epochs: Vec<u32> = pools
            .iter()
            .zip(ticket_requests.iter())
            .map(|(pool, request)| pool.record_estimated_usage(request.requested))
            .collect();

        let receipts: Vec<ConsumeTicketsReceipt> = ticket_requests
            .iter()
            .zip(epochs.iter())
            .map(|(r, &epoch)| ConsumeTicketsReceipt {
                key: r.key.clone(),
                success: true,
                tickets_remaining: r.capacity.saturating_sub(r.requested),
                tickets_consumed: r.requested,
                recorded_epoch: Some(epoch),
            })
            .collect();

        Ok(receipts)
    }

    /// Wait for batching to complete on failed pools.
    ///
    /// Joins the batching buffer for each failed pool and waits for all batches
    /// to complete. The flush callback handles replenishment from the database.
    async fn wait_for_batches(
        &self,
        pools: &[Arc<super::token_pool::TokenPool>],
        limits: &[super::ActiveRateLimit],
        ticket_requests: &[ConsumeTicketsRequest],
        failed_indices: &[usize],
    ) -> Result<(), Error> {
        // Collect batch receivers for all failed pools
        let mut batch_futures = Vec::with_capacity(failed_indices.len());

        for &idx in failed_indices {
            let pool = Arc::clone(&pools[idx]);
            let limit = limits[idx].clone();
            let tokens_needed = ticket_requests[idx].requested;
            let client = Arc::clone(&self.client);

            // Create flush closure that captures necessary state
            let pool_for_flush = Arc::clone(&pool);
            let limit_for_flush = limit.clone();
            let client_for_flush = Arc::clone(&client);

            let rx = pool
                .batching_buffer
                .join_or_open(tokens_needed, move |batch_id| {
                    let pool = pool_for_flush;
                    let limit = limit_for_flush;
                    let client = client_for_flush;

                    async move {
                        Self::flush_batch(&pool, &limit, &client, batch_id).await;
                    }
                })
                .await;

            batch_futures.push(rx);
        }

        // Wait for all batches to complete
        for rx in batch_futures {
            match rx.await {
                Ok(BatchResult::Success) => {}
                Ok(BatchResult::Error(msg)) => {
                    tracing::warn!("Batch flush failed: {msg}");
                    // Continue - Phase 3 will handle the failure
                }
                Err(_) => {
                    tracing::warn!("Batch channel closed unexpectedly");
                    // Continue - Phase 3 will handle the failure
                }
            }
        }

        Ok(())
    }

    /// Flush a batch by draining waiters and borrowing from the database.
    ///
    /// This is called by the batching buffer's flush task after the batch window closes.
    async fn flush_batch(
        pool: &Arc<super::token_pool::TokenPool>,
        limit: &super::ActiveRateLimit,
        client: &Arc<dyn RateLimitQueries>,
        batch_id: u64,
    ) {
        // Drain waiters (this closes the batch window under the lock)
        let (waiters, queued_tokens) = pool.batching_buffer.drain_waiters(batch_id).await;

        if waiters.is_empty() {
            return;
        }

        // Calculate borrow amount based on queued demand and historical P99
        let borrow_amount = pool.calculate_borrow_amount(queued_tokens);

        // Borrow from the database
        let request = ConsumeTicketsRequest {
            key: limit.key.clone(),
            capacity: limit.limit.capacity,
            refill_amount: limit.limit.refill_rate,
            refill_interval: limit.limit.interval.to_pg_interval(),
            requested: borrow_amount,
        };

        let result = client.consume_tickets(&[request]).await;

        match result {
            Ok(receipts) => {
                if let Some(receipt) = receipts.first() {
                    pool.update_tickets_remaining(receipt.tickets_remaining);

                    if receipt.success && receipt.tickets_consumed > 0 {
                        // Tokens available - reset exhaustion backoff
                        pool.exhaustion_backoff.on_available();
                        pool.add_tokens(receipt.tickets_consumed);

                        // Note: Usage is recorded at consume time (Phase 1/3), not here.
                        // This ensures P99 reflects request demand, not batch borrowing.

                        // Notify all waiters of success
                        BatchingBuffer::<TimerFlushStrategy>::notify_waiters(
                            waiters,
                            BatchResult::Success,
                        );
                    } else {
                        // Rate limit exhausted - enter backoff to avoid DB spam
                        pool.exhaustion_backoff.on_exhausted();
                        BatchingBuffer::<TimerFlushStrategy>::notify_waiters(
                            waiters,
                            BatchResult::Error("Rate limit exceeded".to_string()),
                        );
                    }
                } else {
                    // Empty receipts - unexpected but handle gracefully to avoid hanging waiters
                    tracing::warn!("Batch replenishment returned empty receipts");
                    BatchingBuffer::<TimerFlushStrategy>::notify_waiters(
                        waiters,
                        BatchResult::Error("Empty receipts from database".to_string()),
                    );
                }
            }
            Err(e) => {
                // DB error - don't enter backoff (might be transient)
                tracing::warn!("Batch replenishment failed: {e}");
                BatchingBuffer::<TimerFlushStrategy>::notify_waiters(
                    waiters,
                    BatchResult::Error(e.to_string()),
                );
            }
        }
    }

    /// Return tickets based on actual resource usage.
    #[tracing::instrument(skip_all, fields(otel.name = "rate_limiting_return_tickets", actual_usage.tokens, actual_usage.model_inferences, underestimate))]
    pub async fn return_tickets(
        &self,
        ticket_borrows: TicketBorrows,
        actual_usage: RateLimitResourceUsage,
    ) -> Result<(), Error> {
        // We want rate-limiting errors to show up as errors in OpenTelemetry,
        // even though they only get logged as warnings to the console.
        self.return_tickets_inner(ticket_borrows, actual_usage)
            .await
            .inspect_err(|e| e.ensure_otel_span_errored(&Span::current()))
    }

    async fn return_tickets_inner(
        &self,
        ticket_borrows: TicketBorrows,
        actual_usage: RateLimitResourceUsage,
    ) -> Result<(), Error> {
        let span = Span::current();
        // We cast the usage values to i64 so that they are reported as integers in OpenTelemetry (rather than strings)
        let (tokens, model_inferences) = match actual_usage {
            RateLimitResourceUsage::Exact {
                tokens,
                model_inferences,
            } => {
                span.record("actual_usage.tokens", tokens as i64);
                span.record("actual_usage.model_inferences", model_inferences as i64);
                span.record("underestimate", false);
                (tokens, model_inferences)
            }
            RateLimitResourceUsage::UnderEstimate {
                tokens,
                model_inferences,
            } => {
                span.record("actual_usage.tokens", tokens as i64);
                span.record("actual_usage.model_inferences", model_inferences as i64);
                span.record("underestimate", true);
                (tokens, model_inferences)
            }
        };

        // Adjust pool accounting based on actual vs estimated usage.
        // The pool tracks borrowed tokens and actual usage; DB reconciliation happens at shutdown.
        // Also adjusts P99 tracker to reflect actual usage instead of estimated.
        if self.pool_mode() == PoolMode::Pooled {
            for borrow in ticket_borrows.borrows() {
                let pool = self.get_or_create_pool(&borrow.active_limit);
                let actual_usage_this_request = match borrow.active_limit.limit.resource {
                    RateLimitResource::ModelInference => model_inferences,
                    RateLimitResource::Token => tokens,
                };
                pool.adjust_usage(
                    borrow.receipt.tickets_consumed,
                    actual_usage_this_request,
                    borrow.receipt.recorded_epoch,
                );
            }

            // Don't make DB calls in pooled mode - pool handles accounting,
            // and unused tokens are returned to DB at shutdown.
            return Ok(());
        }

        // In direct mode, make DB calls to return/consume the difference between estimated and actual usage.
        let mut requests = Vec::new();
        let mut returns = Vec::new();

        for borrow in ticket_borrows.borrows() {
            let TicketBorrow {
                receipt,
                active_limit,
            } = borrow;

            // Extract the actual value - we'll check 'Exact/UnderEstimate' further on
            let actual_usage_this_request = match active_limit.limit.resource {
                RateLimitResource::ModelInference => match actual_usage {
                    RateLimitResourceUsage::Exact {
                        model_inferences, ..
                    }
                    | RateLimitResourceUsage::UnderEstimate {
                        model_inferences, ..
                    } => model_inferences,
                },
                RateLimitResource::Token => match actual_usage {
                    RateLimitResourceUsage::Exact { tokens, .. }
                    | RateLimitResourceUsage::UnderEstimate { tokens, .. } => tokens,
                },
            };

            match actual_usage_this_request.cmp(&receipt.tickets_consumed) {
                std::cmp::Ordering::Greater => {
                    // Actual usage exceeds borrowed, add the difference to requests and log a warning
                    tracing::warn!(
                        "Actual usage exceeds borrowed for {:?}: {} estimated and {actual_usage_this_request} used",
                        active_limit.limit.resource,
                        receipt.tickets_consumed
                    );
                    let difference = actual_usage_this_request - receipt.tickets_consumed;
                    requests.push(active_limit.get_consume_tickets_request_for_return(difference)?);
                }
                std::cmp::Ordering::Less => {
                    match actual_usage {
                        RateLimitResourceUsage::Exact { .. } => {
                            // Borrowed exceeds actual usage, add the difference to returns
                            let difference = receipt.tickets_consumed - actual_usage_this_request;
                            returns.push(active_limit.get_return_tickets_request(difference)?);
                        }
                        RateLimitResourceUsage::UnderEstimate { .. } => {
                            // If our returned usage is only an estimate, then don't return any tickets,
                            // even if it looks like we over-borrowed.
                        }
                    }
                }
                std::cmp::Ordering::Equal => (),
            };
        }

        // Only make DB calls if there's actually work to do
        if !requests.is_empty() || !returns.is_empty() {
            let consume_future = async {
                if requests.is_empty() {
                    Ok(vec![])
                } else {
                    self.client.consume_tickets(&requests).await
                }
            };
            let return_future = async {
                if returns.is_empty() {
                    Ok(vec![])
                } else {
                    self.client.return_tickets(returns).await
                }
            };

            let (consume_result, return_result) = tokio::join!(consume_future, return_future);

            consume_result?;
            return_result?;
        }

        Ok(())
    }

    /// Shutdown the rate limiting manager.
    /// In pooled mode, returns unused tokens to the database.
    pub async fn shutdown(&self) -> Result<(), Error> {
        if self.pool_mode() == PoolMode::Direct || self.pools.is_empty() {
            return Ok(());
        }

        // Collect all unused tokens from pools and return them to the database
        let mut return_requests = Vec::new();

        for entry in &self.pools {
            let token_pool = entry.value();
            let unused = token_pool.unused_tokens();
            if unused > 0 {
                return_requests.push(ReturnTicketsRequest {
                    key: super::ActiveRateLimitKey::new(entry.key().clone()),
                    capacity: token_pool.limit.capacity,
                    refill_amount: token_pool.limit.refill_rate,
                    refill_interval: token_pool.limit.interval.to_pg_interval(),
                    returned: unused,
                });
            }
        }

        if !return_requests.is_empty() {
            // Use a timeout to avoid blocking shutdown indefinitely
            let timeout = self.config.pooling.shutdown_token_return_timeout();
            match tokio::time::timeout(timeout, self.client.return_tickets(return_requests)).await {
                Ok(result) => {
                    if let Err(e) = result {
                        tracing::warn!("Failed to return tokens during shutdown: {e}");
                    }
                }
                Err(_) => {
                    tracing::warn!("Timeout returning tokens during shutdown");
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{ConsumeTicketsReceipt, MockRateLimitQueries, ReturnTicketsReceipt};
    use crate::rate_limiting::{
        EstimatedRateLimitResourceUsage, PoolingConfig, RateLimit, RateLimitInterval,
        RateLimitingConfigPriority, RateLimitingConfigRule, RateLimitingConfigScopes,
    };
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn make_token_limit(capacity: u64) -> Arc<RateLimit> {
        Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Minute,
            capacity,
            refill_rate: 10,
        })
    }

    fn make_scope_info(tags: Vec<(&str, &str)>) -> ScopeInfo {
        let tags_map: HashMap<String, String> = tags
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        ScopeInfo {
            tags: Arc::new(tags_map),
            api_key_public_id: None,
        }
    }

    #[test]
    fn test_manager_is_empty_with_rules() {
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(100)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = Arc::new(RateLimitingConfig {
            rules: vec![rule],
            enabled: true,
            pooling: PoolingConfig::default(),
        });
        let manager = RateLimitingManager::new(config, PostgresConnectionInfo::Disabled);

        assert!(
            !manager.is_empty(),
            "Manager with rules should not be empty"
        );
    }

    #[tokio::test]
    async fn test_shutdown_returns_ok() {
        let manager = RateLimitingManager::new_dummy();
        let result = manager.shutdown().await;
        assert!(result.is_ok(), "Shutdown should return Ok");
    }

    // Tests for consume_tickets and return_tickets with mock client

    struct MockRateLimitedRequest {
        tokens: u64,
        model_inferences: u64,
    }

    impl RateLimitedRequest for MockRateLimitedRequest {
        fn estimated_resource_usage(
            &self,
            _resources: &[RateLimitResource],
        ) -> Result<EstimatedRateLimitResourceUsage, Error> {
            Ok(EstimatedRateLimitResourceUsage {
                tokens: Some(self.tokens),
                model_inferences: Some(self.model_inferences),
            })
        }
    }

    /// Create a mock client that returns successful consume_tickets responses.
    fn create_mock_client_success() -> MockRateLimitQueries {
        let mut mock = MockRateLimitQueries::new();
        mock.expect_consume_tickets().returning(|requests| {
            let receipts: Vec<ConsumeTicketsReceipt> = requests
                .iter()
                .map(|r| ConsumeTicketsReceipt {
                    key: r.key.clone(),
                    success: true,
                    tickets_remaining: r.capacity.saturating_sub(r.requested),
                    tickets_consumed: r.requested,
                    recorded_epoch: None, // Direct mode doesn't use P99 tracking
                })
                .collect();
            Box::pin(async move { Ok(receipts) })
        });
        mock.expect_return_tickets().returning(|requests| {
            let receipts: Vec<ReturnTicketsReceipt> = requests
                .iter()
                .map(|r| ReturnTicketsReceipt {
                    key: r.key.clone(),
                    balance: r.returned,
                })
                .collect();
            Box::pin(async move { Ok(receipts) })
        });
        mock
    }

    /// Create a mock client that returns failed consume_tickets responses (rate limit exceeded).
    fn create_mock_client_rate_limited() -> MockRateLimitQueries {
        let mut mock = MockRateLimitQueries::new();
        mock.expect_consume_tickets().returning(|requests| {
            let receipts: Vec<ConsumeTicketsReceipt> = requests
                .iter()
                .map(|r| ConsumeTicketsReceipt {
                    key: r.key.clone(),
                    success: false,
                    tickets_remaining: 0,
                    tickets_consumed: 0,
                    recorded_epoch: None, // Direct mode doesn't use P99 tracking
                })
                .collect();
            Box::pin(async move { Ok(receipts) })
        });
        mock
    }

    fn make_direct_config_with_rule(rule: RateLimitingConfigRule) -> Arc<RateLimitingConfig> {
        Arc::new(RateLimitingConfig {
            rules: vec![rule],
            enabled: true,
            pooling: PoolingConfig {
                mode: PoolMode::Direct,
                ..PoolingConfig::default()
            },
        })
    }

    #[tokio::test]
    async fn test_consume_tickets_empty_rules() {
        // With empty rules, consume_tickets doesn't call the client
        let mock = MockRateLimitQueries::new();
        let config = Arc::new(RateLimitingConfig::default());
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);
        let request = MockRateLimitedRequest {
            tokens: 100,
            model_inferences: 1,
        };

        let result = manager.consume_tickets(&scope_info, &request).await;
        assert!(result.is_ok(), "Should succeed with empty rules");
        let borrows = result.unwrap();
        assert!(
            borrows.borrows().is_empty(),
            "Should have no borrows with empty rules"
        );
    }

    #[tokio::test]
    async fn test_consume_tickets_disabled_config() {
        // With disabled config, consume_tickets doesn't call the client
        let mock = MockRateLimitQueries::new();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(100)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = Arc::new(RateLimitingConfig {
            rules: vec![rule],
            enabled: false,
            pooling: PoolingConfig::default(),
        });
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);
        let request = MockRateLimitedRequest {
            tokens: 100,
            model_inferences: 1,
        };

        let result = manager.consume_tickets(&scope_info, &request).await;
        assert!(result.is_ok(), "Should succeed with disabled config");
        let borrows = result.unwrap();
        assert!(
            borrows.borrows().is_empty(),
            "Should have no borrows with disabled config"
        );
    }

    #[tokio::test]
    async fn test_consume_tickets_success_direct_mode() {
        let mock = create_mock_client_success();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1000)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_direct_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);
        let request = MockRateLimitedRequest {
            tokens: 100,
            model_inferences: 1,
        };

        let result = manager.consume_tickets(&scope_info, &request).await;
        assert!(result.is_ok(), "Should succeed with valid request");
        let borrows = result.unwrap();
        assert_eq!(borrows.borrows().len(), 1, "Should have one borrow");
    }

    #[tokio::test]
    async fn test_consume_tickets_rate_limit_exceeded_direct_mode() {
        let mock = create_mock_client_rate_limited();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1000)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_direct_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);
        let request = MockRateLimitedRequest {
            tokens: 100,
            model_inferences: 1,
        };

        let result = manager.consume_tickets(&scope_info, &request).await;
        assert!(result.is_err(), "Should fail when rate limit exceeded");
    }

    #[tokio::test]
    async fn test_return_tickets_exact_usage_equals_estimate() {
        let mock = create_mock_client_success();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1000)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_direct_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);
        let request = MockRateLimitedRequest {
            tokens: 100,
            model_inferences: 1,
        };

        let borrows = manager
            .consume_tickets(&scope_info, &request)
            .await
            .unwrap();

        // Return with exact same usage as estimated - no DB calls needed
        let actual_usage = RateLimitResourceUsage::Exact {
            tokens: 100,
            model_inferences: 1,
        };
        let result = manager.return_tickets(borrows, actual_usage).await;
        assert!(result.is_ok(), "Should succeed when actual equals estimate");
    }

    #[tokio::test]
    async fn test_return_tickets_exact_usage_less_than_estimate() {
        let mock = create_mock_client_success();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1000)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_direct_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);
        let request = MockRateLimitedRequest {
            tokens: 100,
            model_inferences: 1,
        };

        let borrows = manager
            .consume_tickets(&scope_info, &request)
            .await
            .unwrap();

        // Return with less usage than estimated (should return tickets)
        let actual_usage = RateLimitResourceUsage::Exact {
            tokens: 50,
            model_inferences: 1,
        };
        let result = manager.return_tickets(borrows, actual_usage).await;
        assert!(
            result.is_ok(),
            "Should succeed when actual is less than estimate"
        );
    }

    #[tokio::test]
    async fn test_return_tickets_exact_usage_more_than_estimate() {
        let mock = create_mock_client_success();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1000)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_direct_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);
        let request = MockRateLimitedRequest {
            tokens: 100,
            model_inferences: 1,
        };

        let borrows = manager
            .consume_tickets(&scope_info, &request)
            .await
            .unwrap();

        // Return with more usage than estimated (should consume additional)
        let actual_usage = RateLimitResourceUsage::Exact {
            tokens: 150,
            model_inferences: 1,
        };
        let result = manager.return_tickets(borrows, actual_usage).await;
        assert!(
            result.is_ok(),
            "Should succeed when actual exceeds estimate"
        );
    }

    #[tokio::test]
    async fn test_return_tickets_underestimate_no_return() {
        let mock = create_mock_client_success();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1000)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_direct_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);
        let request = MockRateLimitedRequest {
            tokens: 100,
            model_inferences: 1,
        };

        let borrows = manager
            .consume_tickets(&scope_info, &request)
            .await
            .unwrap();

        // Return with UnderEstimate - should NOT return tickets even if usage looks lower
        let actual_usage = RateLimitResourceUsage::UnderEstimate {
            tokens: 50,
            model_inferences: 1,
        };
        let result = manager.return_tickets(borrows, actual_usage).await;
        assert!(result.is_ok(), "Should succeed with underestimate usage");
    }

    // Tests for pooled mode

    fn make_pooled_config_with_rule(rule: RateLimitingConfigRule) -> Arc<RateLimitingConfig> {
        Arc::new(RateLimitingConfig {
            rules: vec![rule],
            enabled: true,
            pooling: PoolingConfig {
                mode: PoolMode::Pooled,
                ..PoolingConfig::default()
            },
        })
    }

    /// Create a mock client that tracks call counts and returns successful responses.
    /// Returns (mock, call_counter) where call_counter can be used to verify DB calls.
    fn create_mock_client_with_counter() -> (MockRateLimitQueries, Arc<std::sync::atomic::AtomicU32>)
    {
        // These point to the same AtomicU32, so we can increment inside the closure.
        let consume_count = Arc::new(AtomicU32::new(0));
        let consume_count_clone = Arc::clone(&consume_count);

        let mut mock = MockRateLimitQueries::new();
        mock.expect_consume_tickets().returning(move |requests| {
            consume_count_clone.fetch_add(1, Ordering::SeqCst);
            let receipts: Vec<ConsumeTicketsReceipt> = requests
                .iter()
                .map(|r| ConsumeTicketsReceipt {
                    key: r.key.clone(),
                    success: true,
                    tickets_remaining: r.capacity.saturating_sub(r.requested),
                    tickets_consumed: r.requested,
                    recorded_epoch: None, // Direct mode doesn't use P99 tracking
                })
                .collect();
            Box::pin(async move { Ok(receipts) })
        });
        mock.expect_return_tickets().returning(|requests| {
            let receipts: Vec<ReturnTicketsReceipt> = requests
                .iter()
                .map(|r| ReturnTicketsReceipt {
                    key: r.key.clone(),
                    balance: r.returned,
                })
                .collect();
            Box::pin(async move { Ok(receipts) })
        });

        (mock, consume_count)
    }

    #[tokio::test]
    async fn test_pooled_mode_batching_reduces_db_calls() {
        // With batching, sequential requests should be batched together,
        // resulting in fewer DB calls than the total number of requests.
        // During cold start, each batch borrows 10x the queued amount.
        let (mock, consume_count) = create_mock_client_with_counter();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(10000)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_pooled_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);

        // Make 10 requests, each needing 10 tokens.
        for i in 0..10 {
            let request = MockRateLimitedRequest {
                tokens: 10,
                model_inferences: 1,
            };
            let result = manager.consume_tickets(&scope_info, &request).await;
            assert!(result.is_ok(), "Request {i} should succeed");
        }

        // Batching should reduce DB calls compared to 10 individual calls, even during cold start.
        // Due to timing variations in async mutex acquisition and batch window timing,
        // we may see more than 1 batch but should still see significant reduction.
        let db_calls = consume_count.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            db_calls < 10,
            "Batching should reduce DB calls, expected < 10 but got {db_calls}"
        );
    }

    #[tokio::test]
    async fn test_pooled_mode_serves_from_pool_after_batching() {
        // After batching fills the pool, subsequent requests should be served
        // from the in-memory pool without hitting the DB (fast path).
        let (mock, consume_count) = create_mock_client_with_counter();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1_000_000)], // Large capacity
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_pooled_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);

        // Make 100 requests, each needing 10 tokens (1000 total)
        // With cold-start multiplier of 10x, batches borrow 100 tokens each.
        // With adaptive cap at 25% of 1M = 250K, we can borrow plenty.
        for _ in 0..100 {
            let request = MockRateLimitedRequest {
                tokens: 10,
                model_inferences: 1,
            };
            let _ = manager.consume_tickets(&scope_info, &request).await;
        }

        // With batching, we should have made far fewer than 100 DB calls
        let initial_db_calls = consume_count.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            initial_db_calls <= 15,
            "Batching should reduce DB calls, expected <= 15 but got {initial_db_calls}"
        );

        // Record DB calls before subsequent requests
        let db_calls_before = consume_count.load(std::sync::atomic::Ordering::SeqCst);

        // Subsequent requests should be served from pool without DB calls
        // (as long as pool has capacity)
        for i in 0..50 {
            let request = MockRateLimitedRequest {
                tokens: 10,
                model_inferences: 1,
            };
            let result = manager.consume_tickets(&scope_info, &request).await;
            assert!(
                result.is_ok(),
                "Request {i} should succeed from pool without DB"
            );
        }
        let db_calls_after = consume_count.load(std::sync::atomic::Ordering::SeqCst);

        // Should have made 0 or very few additional DB calls (only if replenishment triggered)
        let additional_calls = db_calls_after - db_calls_before;
        assert!(
            additional_calls <= 5,
            "Should serve most requests from pool without DB calls, but made {additional_calls} calls"
        );
    }

    #[tokio::test]
    async fn test_pooled_mode_return_tickets_no_db_call_when_exact_match() {
        // In pooled mode, returning tickets with exact usage match should not hit DB
        let (mock, consume_count) = create_mock_client_with_counter();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1_000_000)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_pooled_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);

        // Fill pool with initial requests (batching will borrow tokens)
        for _ in 0..100 {
            let request = MockRateLimitedRequest {
                tokens: 10,
                model_inferences: 1,
            };
            let _ = manager.consume_tickets(&scope_info, &request).await;
        }

        // Get one more borrow after pool is filled
        let request = MockRateLimitedRequest {
            tokens: 100,
            model_inferences: 1,
        };
        let borrows = manager
            .consume_tickets(&scope_info, &request)
            .await
            .unwrap();

        let db_calls_before_return = consume_count.load(std::sync::atomic::Ordering::SeqCst);

        // Return with exact match - should not hit DB
        let actual_usage = RateLimitResourceUsage::Exact {
            tokens: 100,
            model_inferences: 1,
        };
        let result = manager.return_tickets(borrows, actual_usage).await;
        assert!(result.is_ok(), "Return tickets should succeed");

        let db_calls_after_return = consume_count.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            db_calls_before_return, db_calls_after_return,
            "Return tickets with exact match should not hit DB"
        );
    }

    #[tokio::test]
    async fn test_direct_mode_always_hits_db() {
        // In direct mode, every request should hit the DB
        let (mock, consume_count) = create_mock_client_with_counter();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(10000)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_direct_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);

        // Make 150 requests (more than warm-up threshold)
        for i in 0..150 {
            let request = MockRateLimitedRequest {
                tokens: 10,
                model_inferences: 1,
            };
            let result = manager.consume_tickets(&scope_info, &request).await;
            assert!(result.is_ok(), "Request {i} should succeed in direct mode");
        }

        // Every request should have hit the DB in direct mode
        assert_eq!(
            consume_count.load(std::sync::atomic::Ordering::SeqCst),
            150,
            "Direct mode should hit DB for every request"
        );
    }

    #[tokio::test]
    async fn test_pooled_mode_large_request_calls_db() {
        // When a request is too large for the pool to satisfy even after batching,
        // we should fall back to DB.
        let (mock, consume_count) = create_mock_client_with_counter();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1_000_000)], // Large capacity
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_pooled_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);

        // Fill pool with small requests (batching will borrow ~100 tokens per batch)
        for _ in 0..100 {
            let request = MockRateLimitedRequest {
                tokens: 10,
                model_inferences: 1,
            };
            let _ = manager.consume_tickets(&scope_info, &request).await;
        }

        // With batching, should have made far fewer than 100 DB calls
        let initial_db_calls = consume_count.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            initial_db_calls <= 15,
            "Batching should reduce DB calls, expected <= 15 but got {initial_db_calls}"
        );

        // Now make a request that's much larger than what the pool can provide.
        let request = MockRateLimitedRequest {
            tokens: 100_000,
            model_inferences: 1,
        };
        let result = manager.consume_tickets(&scope_info, &request).await;
        assert!(
            result.is_ok(),
            "Large request should succeed by buffering itself into a batch"
        );

        // Should have made 1 additional DB call for the large request
        let post_initial_db_calls =
            consume_count.load(std::sync::atomic::Ordering::SeqCst) - initial_db_calls;
        assert!(
            post_initial_db_calls == 1,
            "Large request should fall back to DB, expected <= 2 additional calls but got {post_initial_db_calls}"
        );
    }

    #[tokio::test]
    async fn test_pooled_mode_request_within_borrow_capacity_uses_pool() {
        // When the pool has tokens from previous batches, requests should be
        // served from pool (fast path) without triggering new batches.
        let (mock, consume_count) = create_mock_client_with_counter();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1_000_000)], // Large capacity
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_pooled_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);

        // Fill pool with medium requests (100 tokens each)
        // With cold-start multiplier of 10x, each batch borrows 1000 tokens
        for _ in 0..100 {
            let request = MockRateLimitedRequest {
                tokens: 100,
                model_inferences: 1,
            };
            let _ = manager.consume_tickets(&scope_info, &request).await;
        }

        // With batching, should have made far fewer than 100 DB calls
        let initial_db_calls = consume_count.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            initial_db_calls <= 15,
            "Batching should reduce DB calls, expected <= 15 but got {initial_db_calls}"
        );

        // The pool should now have tokens from the previous batches
        // Small requests should be served directly from pool (fast path)
        let request = MockRateLimitedRequest {
            tokens: 50, // Well within what the pool can provide
            model_inferences: 1,
        };
        let _borrows = manager
            .consume_tickets(&scope_info, &request)
            .await
            .unwrap();

        // Subsequent small requests should be served from pool without DB calls
        let db_calls_before = consume_count.load(std::sync::atomic::Ordering::SeqCst);
        for _ in 0..10 {
            let request = MockRateLimitedRequest {
                tokens: 50,
                model_inferences: 1,
            };
            let _borrows = manager
                .consume_tickets(&scope_info, &request)
                .await
                .unwrap();
        }
        let db_calls_after = consume_count.load(std::sync::atomic::Ordering::SeqCst);

        // Should have made 0 or very few additional DB calls (all served from pool)
        let additional_calls = db_calls_after - db_calls_before;
        assert!(
            additional_calls <= 1,
            "Subsequent requests should be served from pool without DB calls, but made {additional_calls} calls"
        );
    }
}
