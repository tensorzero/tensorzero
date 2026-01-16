//! Rate limiting manager for coordinating rate limiting operations.
//!
//! The `RateLimitingManager` is the primary interface for rate limiting operations.
//! It wraps a `RateLimitingConfig` and provides methods to consume and return tickets.
//!
//! The manager supports two modes:
//! - **Direct mode**: Every request hits the database directly
//! - **Pooled mode**: Pre-borrows tokens into an in-memory pool to reduce database contention

use std::sync::Arc;

use tracing::Span;

use super::token_pool::TokenPoolRegistry;
use super::{
    PoolMode, RateLimitResource, RateLimitResourceUsage, RateLimitedRequest, RateLimitingConfig,
    ScopeInfo, TicketBorrow, TicketBorrows,
};
use crate::db::postgres::PostgresConnectionInfo;
use crate::db::{
    ConsumeTicketsReceipt, ConsumeTicketsRequest, RateLimitQueries, ReturnTicketsRequest,
};
use crate::error::Error;

/// Manager for rate limiting operations.
///
/// This is the primary interface for consuming and returning rate limit tickets.
/// It wraps a `RateLimitingConfig` and coordinates all rate limiting operations.
pub struct RateLimitingManager {
    /// The rate limiting configuration
    config: Arc<RateLimitingConfig>,
    /// The database client for rate limiting operations
    client: Arc<dyn RateLimitQueries>,
    /// Registry for in-memory token pools (used in pooled mode)
    pool_registry: TokenPoolRegistry,
}

impl std::fmt::Debug for RateLimitingManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimitingManager")
            .field("config", &self.config)
            .field("client", &"<dyn RateLimitQueries>")
            .field("pool_registry", &self.pool_registry)
            .finish()
    }
}

impl RateLimitingManager {
    /// Create a new RateLimitingManager
    pub fn new(config: Arc<RateLimitingConfig>, client: PostgresConnectionInfo) -> Self {
        Self {
            config,
            client: Arc::new(client),
            pool_registry: TokenPoolRegistry::new(),
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
            pool_registry: TokenPoolRegistry::new(),
        }
    }

    /// Create a new, dummy RateLimitingManager for unit tests.
    #[cfg(test)]
    pub fn new_dummy() -> Self {
        Self {
            config: Arc::new(RateLimitingConfig::default()),
            client: Arc::new(PostgresConnectionInfo::Disabled),
            pool_registry: TokenPoolRegistry::new(),
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
        self.config.pool.mode
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
        let (receipts, served_from_pool) = match self.pool_mode() {
            PoolMode::Direct => {
                // Direct mode: hit the database for every request
                let receipts = self.client.consume_tickets(&ticket_requests).await?;
                (receipts, false)
            }
            PoolMode::Pooled => {
                // Pooled mode: try to consume from local pools first
                self.consume_from_pools(&limits, &ticket_requests).await?
            }
        };

        TicketBorrows::new(
            Arc::clone(self),
            receipts,
            limits,
            ticket_requests,
            served_from_pool,
        )
    }

    /// Consume tickets from the in-memory pools, replenishing from DB if needed.
    ///
    /// During warm-up (first 100 requests per key), requests go directly to the DB
    /// while building up P99 usage data in the histogram. After warm-up, requests
    /// are served from the local pool with adaptive pre-borrowing.
    ///
    /// Returns `(receipts, served_from_pool)` where `served_from_pool` indicates whether
    /// tickets were actually consumed from the in-memory pool (vs. direct DB access).
    async fn consume_from_pools(
        &self,
        limits: &[super::ActiveRateLimit],
        ticket_requests: &[ConsumeTicketsRequest],
    ) -> Result<(Vec<ConsumeTicketsReceipt>, bool), Error> {
        // Get or create pools for all limits
        let pools: Vec<Arc<super::token_pool::TokenPool>> = limits
            .iter()
            .map(|limit| self.pool_registry.get_or_create(limit))
            .collect();

        // Check if any pool is still in warm-up phase
        let any_warming_up = pools.iter().any(|p| p.is_warming_up());

        if any_warming_up {
            // Warm-up phase: go directly to DB while building histogram data.
            // Increment request count for each pool (even if rate limit fails).
            for pool in &pools {
                pool.increment_request_count();
            }
            let receipts = self.client.consume_tickets(ticket_requests).await?;

            // Update tickets_remaining for adaptive borrow cap calculation.
            // Also record estimated usage so the histogram has data for post-warmup borrowing.
            for ((pool, receipt), request) in pools
                .iter()
                .zip(receipts.iter())
                .zip(ticket_requests.iter())
            {
                pool.update_tickets_remaining(receipt.tickets_remaining);
                // Record estimated usage for this request. We use the requested amount
                // as the estimate since actual usage isn't known yet. This ensures the
                // histogram has data for rate-aware borrowing after warm-up completes.
                let (tokens, model_inferences) = match pool.limit.resource {
                    super::RateLimitResource::Token => (request.requested, 0),
                    super::RateLimitResource::ModelInference => (0, request.requested),
                };
                pool.record_usage(tokens, model_inferences);
            }

            return Ok((receipts, false)); // served_from_pool = false (warmup)
        }

        // All pools warmed up: use pooled consumption
        let mut consumed_pools = Vec::with_capacity(limits.len());
        let mut consumed_amounts = Vec::with_capacity(limits.len());

        for ((limit, request), pool) in limits.iter().zip(ticket_requests.iter()).zip(pools.iter())
        {
            // Check if we need to replenish before consuming
            if pool.needs_replenishment() {
                self.replenish_pool(limit, pool).await?;
            }

            // Try to consume from the pool
            if pool.try_consume(request.requested) {
                consumed_pools.push(Arc::clone(pool));
                consumed_amounts.push(request.requested);
            } else {
                // Rollback any successful consumptions before this one
                for (prev_pool, amount) in consumed_pools.iter().zip(consumed_amounts.iter()) {
                    prev_pool.rollback_consume(*amount);
                }

                // Fall back to direct DB access for this request
                // This ensures we can still serve the request even if the pool is exhausted
                let receipts = self.client.consume_tickets(ticket_requests).await?;
                return Ok((receipts, false)); // served_from_pool = false (fallback)
            }
        }

        // All consumptions succeeded from local pools
        // Build synthetic receipts for the successful consumptions
        let receipts: Vec<ConsumeTicketsReceipt> = ticket_requests
            .iter()
            .map(|r| ConsumeTicketsReceipt {
                key: r.key.clone(),
                success: true,
                // Note: This value is inaccurate (assumes bucket was at capacity) but is unused.
                // Adaptive borrow cap uses real DB responses only; error reporting only triggers
                // on failed receipts; return_tickets only uses tickets_consumed.
                tickets_remaining: r.capacity.saturating_sub(r.requested),
                tickets_consumed: r.requested,
            })
            .collect();

        Ok((receipts, true)) // served_from_pool = true
    }

    /// Replenish a pool by borrowing tokens from the database.
    async fn replenish_pool(
        &self,
        limit: &super::ActiveRateLimit,
        pool: &super::token_pool::TokenPool,
    ) -> Result<(), Error> {
        // Try to acquire the replenishment lock. If someone else is replenishing,
        // wait for them and check if we still need to replenish.
        loop {
            if !pool
                .replenish_in_progress
                .swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                // We acquired the lock, proceed with replenishment
                break;
            }

            // Another task is already replenishing, wait for it.
            // Use tokio::select! with a timeout to handle the race where
            // signal_replenishment_complete() is called before we start waiting.
            tokio::select! {
                () = pool.wait_for_replenishment() => {}
                () = tokio::time::sleep(std::time::Duration::from_millis(10)) => {}
            }

            // Check if the pool has enough tokens now
            if !pool.needs_replenishment() {
                return Ok(());
            }
            // Still needs replenishment, try to acquire the lock again
        }

        // Calculate how much to borrow
        let borrow_amount = pool.calculate_borrow_amount();

        // Borrow from the database
        let request = ConsumeTicketsRequest {
            key: limit.key.clone(),
            capacity: limit.limit.capacity,
            refill_amount: limit.limit.refill_rate,
            refill_interval: limit.limit.interval.to_pg_interval(),
            requested: borrow_amount,
        };

        let result = self.client.consume_tickets(&[request]).await;

        match result {
            Ok(receipts) => {
                if let Some(receipt) = receipts.first() {
                    // Always update tickets_remaining for adaptive cap calculation
                    pool.update_tickets_remaining(receipt.tickets_remaining);

                    if receipt.success {
                        pool.add_tokens(receipt.tickets_consumed);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to replenish token pool: {e}");
            }
        }

        pool.signal_replenishment_complete();
        Ok(())
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

        // Record usage for P99 tracking and adjust pool accounting.
        // Only adjust pool accounting if tickets were actually consumed from the pool
        // (not during warmup or DB fallback).
        if self.pool_mode() == PoolMode::Pooled {
            let served_from_pool = ticket_borrows.served_from_pool();
            for borrow in ticket_borrows.borrows() {
                let pool = self.pool_registry.get_or_create(&borrow.active_limit);
                pool.record_usage(tokens, model_inferences);

                if served_from_pool {
                    // Adjust pool accounting based on actual vs estimated usage
                    let actual_usage_this_request = match borrow.active_limit.limit.resource {
                        RateLimitResource::ModelInference => model_inferences,
                        RateLimitResource::Token => tokens,
                    };
                    pool.adjust_usage(borrow.receipt.tickets_consumed, actual_usage_this_request);
                }
            }
        }

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
        if self.pool_mode() == PoolMode::Direct || self.pool_registry.is_empty() {
            return Ok(());
        }

        // Collect all unused tokens from pools and return them to the database
        let mut return_requests = Vec::new();

        for (key, pool) in self.pool_registry.iter() {
            let unused = pool.unused_tokens();
            if unused > 0 {
                return_requests.push(ReturnTicketsRequest {
                    key: super::ActiveRateLimitKey::new(key),
                    capacity: pool.limit.capacity,
                    refill_amount: pool.limit.refill_rate,
                    refill_interval: pool.limit.interval.to_pg_interval(),
                    returned: unused,
                });
            }
        }

        if !return_requests.is_empty() {
            // Use a timeout to avoid blocking shutdown indefinitely
            let timeout = self.config.pool.shutdown_timeout();
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
        EstimatedRateLimitResourceUsage, PoolConfig, RateLimit, RateLimitInterval,
        RateLimitingConfigPriority, RateLimitingConfigRule, RateLimitingConfigScopes,
    };
    use std::collections::HashMap;

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
            pool: PoolConfig::default(),
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
            pool: PoolConfig {
                mode: PoolMode::Direct,
                ..PoolConfig::default()
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
            pool: PoolConfig::default(),
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
            pool: PoolConfig {
                mode: PoolMode::Pooled,
                ..PoolConfig::default()
            },
        })
    }

    /// Create a mock client that tracks call counts and returns successful responses.
    /// Returns (mock, call_counter) where call_counter can be used to verify DB calls.
    fn create_mock_client_with_counter() -> (MockRateLimitQueries, Arc<std::sync::atomic::AtomicU32>)
    {
        use std::sync::atomic::{AtomicU32, Ordering};

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
    async fn test_pooled_mode_warmup_hits_db_each_request() {
        // During warm-up (first 100 requests), each request should hit the DB
        let (mock, consume_count) = create_mock_client_with_counter();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(10000)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_pooled_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);

        // Make 10 requests during warm-up
        for i in 0..10 {
            let request = MockRateLimitedRequest {
                tokens: 10,
                model_inferences: 1,
            };
            let result = manager.consume_tickets(&scope_info, &request).await;
            assert!(result.is_ok(), "Request {i} should succeed during warm-up");
        }

        // Each warm-up request should have hit the DB
        assert_eq!(
            consume_count.load(std::sync::atomic::Ordering::SeqCst),
            10,
            "Each warm-up request should hit the DB"
        );
    }

    #[tokio::test]
    async fn test_pooled_mode_after_warmup_serves_from_pool() {
        // After warm-up completes, requests should be served from the in-memory pool
        // without hitting the DB (until replenishment is needed)
        let (mock, consume_count) = create_mock_client_with_counter();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1_000_000)], // Large capacity
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = make_pooled_config_with_rule(rule);
        let manager = Arc::new(RateLimitingManager::new_with_client(config, Arc::new(mock)));
        let scope_info = make_scope_info(vec![]);

        // Complete warm-up phase (100 requests)
        for _ in 0..100 {
            let request = MockRateLimitedRequest {
                tokens: 10,
                model_inferences: 1,
            };
            let _ = manager.consume_tickets(&scope_info, &request).await;
        }

        let warmup_db_calls = consume_count.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(warmup_db_calls, 100, "Warm-up should make 100 DB calls");

        // After warm-up, first post-warmup request triggers one replenishment
        let request = MockRateLimitedRequest {
            tokens: 10,
            model_inferences: 1,
        };
        let result = manager.consume_tickets(&scope_info, &request).await;
        assert!(result.is_ok(), "Post-warmup request should succeed");

        let after_first_post_warmup =
            consume_count.load(std::sync::atomic::Ordering::SeqCst) - warmup_db_calls;
        assert_eq!(
            after_first_post_warmup, 1,
            "First post-warmup request should trigger one replenishment"
        );

        // Subsequent requests should be served from pool without DB calls
        // (as long as pool has capacity)
        let db_calls_before = consume_count.load(std::sync::atomic::Ordering::SeqCst);
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

        // Complete warm-up and get a borrow
        for _ in 0..100 {
            let request = MockRateLimitedRequest {
                tokens: 10,
                model_inferences: 1,
            };
            let _ = manager.consume_tickets(&scope_info, &request).await;
        }

        // Get one more borrow after warm-up
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
}
