//! Rate limiting manager for coordinating rate limiting operations.
//!
//! The `RateLimitingManager` is the primary interface for rate limiting operations.
//! It wraps a `RateLimitingConfig` and provides methods to consume and return tickets.
//!
//! # Failure behavior
//!
//! This implementation uses **fail-closed** semantics: if rate limiting backend is unavailable
//! (connection error, timeout, etc.), rate limiting operations return errors,
//! which causes the gateway to reject requests. This prevents us sending expensive traffic
//! to LLM providers when the rate limiting backend is down.

use std::sync::Arc;

use tracing::Span;

use super::{
    RateLimitResource, RateLimitResourceUsage, RateLimitedRequest, RateLimitingBackend,
    RateLimitingConfig, ScopeInfo, TicketBorrow, TicketBorrows,
};
use crate::db::postgres::PostgresConnectionInfo;
use crate::db::rate_limiting::{ConsumeTicketsRequest, DisabledRateLimitQueries, RateLimitQueries};
use crate::db::valkey::ValkeyConnectionInfo;
use crate::error::{Error, ErrorDetails};

/// Manager for rate limiting operations.
///
/// This is the primary interface for consuming and returning rate limit tickets.
/// It wraps a `RateLimitingConfig` and coordinates all rate limiting operations.
pub struct RateLimitingManager {
    /// The rate limiting configuration
    config: Arc<RateLimitingConfig>,
    /// The database client for rate limiting operations
    client: Arc<dyn RateLimitQueries>,
}

impl std::fmt::Debug for RateLimitingManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimitingManager")
            .field("config", &self.config)
            .field("client", &"<dyn RateLimitQueries>")
            .finish()
    }
}

impl RateLimitingManager {
    /// Create a new RateLimitingManager with a database client
    pub fn new(config: Arc<RateLimitingConfig>, client: Arc<dyn RateLimitQueries>) -> Self {
        Self { config, client }
    }

    /// Create a new RateLimitingManager by selecting the appropriate backend.
    ///
    /// Backend selection is determined by `config.backend`:
    /// - `Auto`: Valkey if available, otherwise Postgres
    /// - `Postgres`: Force Postgres backend (error if unavailable)
    /// - `Valkey`: Force Valkey backend (error if unavailable)
    ///
    /// If rate limiting rules are configured but no backend is available, returns an error.
    pub fn new_from_connections(
        config: Arc<RateLimitingConfig>,
        valkey_connection_info: &ValkeyConnectionInfo,
        postgres_connection_info: &PostgresConnectionInfo,
    ) -> Result<Self, Error> {
        let valkey_available =
            matches!(valkey_connection_info, ValkeyConnectionInfo::Enabled { .. });

        // Postgres is considered available if it is not disabled.
        // In tests, it matches Mock as well.
        let postgres_available =
            !matches!(postgres_connection_info, PostgresConnectionInfo::Disabled);

        // If the backend is explicitly configured, use it if it is available.
        if config.backend == RateLimitingBackend::Valkey {
            if valkey_available {
                tracing::debug!("Using Valkey for rate limiting");
                return Ok(Self::new(config, Arc::new(valkey_connection_info.clone())));
            }
            return Err(Error::new(ErrorDetails::Config {
                message: "Rate limiting is configured to use Valkey, but Valkey is not available. Please check the environment variable `TENSORZERO_VALKEY_URL` is set.".to_string(),
            }));
        }

        if config.backend == RateLimitingBackend::Postgres {
            if postgres_available {
                tracing::debug!("Using Postgres for rate limiting");
                return Ok(Self::new(
                    config,
                    Arc::new(postgres_connection_info.clone()),
                ));
            }
            return Err(Error::new(ErrorDetails::Config {
                message: "Rate limiting is configured to use Postgres, but Postgres is not available. Please check the environment variable `TENSORZERO_POSTGRES_URL` is set.".to_string(),
            }));
        }

        // Otherwise, pick Valkey and Postgres in this order.
        if valkey_available {
            tracing::debug!("Using Valkey for rate limiting");
            return Ok(Self::new(config, Arc::new(valkey_connection_info.clone())));
        }
        if postgres_available {
            tracing::debug!("Using Postgres for rate limiting");
            return Ok(Self::new(
                config,
                Arc::new(postgres_connection_info.clone()),
            ));
        }

        // No backend available - this is only an error if rate limiting is enabled and rules are configured
        let rate_limiting_enabled = config.enabled() && !config.rules().is_empty();
        if !rate_limiting_enabled {
            return Ok(Self::new(config, Arc::new(DisabledRateLimitQueries)));
        }

        Err(Error::new(ErrorDetails::Config {
            message: "No rate limiting backend is available and rate limiting rules are configured. Please set either `TENSORZERO_VALKEY_URL` or `TENSORZERO_POSTGRES_URL` environment variable, or disable rate limiting.".to_string(),
        }))
    }

    /// Create a new, dummy RateLimitingManager for unit tests.
    #[cfg(test)]
    pub fn new_dummy() -> Self {
        Self {
            config: Arc::new(RateLimitingConfig::default()),
            client: Arc::new(PostgresConnectionInfo::Disabled),
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

        // Consume tickets directly from the database
        let ticket_requests: Result<Vec<ConsumeTicketsRequest>, Error> = limits
            .iter()
            .map(|limit| limit.get_consume_tickets_request(&rate_limit_resource_requests))
            .collect();
        let ticket_requests = ticket_requests?;

        let receipts = self.client.consume_tickets(&ticket_requests).await?;

        TicketBorrows::new(Arc::clone(self), receipts, limits, ticket_requests)
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
        match actual_usage {
            RateLimitResourceUsage::Exact {
                tokens,
                model_inferences,
            } => {
                span.record("actual_usage.tokens", tokens as i64);
                span.record("actual_usage.model_inferences", model_inferences as i64);
                span.record("underestimate", false);
            }
            RateLimitResourceUsage::UnderEstimate {
                tokens,
                model_inferences,
            } => {
                span.record("actual_usage.tokens", tokens as i64);
                span.record("actual_usage.model_inferences", model_inferences as i64);
                span.record("underestimate", true);
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

        let (consume_result, return_result) = tokio::join!(
            self.client.consume_tickets(&requests),
            self.client.return_tickets(returns)
        );

        consume_result?;
        return_result?;

        Ok(())
    }

    /// Shutdown the rate limiting manager.
    /// This is a no-op in direct mode (no tokens to return).
    pub fn shutdown(&self) -> Result<(), Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::postgres::PostgresConnectionInfo;
    use crate::db::{ConsumeTicketsReceipt, MockRateLimitQueries, ReturnTicketsReceipt};
    use crate::rate_limiting::{
        EstimatedRateLimitResourceUsage, RateLimit, RateLimitInterval, RateLimitingConfigPriority,
        RateLimitingConfigRule, RateLimitingConfigScopes,
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

    fn make_dummy_manager() -> RateLimitingManager {
        RateLimitingManager::new(
            Arc::new(RateLimitingConfig::default()),
            Arc::new(PostgresConnectionInfo::Disabled),
        )
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
            backend: RateLimitingBackend::default(),
        });
        let manager = RateLimitingManager::new(config, Arc::new(PostgresConnectionInfo::Disabled));

        assert!(
            !manager.is_empty(),
            "Manager with rules should not be empty"
        );
    }

    #[test]
    fn test_shutdown_returns_ok() {
        let manager = make_dummy_manager();
        let result = manager.shutdown();
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

    #[tokio::test]
    async fn test_consume_tickets_empty_rules() {
        // With empty rules, consume_tickets doesn't call the client
        let mock = MockRateLimitQueries::new();
        let config = Arc::new(RateLimitingConfig::default());
        let manager = Arc::new(RateLimitingManager::new(config, Arc::new(mock)));
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
            backend: RateLimitingBackend::default(),
        });
        let manager = Arc::new(RateLimitingManager::new(config, Arc::new(mock)));
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
    async fn test_consume_tickets_success() {
        let mock = create_mock_client_success();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1000)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = Arc::new(RateLimitingConfig {
            rules: vec![rule],
            enabled: true,
            backend: RateLimitingBackend::default(),
        });
        let manager = Arc::new(RateLimitingManager::new(config, Arc::new(mock)));
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
    async fn test_consume_tickets_rate_limit_exceeded() {
        let mock = create_mock_client_rate_limited();
        let rule = RateLimitingConfigRule {
            limits: vec![make_token_limit(1000)],
            scope: RateLimitingConfigScopes::new(vec![]).unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };
        let config = Arc::new(RateLimitingConfig {
            rules: vec![rule],
            enabled: true,
            backend: RateLimitingBackend::default(),
        });
        let manager = Arc::new(RateLimitingManager::new(config, Arc::new(mock)));
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
        let config = Arc::new(RateLimitingConfig {
            rules: vec![rule],
            enabled: true,
            backend: RateLimitingBackend::default(),
        });
        let manager = Arc::new(RateLimitingManager::new(config, Arc::new(mock)));
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
        let config = Arc::new(RateLimitingConfig {
            rules: vec![rule],
            enabled: true,
            backend: RateLimitingBackend::default(),
        });
        let manager = Arc::new(RateLimitingManager::new(config, Arc::new(mock)));
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
        let config = Arc::new(RateLimitingConfig {
            rules: vec![rule],
            enabled: true,
            backend: RateLimitingBackend::default(),
        });
        let manager = Arc::new(RateLimitingManager::new(config, Arc::new(mock)));
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
        let config = Arc::new(RateLimitingConfig {
            rules: vec![rule],
            enabled: true,
            backend: RateLimitingBackend::default(),
        });
        let manager = Arc::new(RateLimitingManager::new(config, Arc::new(mock)));
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
}
