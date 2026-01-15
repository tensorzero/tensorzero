use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use rlt::{BenchSuite, IterInfo, IterReport, Status};
use sqlx::PgPool;
use sqlx::postgres::types::PgInterval;
use tensorzero_core::{
    db::{ConsumeTicketsRequest, RateLimitQueries, postgres::PostgresConnectionInfo},
    rate_limiting::{
        ActiveRateLimitKey, PoolConfig, PoolMode, RateLimitInterval, RateLimitResource,
        RateLimitingConfig, pool::TokenPoolManager,
    },
};
use tokio::time::Instant;

use crate::BenchmarkMode;

/// If contention is Full, we try to maximally contend for keys by using rate_limit_key_{0, ..., requests_per_iteration - 1}
/// Otherwise we use the atomic counter to cycle through available keys up to NumKeys
///
/// If requests_per_iteration > NumKeys, this will fail
/// (this is not a concern in production since the keys are determined by the semantics of rate limiting rather than arbitrarily)
#[derive(Clone)]
pub enum Contention {
    Full,
    NumKeys(usize),
}

impl Contention {
    pub fn new(num_keys: usize) -> Self {
        if num_keys == 0 {
            Contention::Full
        } else {
            Contention::NumKeys(num_keys)
        }
    }

    pub fn get_key(&self, batch_index: usize, global_counter: u64) -> String {
        match self {
            Contention::Full => {
                // For full contention, use batch index to avoid duplicates within a batch
                format!("rate_limit_key_{batch_index}")
            }
            Contention::NumKeys(num) => {
                // For limited keys, use global counter to cycle through available keys
                let key_index = global_counter % (*num as u64);
                format!("rate_limit_key_{key_index}")
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct BucketSettings {
    pub capacity: i64,
    pub refill_amount: i64,
    pub interval: PgInterval,
}

#[derive(Clone)]
pub struct RateLimitBenchmark {
    pub client: PostgresConnectionInfo,
    pub bucket_settings: Arc<BucketSettings>,
    pub contention: Contention,
    pub tickets_per_request: u64,
    pub requests_per_iteration: usize,
    pub request_counter: Arc<AtomicU64>,
    pub mode: BenchmarkMode,
    pub token_pool_manager: Option<Arc<TokenPoolManager>>,
}

impl RateLimitBenchmark {
    pub fn new(
        client: PostgresConnectionInfo,
        bucket_settings: Arc<BucketSettings>,
        contention: Contention,
        tickets_per_request: u64,
        requests_per_iteration: usize,
        mode: BenchmarkMode,
    ) -> Self {
        let token_pool_manager = match mode {
            BenchmarkMode::Direct => None,
            BenchmarkMode::Pooled => {
                // Create a RateLimitingConfig with pooled mode
                let pool_config = PoolConfig {
                    mode: PoolMode::Pooled,
                    ..Default::default()
                };
                let config = RateLimitingConfig::new_for_load_test(
                    bucket_settings.capacity as u64,
                    bucket_settings.refill_amount as u64,
                    RateLimitInterval::Second,
                    pool_config,
                );
                Some(Arc::new(TokenPoolManager::new(Arc::new(config))))
            }
        };

        Self {
            client,
            bucket_settings,
            contention,
            tickets_per_request,
            requests_per_iteration,
            request_counter: Arc::new(AtomicU64::new(0)),
            mode,
            token_pool_manager,
        }
    }
}

pub struct WorkerState {
    pub client: PostgresConnectionInfo,
    pub bucket_settings: Arc<BucketSettings>,
    pub token_pool_manager: Option<Arc<TokenPoolManager>>,
}

#[async_trait]
impl BenchSuite for RateLimitBenchmark {
    type WorkerState = WorkerState;

    async fn state(&self, _worker_id: u32) -> Result<Self::WorkerState> {
        Ok(WorkerState {
            client: self.client.clone(),
            bucket_settings: self.bucket_settings.clone(),
            token_pool_manager: self.token_pool_manager.clone(),
        })
    }

    async fn bench(
        &mut self,
        state: &mut Self::WorkerState,
        _info: &IterInfo,
    ) -> Result<IterReport> {
        match self.mode {
            BenchmarkMode::Direct => self.bench_direct(state).await,
            BenchmarkMode::Pooled => self.bench_pooled(state).await,
        }
    }
}

impl RateLimitBenchmark {
    /// Direct mode: hit the database for every request
    async fn bench_direct(&mut self, state: &mut WorkerState) -> Result<IterReport> {
        // Create multiple requests, each with a unique key
        let requests: Vec<ConsumeTicketsRequest> = (0..self.requests_per_iteration)
            .map(|i| {
                let global_counter = self
                    .request_counter
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let key = self.contention.get_key(i, global_counter);
                ConsumeTicketsRequest {
                    key: ActiveRateLimitKey(key),
                    requested: self.tickets_per_request,
                    capacity: state.bucket_settings.capacity as u64,
                    refill_amount: state.bucket_settings.refill_amount as u64,
                    refill_interval: state.bucket_settings.interval,
                }
            })
            .collect();

        let start = Instant::now();
        let result = state.client.consume_tickets(&requests).await;
        let duration = start.elapsed();

        match result {
            Ok(responses) => {
                // Count successful requests
                let successful = responses.iter().filter(|r| r.success).count();
                let total = responses.len();

                if successful != total && successful != 0 {
                    tracing::error!(
                        "Partial success in rate limiting: {} out of {} requests succeeded. This should not happen.",
                        successful,
                        total
                    );
                    return Ok(IterReport {
                        duration,
                        status: Status::error(500),
                        bytes: 0,
                        items: 0,
                    });
                }

                Ok(IterReport {
                    duration,
                    status: if successful == total {
                        Status::success(200)
                    } else {
                        Status::error(429) // All rate limited
                    },
                    bytes: 0,
                    items: successful as u64,
                })
            }
            Err(_e) => Ok(IterReport {
                duration,
                status: Status::error(500),
                bytes: 0,
                items: 0,
            }),
        }
    }

    /// Pooled mode: use TokenPoolManager with in-memory pool
    async fn bench_pooled(&mut self, state: &mut WorkerState) -> Result<IterReport> {
        let pool_manager = state
            .token_pool_manager
            .as_ref()
            .ok_or_else(|| anyhow!("TokenPoolManager not initialized for pooled mode"))?;

        let mut successful = 0usize;
        let mut total = 0usize;

        let start = Instant::now();

        for i in 0..self.requests_per_iteration {
            let global_counter = self
                .request_counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let key = self.contention.get_key(i, global_counter);

            // Create a synthetic request for the pool manager
            let request = LoadTestRequest {
                key,
                tokens: self.tickets_per_request,
            };

            let result = pool_manager
                .consume_tickets(&state.client, &request.scope_info(), &request)
                .await;

            total += 1;
            if result.is_ok() {
                successful += 1;
                // In a real scenario, we'd call return_tickets after the operation completes
                // For the benchmark, we skip this as we're measuring consume performance
            }
        }

        let duration = start.elapsed();

        if successful != total && successful != 0 {
            tracing::error!(
                "Partial success in rate limiting: {} out of {} requests succeeded.",
                successful,
                total
            );
            return Ok(IterReport {
                duration,
                status: Status::error(500),
                bytes: 0,
                items: 0,
            });
        }

        Ok(IterReport {
            duration,
            status: if successful == total {
                Status::success(200)
            } else {
                Status::error(429)
            },
            bytes: 0,
            items: successful as u64,
        })
    }
}

/// A synthetic request for load testing
struct LoadTestRequest {
    key: String,
    tokens: u64,
}

impl LoadTestRequest {
    fn scope_info(&self) -> tensorzero_core::rate_limiting::ScopeInfo {
        tensorzero_core::rate_limiting::ScopeInfo::new_for_load_test(&self.key)
    }
}

impl tensorzero_core::rate_limiting::RateLimitedRequest for LoadTestRequest {
    fn estimated_resource_usage(
        &self,
        _resources: &[RateLimitResource],
    ) -> Result<
        tensorzero_core::rate_limiting::EstimatedRateLimitResourceUsage,
        tensorzero_core::error::Error,
    > {
        Ok(
            tensorzero_core::rate_limiting::EstimatedRateLimitResourceUsage {
                tokens: Some(self.tokens),
                model_inferences: Some(1),
            },
        )
    }
}

pub async fn create_postgres_pool(pool_size: u32) -> Result<PgPool> {
    let database_url = std::env::var("TENSORZERO_POSTGRES_URL").unwrap_or_else(|_| {
        "postgres://postgres:postgres@localhost:5432/tensorzero_e2e_tests".to_string()
    });

    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(pool_size)
        .connect(&database_url)
        .await
        .map_err(|e| anyhow!("Failed to connect to database: {e}"))?;

    Ok(pool)
}

pub fn create_bucket_settings(capacity: i64, refill_amount: i64) -> BucketSettings {
    BucketSettings {
        capacity,
        refill_amount,
        interval: PgInterval {
            months: 0,
            days: 0,
            microseconds: 1_000_000, // 1 second
        },
    }
}
