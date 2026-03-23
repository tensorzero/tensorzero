//! DB pressure load test for durable execution.
//!
//! Measures connection pool behavior, heartbeat overhead, and claim_task latency
//! under concurrent durable task execution. Run with:
//!
//!   cargo test --test db_pressure -- --nocapture
//!
//! Requires Postgres (uses `#[sqlx::test]` with the durable migrator).
#![expect(clippy::expect_used, clippy::disallowed_methods)]

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use durable::MIGRATOR;
use durable::WorkerOptions;
use durable_tools::{
    MockTensorZeroClient, TaskTool, TensorZeroClient, TensorZeroClientError, ToolContext,
    ToolExecutor, ToolMetadata, ToolResult,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tokio::sync::Barrier;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_test_writer()
        .try_init();
}

// ============================================================================
// Metrics collection
// ============================================================================

#[derive(Default)]
struct Metrics {
    tasks_completed: AtomicU64,
    tasks_failed: AtomicU64,
    heartbeats_sent: AtomicU64,
    steps_executed: AtomicU64,
    total_step_duration_us: AtomicU64,
    max_step_duration_us: AtomicU64,
}

impl Metrics {
    fn record_step(&self, duration: Duration) {
        self.steps_executed.fetch_add(1, Ordering::Relaxed);
        let us = duration.as_micros() as u64;
        self.total_step_duration_us.fetch_add(us, Ordering::Relaxed);
        self.max_step_duration_us.fetch_max(us, Ordering::Relaxed);
    }

    fn log_summary(&self, label: &str) {
        let completed = self.tasks_completed.load(Ordering::Relaxed);
        let failed = self.tasks_failed.load(Ordering::Relaxed);
        let heartbeats = self.heartbeats_sent.load(Ordering::Relaxed);
        let steps = self.steps_executed.load(Ordering::Relaxed);
        let total_us = self.total_step_duration_us.load(Ordering::Relaxed);
        let max_us = self.max_step_duration_us.load(Ordering::Relaxed);
        let avg_us = if steps > 0 { total_us / steps } else { 0 };

        tracing::info!(
            label,
            completed,
            failed,
            steps,
            avg_step_us = avg_us,
            max_step_us = max_us,
            heartbeats,
            "metrics"
        );
    }
}

// ============================================================================
// Test tool: simulates work with configurable steps and heartbeats
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct PressureParams {
    num_steps: u32,
    work_ms: u64,
    heartbeat: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PressureOutput {
    steps_done: u32,
}

struct PressureTaskTool {
    metrics: Arc<Metrics>,
}

impl ToolMetadata for PressureTaskTool {
    type SideInfo = ();
    type Output = PressureOutput;
    type LlmParams = PressureParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("pressure")
    }
    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("DB pressure test tool")
    }
    fn timeout(&self) -> Duration {
        Duration::from_secs(300)
    }
    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::UNIT
    }
    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle_type_name() -> String {
        "void".to_string()
    }
    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::UNIT
    }
    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle_type_name() -> String {
        "void".to_string()
    }
}

#[async_trait]
impl TaskTool for PressureTaskTool {
    type ExtraState = ();

    async fn execute(
        &self,
        llm_params: PressureParams,
        _side_info: (),
        ctx: &mut ToolContext,
    ) -> ToolResult<PressureOutput> {
        for i in 0..llm_params.num_steps {
            if llm_params.work_ms > 0 {
                tokio::time::sleep(Duration::from_millis(llm_params.work_ms)).await;
            }

            // Checkpointed step — hits DB via set_task_checkpoint_state
            let step_start = Instant::now();
            let _: u32 = ctx
                .step(
                    &format!("work_{i}"),
                    i,
                    |i, _step_state| async move { Ok(i) },
                )
                .await?;
            self.metrics.record_step(step_start.elapsed());

            // Optional heartbeat — hits DB via extend_claim (FOR UPDATE)
            if llm_params.heartbeat {
                ctx.heartbeat(Some(Duration::from_secs(60))).await?;
                self.metrics.heartbeats_sent.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.metrics.tasks_completed.fetch_add(1, Ordering::Relaxed);
        Ok(PressureOutput {
            steps_done: llm_params.num_steps,
        })
    }
}

// ============================================================================
// Scenario runner
// ============================================================================

struct ScenarioConfig {
    num_tasks: u32,
    num_steps: u32,
    work_ms: u64,
    heartbeat: bool,
    worker_concurrency: usize,
    worker_poll_ms: u64,
}

struct ScenarioResult {
    wall_time: Duration,
    metrics: Arc<Metrics>,
    pool_size: u32,
}

async fn run_scenario(pool: PgPool, config: ScenarioConfig) -> ScenarioResult {
    let queue_name = format!("pressure_{}", Uuid::now_v7());
    let metrics = Arc::new(Metrics::default());

    let mut mock = MockTensorZeroClient::new();
    mock.expect_inference()
        .returning(|_| Box::pin(async { Err(TensorZeroClientError::StreamingNotSupported) }));
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock);

    let executor = ToolExecutor::builder(())
        .pool(pool.clone())
        .queue_name(&queue_name)
        .t0_client(t0_client)
        .register_task_tool_instance(PressureTaskTool {
            metrics: metrics.clone(),
        })
        .expect("register failed")
        .build()
        .await
        .expect("build failed");

    executor
        .durable()
        .create_queue(None)
        .await
        .expect("create_queue failed");

    for _ in 0..config.num_tasks {
        executor
            .spawn_tool_by_name(
                "pressure",
                serde_json::json!({
                    "num_steps": config.num_steps,
                    "work_ms": config.work_ms,
                    "heartbeat": config.heartbeat,
                }),
                serde_json::json!(null),
                Uuid::now_v7(),
            )
            .await
            .expect("spawn failed");
    }

    let wall_start = Instant::now();

    let worker = executor
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(config.worker_poll_ms),
            concurrency: config.worker_concurrency,
            claim_timeout: Duration::from_secs(120),
            ..Default::default()
        })
        .await
        .expect("start_worker failed");

    let expected = config.num_tasks as u64;
    let deadline = Instant::now() + Duration::from_secs(120);

    loop {
        let completed = metrics.tasks_completed.load(Ordering::Relaxed);
        let failed = metrics.tasks_failed.load(Ordering::Relaxed);
        if completed + failed >= expected {
            break;
        }
        if Instant::now() > deadline {
            tracing::warn!(completed, expected, failed, "scenario timed out");
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let wall_time = wall_start.elapsed();
    worker.shutdown().await;
    let pool_size = pool.size();

    ScenarioResult {
        wall_time,
        metrics,
        pool_size,
    }
}

// ============================================================================
// Pool probe
// ============================================================================

async fn measure_pool_acquisition_latency(pool: &PgPool, num_samples: u32) -> (Duration, Duration) {
    let mut total = Duration::ZERO;
    let mut max = Duration::ZERO;

    for _ in 0..num_samples {
        let start = Instant::now();
        let _row: (i32,) = sqlx::query_as("SELECT 1")
            .fetch_one(pool)
            .await
            .expect("pool acquisition failed");
        let elapsed = start.elapsed();
        total += elapsed;
        if elapsed > max {
            max = elapsed;
        }
    }

    let avg = total / num_samples;
    (avg, max)
}

// ============================================================================
// Tests
// ============================================================================

/// Baseline: pool acquisition latency with no durable load.
#[sqlx::test(migrator = "MIGRATOR")]
async fn pool_acquisition_baseline(pool: PgPool) -> sqlx::Result<()> {
    init_tracing();
    let (avg, max) = measure_pool_acquisition_latency(&pool, 100).await;
    tracing::info!(
        ?avg,
        ?max,
        pool_size = pool.size(),
        "pool_acquisition_baseline"
    );
    Ok(())
}

/// Pool acquisition latency WHILE durable tasks are running.
#[sqlx::test(migrator = "MIGRATOR")]
async fn pool_acquisition_under_durable_load(pool: PgPool) -> sqlx::Result<()> {
    init_tracing();
    let pool_for_probe = pool.clone();

    let scenario_handle = tokio::spawn(run_scenario(
        pool.clone(),
        ScenarioConfig {
            num_tasks: 20,
            num_steps: 10,
            work_ms: 5,
            heartbeat: true,
            worker_concurrency: 8,
            worker_poll_ms: 100,
        },
    ));

    // Let the worker start claiming
    tokio::time::sleep(Duration::from_millis(200)).await;

    let (avg, max) = measure_pool_acquisition_latency(&pool_for_probe, 50).await;
    tracing::info!(?avg, ?max, "pool_acquisition_under_load");

    let result = scenario_handle.await.expect("scenario panicked");
    tracing::info!(
        ?result.wall_time,
        pool_size = result.pool_size,
        "scenario_complete"
    );
    result.metrics.log_summary("under_load");
    Ok(())
}

/// A/B: heartbeat overhead — same workload with and without heartbeats.
#[sqlx::test(migrator = "MIGRATOR")]
async fn heartbeat_overhead_comparison(pool: PgPool) -> sqlx::Result<()> {
    init_tracing();
    let num_tasks: u32 = 30;
    let num_steps: u32 = 10;
    let concurrency: usize = 8;

    let no_hb = run_scenario(
        pool.clone(),
        ScenarioConfig {
            num_tasks,
            num_steps,
            work_ms: 0,
            heartbeat: false,
            worker_concurrency: concurrency,
            worker_poll_ms: 50,
        },
    )
    .await;

    let with_hb = run_scenario(
        pool.clone(),
        ScenarioConfig {
            num_tasks,
            num_steps,
            work_ms: 0,
            heartbeat: true,
            worker_concurrency: concurrency,
            worker_poll_ms: 50,
        },
    )
    .await;

    let overhead_pct = if no_hb.wall_time.as_millis() > 0 {
        ((with_hb.wall_time.as_millis() as f64 / no_hb.wall_time.as_millis() as f64) - 1.0) * 100.0
    } else {
        0.0
    };

    tracing::info!(
        num_tasks,
        num_steps,
        concurrency,
        no_hb_ms = no_hb.wall_time.as_millis() as u64,
        with_hb_ms = with_hb.wall_time.as_millis() as u64,
        overhead_pct = format!("{overhead_pct:.1}"),
        "heartbeat_overhead"
    );
    no_hb.metrics.log_summary("without_heartbeat");
    with_hb.metrics.log_summary("with_heartbeat");
    Ok(())
}

/// Stress: many tasks, high concurrency, step latency distribution.
#[sqlx::test(migrator = "MIGRATOR")]
async fn step_latency_under_contention(pool: PgPool) -> sqlx::Result<()> {
    init_tracing();
    let result = run_scenario(
        pool.clone(),
        ScenarioConfig {
            num_tasks: 50,
            num_steps: 20,
            work_ms: 0,
            heartbeat: false,
            worker_concurrency: 16,
            worker_poll_ms: 50,
        },
    )
    .await;

    let completed = result.metrics.tasks_completed.load(Ordering::Relaxed);
    let steps = result.metrics.steps_executed.load(Ordering::Relaxed);
    let tasks_per_sec = completed as f64 / result.wall_time.as_secs_f64();
    let steps_per_sec = steps as f64 / result.wall_time.as_secs_f64();

    tracing::info!(
        wall_ms = result.wall_time.as_millis() as u64,
        pool_size = result.pool_size,
        tasks_per_sec = format!("{tasks_per_sec:.1}"),
        steps_per_sec = format!("{steps_per_sec:.1}"),
        "step_latency_under_contention"
    );
    result.metrics.log_summary("contention");
    Ok(())
}

/// GEPA-like: many steps per task, all heartbeating, high concurrency.
#[sqlx::test(migrator = "MIGRATOR")]
async fn gepa_like_sequential_steps_with_heartbeat(pool: PgPool) -> sqlx::Result<()> {
    init_tracing();
    let result = run_scenario(
        pool.clone(),
        ScenarioConfig {
            num_tasks: 8,
            num_steps: 50,
            work_ms: 10,
            heartbeat: true,
            worker_concurrency: 8,
            worker_poll_ms: 100,
        },
    )
    .await;

    let heartbeats = result.metrics.heartbeats_sent.load(Ordering::Relaxed);
    let steps = result.metrics.steps_executed.load(Ordering::Relaxed);
    let db_calls = steps + heartbeats;
    let db_calls_per_sec = db_calls as f64 / result.wall_time.as_secs_f64();

    tracing::info!(
        wall_ms = result.wall_time.as_millis() as u64,
        steps,
        heartbeats,
        total_db_calls = db_calls,
        db_calls_per_sec = format!("{db_calls_per_sec:.1}"),
        "gepa_like_pattern"
    );
    result.metrics.log_summary("gepa_like");
    Ok(())
}

/// Claim latency: time from spawn to first execution.
#[sqlx::test(migrator = "MIGRATOR")]
async fn claim_task_latency(pool: PgPool) -> sqlx::Result<()> {
    init_tracing();
    let queue_name = format!("claim_test_{}", Uuid::now_v7());
    let metrics = Arc::new(Metrics::default());
    let claim_barrier = Arc::new(Barrier::new(2));

    let mut mock = MockTensorZeroClient::new();
    mock.expect_inference()
        .returning(|_| Box::pin(async { Err(TensorZeroClientError::StreamingNotSupported) }));
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock);

    let barrier_clone = claim_barrier.clone();
    let metrics_clone = metrics.clone();

    struct ClaimTimingTool {
        barrier: Arc<Barrier>,
        metrics: Arc<Metrics>,
    }

    impl ToolMetadata for ClaimTimingTool {
        type SideInfo = ();
        type Output = PressureOutput;
        type LlmParams = PressureParams;
        fn name(&self) -> Cow<'static, str> {
            Cow::Borrowed("claim_timing")
        }
        fn description(&self) -> Cow<'static, str> {
            Cow::Borrowed("Measures claim latency")
        }
        fn timeout(&self) -> Duration {
            Duration::from_secs(60)
        }
        #[cfg(feature = "ts-bindings")]
        fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
            tensorzero_ts_types::UNIT
        }
        #[cfg(feature = "ts-bindings")]
        fn llm_params_ts_bundle_type_name() -> String {
            "void".to_string()
        }
        #[cfg(feature = "ts-bindings")]
        fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
            tensorzero_ts_types::UNIT
        }
        #[cfg(feature = "ts-bindings")]
        fn output_ts_bundle_type_name() -> String {
            "void".to_string()
        }
    }

    #[async_trait]
    impl TaskTool for ClaimTimingTool {
        type ExtraState = ();
        async fn execute(
            &self,
            _llm_params: PressureParams,
            _side_info: (),
            _ctx: &mut ToolContext,
        ) -> ToolResult<PressureOutput> {
            self.barrier.wait().await;
            self.metrics.tasks_completed.fetch_add(1, Ordering::Relaxed);
            Ok(PressureOutput { steps_done: 0 })
        }
    }

    let executor = ToolExecutor::builder(())
        .pool(pool.clone())
        .queue_name(&queue_name)
        .t0_client(t0_client)
        .register_task_tool_instance(ClaimTimingTool {
            barrier: barrier_clone,
            metrics: metrics_clone,
        })
        .expect("register failed")
        .build()
        .await
        .expect("build failed");

    executor
        .durable()
        .create_queue(None)
        .await
        .expect("create_queue failed");

    let worker = executor
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(50),
            concurrency: 1,
            claim_timeout: Duration::from_secs(30),
            ..Default::default()
        })
        .await
        .expect("start_worker failed");

    let spawn_time = Instant::now();
    executor
        .spawn_tool_by_name(
            "claim_timing",
            serde_json::json!({
                "num_steps": 0,
                "work_ms": 0,
                "heartbeat": false,
            }),
            serde_json::json!(null),
            Uuid::now_v7(),
        )
        .await
        .expect("spawn failed");

    tokio::time::timeout(Duration::from_secs(10), claim_barrier.wait())
        .await
        .expect("task was never claimed");

    let claim_latency = spawn_time.elapsed();
    tracing::info!(
        claim_latency_ms = claim_latency.as_millis() as u64,
        "claim_task_latency (includes poll_interval=50ms + claim_task query + context setup)"
    );

    worker.shutdown().await;
    Ok(())
}
