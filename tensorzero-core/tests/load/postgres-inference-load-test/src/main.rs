use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicU64};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use benchmark::{BenchmarkConfig, InferenceBenchmark, RunStats};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

mod benchmark;

#[derive(Parser, Clone)]
pub struct Args {
    #[command(flatten)]
    pub bench_opts: rlt::cli::BenchCli,

    /// Base URL of a running TensorZero gateway
    #[arg(long, default_value = "http://localhost:3000")]
    pub gateway_url: String,

    /// Function name to invoke on /inference
    #[arg(long, default_value = "load_test_chat")]
    pub function_name: String,

    /// max_tokens to send in params.chat_completion.max_tokens
    #[arg(long, default_value_t = 128)]
    pub max_tokens: u32,

    /// Approximate prompt payload size (characters)
    #[arg(long, default_value_t = 256)]
    pub prompt_chars: usize,

    /// Add per-request suffixes to vary payload content
    #[arg(long, default_value_t = false)]
    pub randomize_prompt: bool,

    /// Optional run ID override. Defaults to UUIDv7.
    #[arg(long)]
    pub run_id: Option<String>,

    /// Additional run-scoped tag for grouping benchmark cases
    #[arg(long, default_value = "manual")]
    pub load_test_case: String,

    /// Wait after traffic stops before DB verification
    #[arg(long, default_value_t = 5000)]
    pub drain_wait_ms: u64,

    /// Postgres URL for verification queries (defaults to TENSORZERO_POSTGRES_URL)
    #[arg(long)]
    pub postgres_url: Option<String>,

    /// Maximum time to wait for DB parity checks after drain
    #[arg(long, default_value_t = 30)]
    pub verify_timeout_s: u64,

    /// Override target QPS for pass/fail. Defaults to -r/--rate value.
    #[arg(long)]
    pub target_qps: Option<f64>,

    /// Required achieved-QPS ratio to pass (achieved >= target * ratio)
    #[arg(long, default_value_t = 0.98)]
    pub min_achieved_qps_ratio: f64,

    /// Optional p99 latency threshold in milliseconds
    #[arg(long)]
    pub max_p99_latency_ms: Option<f64>,

    /// Maximum allowed error rate fraction (0.01 = 1%)
    #[arg(long, default_value_t = 0.01)]
    pub max_error_rate: f64,

    /// Optional output path for machine-readable benchmark summary JSON
    #[arg(long)]
    pub benchmark_report_json: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct RltReport {
    summary: RltSummary,
    latency: Option<RltLatency>,
}

#[derive(Debug, Deserialize)]
struct RltSummary {
    success_ratio: f64,
    iters: RltMetric,
    items: RltMetric,
}

#[derive(Debug, Deserialize)]
struct RltMetric {
    total: u64,
    rate: f64,
}

#[derive(Debug, Deserialize)]
struct RltLatency {
    percentiles: std::collections::HashMap<String, f64>,
}

#[derive(Debug, Serialize)]
struct DbMetrics {
    /// Minimum expected rows: successful responses received by the client
    min_expected_inference_rows: u64,
    /// Maximum expected rows: total requests sent (includes in-flight requests cancelled at benchmark end)
    max_expected_inference_rows: u64,
    inference_rows: i64,
    model_inference_rows: i64,
    verified_within_timeout: bool,
}

#[derive(Debug, Serialize)]
struct CheckResult {
    passed: bool,
    reason: String,
}

#[derive(Debug, Serialize)]
struct BenchmarkSummary {
    run_id: String,
    function_name: String,
    target_qps: f64,
    achieved_qps: f64,
    min_required_qps: f64,
    p99_latency_ms: Option<f64>,
    error_rate: f64,
    steady_state: CheckResult,
    p99_latency: CheckResult,
    error_rate_check: CheckResult,
    db_correctness: CheckResult,
    db_metrics: DbMetrics,
    rlt_iters_total: u64,
    rlt_success_total: u64,
    total_requests_sent: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    match args.bench_opts.collector() {
        rlt::cli::Collector::Tui => {
            tracing_subscriber::registry()
                .with(EnvFilter::from_default_env())
                .with(rlt::TuiTracingSubscriberLayer)
                .init();
        }
        rlt::cli::Collector::Silent => {
            tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::from_default_env())
                .init();
        }
    }

    validate_args(&args)?;
    preflight_flags()?;

    let run_id = args
        .run_id
        .clone()
        .unwrap_or_else(|| Uuid::now_v7().to_string());
    let target_qps = resolve_target_qps(&args)?;

    let inference_url = reqwest::Url::parse(&args.gateway_url)
        .with_context(|| format!("Invalid gateway URL: {}", args.gateway_url))?
        .join("inference")
        .context("Failed to build /inference URL from gateway URL")?;

    let run_stats = Arc::new(RunStats::default());
    let benchmark = InferenceBenchmark {
        config: Arc::new(BenchmarkConfig {
            inference_url,
            function_name: args.function_name.clone(),
            max_tokens: args.max_tokens,
            prompt_chars: args.prompt_chars,
            randomize_prompt: args.randomize_prompt,
            run_id: run_id.clone(),
            load_test_case: args.load_test_case.clone(),
        }),
        run_stats: run_stats.clone(),
        request_counter: Arc::new(AtomicU64::new(0)),
    };

    let rlt_report_path =
        std::env::temp_dir().join(format!("tensorzero-postgres-load-{run_id}.json"));
    let mut bench_opts = args.bench_opts.clone();
    bench_opts.output = rlt::cli::ReportFormat::Json;
    bench_opts.output_file = Some(rlt_report_path.clone());

    println!("Starting Postgres inference load test:");
    println!("  run_id: {run_id}");
    println!("  target_qps: {:.2}", target_qps);
    println!("  gateway: {}", args.gateway_url);
    println!("  function_name: {}", args.function_name);
    println!("  max_tokens: {}", args.max_tokens);
    println!("  prompt_chars: {}", args.prompt_chars);
    println!("  load_test_case: {}", args.load_test_case);
    println!();

    rlt::cli::run(bench_opts, benchmark).await?;

    let rlt_report: RltReport =
        serde_json::from_slice(&std::fs::read(&rlt_report_path).with_context(|| {
            format!(
                "Failed to read rlt report file: {}",
                rlt_report_path.display()
            )
        })?)
        .context("Failed to parse rlt JSON report")?;

    // Drain + DB verify phase: allow Ctrl+C to abort immediately
    let post_benchmark = async {
        tokio::time::sleep(Duration::from_millis(args.drain_wait_ms)).await;

        let postgres_url = resolve_postgres_url(args.postgres_url.clone())?;
        let successful_requests = run_stats
            .successful_requests
            .load(std::sync::atomic::Ordering::Relaxed);
        let total_requests = run_stats
            .total_requests
            .load(std::sync::atomic::Ordering::Relaxed);
        verify_db_parity(
            &postgres_url,
            &run_id,
            successful_requests,
            total_requests,
            Duration::from_secs(args.verify_timeout_s),
        )
        .await
    };

    let db_metrics = tokio::select! {
        biased;
        _ = tokio::signal::ctrl_c() => bail!("Interrupted during post-benchmark verification"),
        result = post_benchmark => result?,
    };

    let p99_latency_ms = rlt_report
        .latency
        .as_ref()
        .and_then(|latency| latency.percentiles.get("p99").copied())
        .map(|seconds| seconds * 1000.0);
    let error_rate = 1.0 - rlt_report.summary.success_ratio;
    let achieved_qps = rlt_report.summary.items.rate;
    let min_required_qps = target_qps * args.min_achieved_qps_ratio;

    let steady_state = if achieved_qps >= min_required_qps {
        CheckResult {
            passed: true,
            reason: format!(
                "Achieved QPS {:.2} >= required {:.2}",
                achieved_qps, min_required_qps
            ),
        }
    } else {
        CheckResult {
            passed: false,
            reason: format!(
                "Achieved QPS {:.2} < required {:.2}",
                achieved_qps, min_required_qps
            ),
        }
    };

    let p99_latency = match args.max_p99_latency_ms {
        Some(limit_ms) => match p99_latency_ms {
            Some(actual_ms) if actual_ms <= limit_ms => CheckResult {
                passed: true,
                reason: format!("p99 latency {:.2}ms <= {:.2}ms", actual_ms, limit_ms),
            },
            Some(actual_ms) => CheckResult {
                passed: false,
                reason: format!("p99 latency {:.2}ms > {:.2}ms", actual_ms, limit_ms),
            },
            None => CheckResult {
                passed: false,
                reason: "p99 latency unavailable (no successful iterations)".to_string(),
            },
        },
        None => CheckResult {
            passed: true,
            reason: "No p99 threshold configured".to_string(),
        },
    };

    let error_rate_check = if error_rate <= args.max_error_rate {
        CheckResult {
            passed: true,
            reason: format!(
                "Error rate {:.4} <= threshold {:.4}",
                error_rate, args.max_error_rate
            ),
        }
    } else {
        CheckResult {
            passed: false,
            reason: format!(
                "Error rate {:.4} > threshold {:.4}",
                error_rate, args.max_error_rate
            ),
        }
    };

    let db_correctness = if db_metrics.verified_within_timeout {
        CheckResult {
            passed: true,
            reason: format!(
                "DB parity verified: inference_rows={} model_inference_rows={} expected=[{}..={}]",
                db_metrics.inference_rows,
                db_metrics.model_inference_rows,
                db_metrics.min_expected_inference_rows,
                db_metrics.max_expected_inference_rows,
            ),
        }
    } else {
        CheckResult {
            passed: false,
            reason: format!(
                "DB parity failed: inference_rows={} model_inference_rows={} expected=[{}..={}]",
                db_metrics.inference_rows,
                db_metrics.model_inference_rows,
                db_metrics.min_expected_inference_rows,
                db_metrics.max_expected_inference_rows,
            ),
        }
    };

    let summary = BenchmarkSummary {
        run_id,
        function_name: args.function_name,
        target_qps,
        achieved_qps,
        min_required_qps,
        p99_latency_ms,
        error_rate,
        steady_state,
        p99_latency,
        error_rate_check,
        db_correctness,
        db_metrics,
        rlt_iters_total: rlt_report.summary.iters.total,
        rlt_success_total: rlt_report.summary.items.total,
        total_requests_sent: run_stats
            .total_requests
            .load(std::sync::atomic::Ordering::Relaxed),
    };

    print_summary(&summary);

    if let Some(path) = args.benchmark_report_json {
        std::fs::write(&path, serde_json::to_vec_pretty(&summary)?).with_context(|| {
            format!(
                "Failed to write benchmark summary JSON to {}",
                path.display()
            )
        })?;
        println!("Summary JSON written to {}", path.display());
    }

    let all_checks_passed = summary.steady_state.passed
        && summary.p99_latency.passed
        && summary.error_rate_check.passed
        && summary.db_correctness.passed;

    if all_checks_passed {
        Ok(())
    } else {
        bail!("Load test failed one or more pass/fail gates")
    }
}

fn validate_args(args: &Args) -> Result<()> {
    if !(0.0..=1.0).contains(&args.min_achieved_qps_ratio) {
        bail!("--min-achieved-qps-ratio must be between 0.0 and 1.0")
    }
    if !(0.0..=1.0).contains(&args.max_error_rate) {
        bail!("--max-error-rate must be between 0.0 and 1.0")
    }
    if args.prompt_chars == 0 {
        bail!("--prompt-chars must be greater than 0")
    }
    Ok(())
}

fn preflight_flags() -> Result<()> {
    let enable_postgres_write = std::env::var("TENSORZERO_INTERNAL_FLAG_ENABLE_POSTGRES_WRITE")
        .unwrap_or_else(|_| "".to_string());
    if enable_postgres_write != "1" {
        bail!(
            "TENSORZERO_INTERNAL_FLAG_ENABLE_POSTGRES_WRITE must be set to `1` for this benchmark (current value: `{enable_postgres_write}`)"
        )
    }
    Ok(())
}

fn resolve_target_qps(args: &Args) -> Result<f64> {
    let rate_qps = args
        .bench_opts
        .rate
        .map(|rate| f64::from(rate.get()))
        .ok_or_else(|| {
            anyhow!("The rlt -r/--rate option is required for steady-state QPS testing.")
        })?;

    if let Some(target_qps) = args.target_qps {
        if target_qps > 0.0 {
            return Ok(target_qps);
        }
        bail!("--target-qps must be > 0")
    }

    Ok(rate_qps)
}

fn resolve_postgres_url(postgres_url_arg: Option<String>) -> Result<String> {
    match postgres_url_arg {
        Some(url) => Ok(url),
        None => std::env::var("TENSORZERO_POSTGRES_URL").map_err(|_| {
            anyhow!("Missing Postgres URL. Set --postgres-url or TENSORZERO_POSTGRES_URL.")
        }),
    }
}

/// Verifies that the DB has the expected number of rows.
///
/// When the benchmark ends via duration-based termination, some requests may be in-flight:
/// the client sent them but didn't receive a response before cancellation. The gateway may
/// still process these requests and write them to the DB. So the expected inference row count
/// is a range: `[min_expected, max_expected]` where min is `successful_requests` (client got
/// a response) and max is `total_requests` (all requests sent, including cancelled ones).
async fn verify_db_parity(
    postgres_url: &str,
    run_id: &str,
    min_expected_inference_rows: u64,
    max_expected_inference_rows: u64,
    verify_timeout: Duration,
) -> Result<DbMetrics> {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(5)
        .connect(postgres_url)
        .await
        .context("Failed to connect to Postgres for verification")?;

    let mut inference_rows = query_inference_rows(&pool, run_id).await?;
    let mut model_inference_rows = query_model_inference_rows(&pool, run_id).await?;
    let deadline = tokio::time::Instant::now() + verify_timeout;

    loop {
        let min_ok = i64::try_from(min_expected_inference_rows)
            .ok()
            .map(|min| inference_rows >= min)
            .unwrap_or(false);
        let max_ok = i64::try_from(max_expected_inference_rows)
            .ok()
            .map(|max| inference_rows <= max)
            .unwrap_or(false);
        let model_rows_match = model_inference_rows >= inference_rows;

        if min_ok && max_ok && model_rows_match {
            return Ok(DbMetrics {
                min_expected_inference_rows,
                max_expected_inference_rows,
                inference_rows,
                model_inference_rows,
                verified_within_timeout: true,
            });
        }

        if tokio::time::Instant::now() >= deadline {
            break;
        }

        tokio::time::sleep(Duration::from_millis(250)).await;
        inference_rows = query_inference_rows(&pool, run_id).await?;
        model_inference_rows = query_model_inference_rows(&pool, run_id).await?;
    }

    Ok(DbMetrics {
        min_expected_inference_rows,
        max_expected_inference_rows,
        inference_rows,
        model_inference_rows,
        verified_within_timeout: false,
    })
}

async fn query_inference_rows(pool: &PgPool, run_id: &str) -> Result<i64> {
    let count = sqlx::query_scalar::<_, i64>(
        r#"
        WITH run_inference_ids AS (
            SELECT id
            FROM tensorzero.chat_inferences
            WHERE tags->>'load_test_run_id' = $1
            UNION ALL
            SELECT id
            FROM tensorzero.json_inferences
            WHERE tags->>'load_test_run_id' = $1
        )
        SELECT COUNT(*)::BIGINT
        FROM run_inference_ids
        "#,
    )
    .bind(run_id)
    .fetch_one(pool)
    .await
    .context("Failed to query inference row count")?;

    Ok(count)
}

async fn query_model_inference_rows(pool: &PgPool, run_id: &str) -> Result<i64> {
    let count = sqlx::query_scalar::<_, i64>(
        r#"
        WITH run_inference_ids AS (
            SELECT id
            FROM tensorzero.chat_inferences
            WHERE tags->>'load_test_run_id' = $1
            UNION ALL
            SELECT id
            FROM tensorzero.json_inferences
            WHERE tags->>'load_test_run_id' = $1
        )
        SELECT COUNT(*)::BIGINT
        FROM tensorzero.model_inferences m
        JOIN run_inference_ids r ON m.inference_id = r.id
        "#,
    )
    .bind(run_id)
    .fetch_one(pool)
    .await
    .context("Failed to query model_inference row count")?;

    Ok(count)
}

fn print_summary(summary: &BenchmarkSummary) {
    println!();
    println!("Postgres Inference Load Test Summary:");
    println!("  run_id: {}", summary.run_id);
    println!("  achieved_qps: {:.2}", summary.achieved_qps);
    println!("  target_qps: {:.2}", summary.target_qps);
    println!("  p99_latency_ms: {:?}", summary.p99_latency_ms);
    println!("  error_rate: {:.4}", summary.error_rate);
    println!(
        "  db_rows: inference={} model_inference={} expected=[{}..={}]",
        summary.db_metrics.inference_rows,
        summary.db_metrics.model_inference_rows,
        summary.db_metrics.min_expected_inference_rows,
        summary.db_metrics.max_expected_inference_rows,
    );
    println!();
    println!(
        "  steady_state: {} ({})",
        pass_fail(summary.steady_state.passed),
        summary.steady_state.reason
    );
    println!(
        "  p99_latency: {} ({})",
        pass_fail(summary.p99_latency.passed),
        summary.p99_latency.reason
    );
    println!(
        "  error_rate_check: {} ({})",
        pass_fail(summary.error_rate_check.passed),
        summary.error_rate_check.reason
    );
    println!(
        "  db_correctness: {} ({})",
        pass_fail(summary.db_correctness.passed),
        summary.db_correctness.reason
    );
    println!();
}

fn pass_fail(passed: bool) -> &'static str {
    if passed { "PASS" } else { "FAIL" }
}
