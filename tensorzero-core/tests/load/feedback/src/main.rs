use std::sync::{atomic::AtomicU64, Arc};

use anyhow::Result;
use benchmark::RateLimitBenchmark;
use clap::Parser;
use serde::Deserialize;
use tensorzero::test_helpers::make_embedded_gateway_with_config;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use uuid::Uuid;

mod benchmark;

#[derive(Parser)]
pub struct Args {
    #[command(flatten)]
    pub bench_opts: rlt::cli::BenchCli,

    /// Use batch writes
    #[arg(long, default_value_t = false)]
    pub batch_writes: bool,

    /// Use async writes
    #[arg(long, default_value_t = false)]
    pub async_writes: bool,

    /// Flush interval for batch writes (ms)
    #[arg(long, default_value_t = 100)]
    pub flush_interval_ms: u64,

    /// Max batch size for batch writes
    #[arg(long, default_value_t = 1000)]
    pub max_rows: usize,

    /// Number of distinct inferences to send feedback to
    #[arg(long, default_value_t = 10000)]
    pub num_distinct_inferences: usize,

    #[arg(long, default_value_t = false)]
    pub disable_feedback_target_validation: bool,
}

#[derive(Deserialize)]
struct InferenceIdInfo {
    id: Vec<Uuid>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Setup tracing based on collector type
    match args.bench_opts.collector() {
        rlt::cli::Collector::Tui => {
            tracing_subscriber::registry()
                .with(EnvFilter::from_default_env())
                .with(rlt::tui_tracing_subscriber_layer())
                .init();
        }
        rlt::cli::Collector::Silent => {
            tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::from_default_env())
                .init();
        }
    }

    if args.async_writes && args.batch_writes {
        panic!("Async writes and batch writes cannot be used together");
    }

    // Get a long list of inference IDs from clickhouse
    let clickhouse_connection = get_clickhouse().await;
    let inference_id_response: InferenceIdInfo = clickhouse_connection
        .run_query_synchronous_no_params_de(format!(
            "SELECT uint_to_uuid(id_uint) as id FROM InferenceById LIMIT {} FORMAT JSONColumns",
            args.num_distinct_inferences
        ))
        .await?;

    let write_config = if args.async_writes {
        "gateway.observability.async_writes = true".to_string()
    } else if args.batch_writes {
        format!(
            r"
[gateway.observability.batch_writes]
enabled = true
flush_interval_ms = {}
max_rows = {}",
            args.flush_interval_ms, args.max_rows
        )
    } else {
        "".to_string()
    };

    // Set up a TensorZero client with a metric and the appropriate writing settings
    let config = format!(
        r#"
gateway.unstable_disable_feedback_target_validation = {}
{write_config}
[metrics.test]
type = "float"
optimize = "max"
level = "inference"
    "#,
        args.disable_feedback_target_validation
    );

    let client = make_embedded_gateway_with_config(&config).await;

    // Create benchmark
    let benchmark = RateLimitBenchmark {
        client: Arc::new(client),
        inference_ids: Arc::new(inference_id_response.id),
        request_counter: Arc::new(AtomicU64::new(0)),
    };
    println!("Starting feedback load test with configuration:");
    println!("  Distinct inferences: {}", args.num_distinct_inferences);
    println!("  Batch writes: {}", args.batch_writes);
    println!("  Async writes: {}", args.async_writes);
    if args.batch_writes {
        println!("  Flush interval: {}ms", args.flush_interval_ms);
        println!("  Max rows: {}", args.max_rows);
    }
    println!();

    // Run the benchmark
    rlt::cli::run(args.bench_opts, benchmark).await
}
