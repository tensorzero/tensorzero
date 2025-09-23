use std::sync::{atomic::AtomicU64, Arc};

use anyhow::Result;
use clap::Parser;
use tensorzero::Client;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod benchmark;

use benchmark::{create_bucket_settings, create_postgres_pool, Contention, RateLimitBenchmark};

#[derive(Parser)]
pub struct Args {
    #[command(flatten)]
    pub bench_opts: rlt::cli::BenchCli,

    /// Use batch writes
    pub batch_writes: bool,

    /// Use async writes
    pub async_writes: bool,

    /// Flush interval for batch writes (ms)
    pub flush_interval_ms: u64,

    /// Max batch size for batch writes
    pub max_batch_size: usize,

    /// Number of distinct inferences to send feedback to
    pub num_distinct_inferences: usize,
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

    // Create connection pool
    let pool_size = args.pool_size.unwrap_or(50);
    let pool = create_postgres_pool(pool_size).await?;
    let client = PostgresConnectionInfo::new_with_pool(pool);

    // Create bucket settings
    let bucket_settings = Arc::new(create_bucket_settings(args.capacity, args.refill_amount));

    // Create benchmark
    let benchmark = RateLimitBenchmark {
        client,
        request_counter: Arc::new(AtomicU64::new(0)),
    };

    println!("Starting rate limiting load test with configuration:");
    println!("  Capacity: {} tickets", args.capacity);
    println!("  Refill: {} tickets/second", args.refill_amount);
    println!(
        "  Contention: {} keys",
        if args.contention_keys == 0 {
            1
        } else {
            args.contention_keys
        }
    );
    println!("  Tickets per request: {}", args.tickets_per_request);
    println!("  Pool size: {pool_size}");
    println!("  Requests per iteration: {}", args.requests_per_iteration);
    println!();

    // Run the benchmark
    rlt::cli::run(args.bench_opts, benchmark).await
}
