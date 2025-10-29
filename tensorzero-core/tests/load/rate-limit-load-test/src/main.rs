use std::sync::{atomic::AtomicU64, Arc};

use anyhow::Result;
use clap::Parser;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod benchmark;

use benchmark::{create_bucket_settings, create_postgres_pool, Contention, RateLimitBenchmark};

#[derive(Parser)]
pub struct Args {
    #[command(flatten)]
    pub bench_opts: rlt::cli::BenchCli,

    /// Rate limit capacity (number of tickets in bucket)
    #[arg(long, default_value_t = 1_000_000)]
    pub capacity: i64,

    /// Refill amount (tickets added per interval)
    #[arg(long, default_value_t = 1_000)]
    pub refill_amount: i64,

    /// Number of different rate limit keys (0 is a special value that sets up maximum contention)
    /// If not set to zero this should be at least as large as requests_per_iteration
    #[arg(long, default_value_t = 0)]
    pub contention_keys: usize,

    /// Tickets consumed per request
    #[arg(long, default_value_t = 10)]
    pub tickets_per_request: u64,

    /// Connection pool size
    #[arg(long)]
    pub pool_size: Option<u32>,

    /// Number of rate limit requests per iteration (each gets a different key)
    #[arg(long, default_value_t = 1)]
    pub requests_per_iteration: usize,
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
    let client = PostgresConnectionInfo::new_with_pool(pool, None);

    // Create bucket settings
    let bucket_settings = Arc::new(create_bucket_settings(args.capacity, args.refill_amount));

    // Create benchmark
    let benchmark = RateLimitBenchmark {
        client,
        bucket_settings,
        contention: Contention::new(args.contention_keys),
        tickets_per_request: args.tickets_per_request,
        requests_per_iteration: args.requests_per_iteration,
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
