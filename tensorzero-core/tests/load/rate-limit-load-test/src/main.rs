use std::sync::{Arc, atomic::AtomicU64};

use anyhow::Result;
use clap::{Parser, ValueEnum};
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

mod benchmark;

use benchmark::{
    Contention, RateLimitBenchmark, RateLimitClient, create_bucket_settings, create_postgres_pool,
    create_valkey_client,
};

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum Backend {
    #[default]
    Postgres,
    Valkey,
}

#[derive(Parser)]
pub struct Args {
    #[command(flatten)]
    pub bench_opts: rlt::cli::BenchCli,

    /// Rate limiting backend to use
    #[arg(long, value_enum, default_value_t = Backend::Postgres)]
    pub backend: Backend,

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

    /// Connection pool size (only used for Postgres backend)
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
                .with(rlt::TuiTracingSubscriberLayer)
                .init();
        }
        rlt::cli::Collector::Silent => {
            tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::from_default_env())
                .init();
        }
    }

    // Create client based on backend
    let client = match args.backend {
        Backend::Postgres => {
            let pool_size = args.pool_size.unwrap_or(50);
            let pool = create_postgres_pool(pool_size).await?;
            RateLimitClient::Postgres(PostgresConnectionInfo::new_with_pool(pool))
        }
        Backend::Valkey => {
            let valkey_client = create_valkey_client().await?;
            RateLimitClient::Valkey(valkey_client)
        }
    };

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
    println!("  Backend: {:?}", args.backend);
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
    if matches!(args.backend, Backend::Postgres) {
        println!("  Pool size: {}", args.pool_size.unwrap_or(50));
    }
    println!("  Requests per iteration: {}", args.requests_per_iteration);
    println!();

    // Run the benchmark
    rlt::cli::run(args.bench_opts, benchmark).await
}
