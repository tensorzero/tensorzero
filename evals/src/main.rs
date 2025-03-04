use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::Parser;
use tensorzero::{ClientBuilder, ClientBuilderMode};
use tensorzero_internal::clickhouse::ClickHouseConnectionInfo;
use tensorzero_internal::config_parser::Config;
use tokio::sync::Semaphore;
// use tokio::task::JoinSet;
use url::Url;

use evals::dataset::query_dataset;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to tensorzero.toml.
    #[arg(short, long, default_value = "./config/tensorzero.toml")]
    config_file: PathBuf,

    /// URL of a running TensorZero HTTP gateway server to use for requests. This runs evals using that gateway.
    #[arg(short, long)]
    gateway_url: Option<Url>,

    /// Name of the eval to run.
    #[arg(short, long)]
    name: String,

    /// Name of the variant to run.
    #[arg(short, long)]
    variant: String,

    /// Number of concurrent requests to make.
    #[arg(short, long, default_value = "1")]
    concurrency: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = tracing_subscriber::FmtSubscriber::new();
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| anyhow!("Failed to initialize tracing: {}", e))?;
    let args = Args::parse();
    let clickhouse_url = std::env::var("TENSORZERO_CLICKHOUSE_URL")
        .map_err(|_| anyhow!("Missing ClickHouse URL at TENSORZERO_CLICKHOUSE_URL"))?;

    let config = Config::load_and_verify_from_path(&args.config_file).await?;
    let function_config = config.get_function(&args.name)?;
    #[allow(unused)]
    let tensorzero_client = match args.gateway_url {
        Some(gateway_url) => {
            ClientBuilder::new(ClientBuilderMode::HTTPGateway { url: gateway_url })
        }
        None => ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(args.config_file),
            clickhouse_url: Some(clickhouse_url.clone()),
        }),
    }
    .build()
    .await
    .map_err(|e| anyhow!("Failed to build client: {}", e))?;

    #[allow(unused)]
    let semaphore = Semaphore::new(args.concurrency);
    let clickhouse_client = ClickHouseConnectionInfo::new(&clickhouse_url).await?;
    // let mut join_set = JoinSet::new();

    #[allow(unused)]
    let dataset = query_dataset(
        &clickhouse_client,
        &args.name,
        &args.variant,
        function_config,
    )
    .await?;

    Ok(())
}
