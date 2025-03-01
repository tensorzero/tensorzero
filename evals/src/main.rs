use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::Parser;
use tensorzero::{ClientBuilder, ClientBuilderMode};
use url::Url;

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
}

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = tracing_subscriber::FmtSubscriber::new();
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| anyhow!("Failed to initialize tracing: {}", e))?;
    let args = Args::parse();

    #[allow(unused)]
    let tensorzero_client = match args.gateway_url {
        Some(gateway_url) => {
            ClientBuilder::new(ClientBuilderMode::HTTPGateway { url: gateway_url })
        }
        None => ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(args.config_file),
            clickhouse_url: Some(
                std::env::var("TENSORZERO_CLICKHOUSE_URL")
                    .map_err(|_| anyhow!("Missing ClickHouse URL"))?,
            ),
        }),
    }
    .build()
    .await
    .map_err(|e| anyhow!("Failed to build client: {}", e))?;

    Ok(())
}
