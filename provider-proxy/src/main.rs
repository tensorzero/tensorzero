use clap::{Parser, Subcommand};
use provider_proxy::{Args, run_server};
use std::process::ExitCode;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    #[command(flatten)]
    args: Args,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Check if the health endpoint is responding (for container health checks)
    HealthCheck {
        /// Health check port
        #[arg(long, default_value = "3004")]
        port: u16,
    },
}

async fn run_health_check(port: u16) -> ExitCode {
    let url = format!("http://127.0.0.1:{port}/health");
    match reqwest::get(&url).await {
        Ok(response) if response.status().is_success() => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::HealthCheck { port }) => run_health_check(port).await,
        None => {
            let (server_started_tx, _server_started_rx) = oneshot::channel();
            run_server(cli.args, server_started_tx).await;
            ExitCode::SUCCESS
        }
    }
}
