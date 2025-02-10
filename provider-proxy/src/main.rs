use clap::Parser;
use provider_proxy::{run_server, Args};
use tokio::sync::oneshot;

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let (server_started_tx, _server_started_rx) = oneshot::channel();
    run_server(args, server_started_tx).await;
}
