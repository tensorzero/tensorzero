use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use mimalloc::MiMalloc;
use std::fmt::Display;
use std::io::ErrorKind;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;

use tensorzero_internal::clickhouse::ClickHouseConnectionInfo;
use tensorzero_internal::config_parser::Config;
use tensorzero_internal::endpoints;
use tensorzero_internal::endpoints::status::TENSORZERO_VERSION;
use tensorzero_internal::error;
use tensorzero_internal::gateway_util;
use tensorzero_internal::observability;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Path to tensorzero.toml
    #[arg(long)]
    config_file: Option<PathBuf>,

    /// Deprecated: use `--config-file` instead
    tensorzero_toml: Option<PathBuf>,
}

#[tokio::main]
async fn main() {
    // Set up logs and metrics
    observability::setup_logs(true);
    let metrics_handle = observability::setup_metrics().expect_pretty("Failed to set up metrics");
    let args = Args::parse();

    if args.tensorzero_toml.is_some() && args.config_file.is_some() {
        tracing::error!("Cannot specify both `--config-file` and a positional path argument");
        std::process::exit(1);
    }

    if args.tensorzero_toml.is_some() {
        tracing::warn!(
            "`Specifying a positional path argument is deprecated. Use `--config-file path/to/tensorzero.toml` instead."
        );
    }

    let config_path = args.config_file.or(args.tensorzero_toml);

    let config = if let Some(path) = &config_path {
        Arc::new(
            Config::load_and_verify_from_path(Path::new(&path))
                .await
                .expect_pretty("Failed to load config"),
        )
    } else {
        tracing::warn!("No config file provided, so only default functions will be available. Use `--config-file path/to/tensorzero.toml` to specify a config file.");
        Arc::new(Config::default())
    };

    // Initialize AppState
    let app_state = gateway_util::AppStateData::new(config.clone())
        .await
        .expect_pretty("Failed to initialize AppState");

    // Create a new observability_enabled_pretty string for the log message below
    let observability_enabled_pretty = match &app_state.clickhouse_connection_info {
        ClickHouseConnectionInfo::Disabled => "disabled".to_string(),
        ClickHouseConnectionInfo::Mock { healthy, .. } => {
            format!("mocked (healthy={healthy})")
        }
        ClickHouseConnectionInfo::Production { database, .. } => {
            format!("enabled (database: {database})")
        }
    };

    // Set debug mode
    error::set_debug(config.gateway.debug).expect_pretty("Failed to set debug mode");

    let router = Router::new()
        .route("/inference", post(endpoints::inference::inference_handler))
        .route(
            "/batch_inference",
            post(endpoints::batch_inference::start_batch_inference_handler),
        )
        .route(
            "/batch_inference/:batch_id",
            get(endpoints::batch_inference::poll_batch_inference_handler),
        )
        .route(
            "/batch_inference/:batch_id/inference/:inference_id",
            get(endpoints::batch_inference::poll_batch_inference_handler),
        )
        .route(
            "/openai/v1/chat/completions",
            post(endpoints::openai_compatible::inference_handler),
        )
        .route("/feedback", post(endpoints::feedback::feedback_handler))
        .route("/status", get(endpoints::status::status_handler))
        .route("/health", get(endpoints::status::health_handler))
        .route(
            "/datasets/:dataset/datapoints",
            post(endpoints::datasets::create_datapoint_handler),
        )
        .route(
            "/metrics",
            get(move || std::future::ready(metrics_handle.render())),
        )
        .fallback(endpoints::fallback::handle_404)
        .with_state(app_state);

    // Bind to the socket address specified in the config, or default to 0.0.0.0:3000
    let bind_address = config
        .gateway
        .bind_address
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 3000)));

    let listener = match tokio::net::TcpListener::bind(bind_address).await {
        Ok(listener) => listener,
        Err(e) if e.kind() == ErrorKind::AddrInUse => {
            tracing::error!(
                "Failed to bind to socket address {bind_address}: {e}. Tip: Ensure no other process is using port {} or try a different port.",
                bind_address.port()
            );
            std::process::exit(1);
        }
        Err(e) => {
            tracing::error!("Failed to bind to socket address {bind_address}: {e}");
            std::process::exit(1);
        }
    };

    let config_path_pretty = if let Some(path) = &config_path {
        format!("config file `{}`", path.to_string_lossy())
    } else {
        "no config file".to_string()
    };

    tracing::info!(
        "TensorZero Gateway version {TENSORZERO_VERSION} is listening on {bind_address} with {config_path_pretty} and observability {observability_enabled_pretty}.",
    );

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect_pretty("Failed to start server");
}

pub async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect_pretty("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect_pretty("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(unix)]
    let hangup = async {
        signal::unix::signal(signal::unix::SignalKind::hangup())
            .expect_pretty("Failed to install SIGHUP handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C signal");
        }
        _ = terminate => {
            tracing::info!("Received SIGTERM signal");
        }
        _ = hangup => {
            tokio::time::sleep(Duration::from_secs(1)).await;
            tracing::info!("Received SIGHUP signal");
        }
    };
}

/// ┌──────────────────────────────────────────────────────────────────────────┐
/// │                           MAIN.RS ESCAPE HATCH                           │
/// └──────────────────────────────────────────────────────────────────────────┘
///
/// We don't allow panic, escape, unwrap, or similar methods in the codebase,
/// except for the private `expect_pretty` method, which is to be used only in
/// main.rs during initialization. After initialization, we expect all code to
/// handle errors gracefully.
///
/// We use `expect_pretty` for better DX when handling errors in main.rs.
/// `expect_pretty` will print an error message and exit with a status code of 1.
trait ExpectPretty<T> {
    fn expect_pretty(self, msg: &str) -> T;
}

impl<T, E: Display> ExpectPretty<T> for Result<T, E> {
    fn expect_pretty(self, msg: &str) -> T {
        match self {
            Ok(value) => value,
            Err(err) => {
                tracing::error!("{msg}: {err}");
                std::process::exit(1);
            }
        }
    }
}

impl<T> ExpectPretty<T> for Option<T> {
    fn expect_pretty(self, msg: &str) -> T {
        match self {
            Some(value) => value,
            None => {
                tracing::error!("{msg}");
                std::process::exit(1);
            }
        }
    }
}
