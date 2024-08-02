use axum::routing::{get, post};
use axum::Router;
use std::fmt::Display;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;

use gateway::config_parser;
use gateway::endpoints;
use gateway::gateway_util;
use gateway::observability;

#[tokio::main]
async fn main() {
    // Set up observability
    observability::setup().expect_pretty("Failed to set up observability");

    // Load config
    let config =
        Arc::new(config_parser::Config::load().expect_pretty("Failed to load TensorZero config"));

    let app_state = gateway_util::AppStateData::with_config(config.clone())
        .expect_pretty("Failed to initialize AppState");

    let router = Router::new()
        .route("/inference", post(endpoints::inference::inference_handler))
        .route("/feedback", post(endpoints::feedback::feedback_handler))
        .route("/status", get(endpoints::status::status_handler))
        .route("/health", get(endpoints::status::health_handler))
        .with_state(app_state);

    // Bind to the socket address specified in the config, or default to 0.0.0.0:3000
    let bind_address = config
        .gateway
        .as_ref()
        .and_then(|gateway_config| gateway_config.bind_address)
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 3000)));

    let listener = tokio::net::TcpListener::bind(bind_address)
        .await
        .expect_pretty(&format!(
            "Failed to bind to socket address `{bind_address}`"
        ));

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
