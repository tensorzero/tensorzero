#![forbid(unsafe_code)]

use std::time::Duration;

use axum::routing::{get, post};
use axum::Router;
use tokio::signal;

use api::api_util;
use api::endpoints;
use api::observability;

#[tokio::main]
async fn main() {
    // Set up observability
    observability::setup();

    let router = Router::new()
        .route("/inference", post(endpoints::inference::inference_handler))
        .route("/feedback", post(endpoints::feedback::feedback_handler))
        .route("/status", get(endpoints::status::status_handler))
        .route("/health", get(endpoints::status::health_handler))
        .with_state(api_util::AppStateData::default());

    // TODO: allow the user to configure the port
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .expect("Failed to bind to port 3000");

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("Failed to start server")
}

pub async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(unix)]
    let hangup = async {
        signal::unix::signal(signal::unix::SignalKind::hangup())
            .expect("Failed to install SIGHUP handler")
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
