//! Utilities for handling OS shutdown signals
//!
//! This crate provides a unified way to listen for shutdown signals (Ctrl+C, SIGTERM, SIGHUP)
//! across different platforms.

use tokio::signal;

pub async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = signal::ctrl_c()
            .await
            .inspect_err(|e| tracing::error!("Failed to install Ctrl+C handler: {e}"));
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) = signal::unix::signal(signal::unix::SignalKind::terminate())
            .inspect_err(|e| tracing::error!("Failed to install SIGTERM handler: {e}"))
        {
            sig.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    #[cfg(unix)]
    let hangup = async {
        if let Ok(mut sig) = signal::unix::signal(signal::unix::SignalKind::hangup())
            .inspect_err(|e| tracing::error!("Failed to install SIGHUP handler: {e}"))
        {
            sig.recv().await;
        }
    };

    #[cfg(not(unix))]
    let hangup = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            tracing::info!("Received Ctrl+C signal");
        }
        () = terminate => {
            tracing::info!("Received SIGTERM signal");
        }
        () = hangup => {
            tracing::info!("Received SIGHUP signal");
        }
    };
}
