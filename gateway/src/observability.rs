use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::error::Error;

/// Set up logs
pub fn setup_logs() {
    // Get the current log level from the environment variable `RUST_LOG`
    let log_level = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "gateway=debug,warn".into());

    tracing_subscriber::registry()
        .with(log_level)
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .flatten_event(true)
                .with_current_span(false)
                .with_target(false),
        )
        .init();
}

/// Set up Prometheus metrics exporter
pub fn setup_metrics() -> Result<PrometheusHandle, Error> {
    PrometheusBuilder::new()
        .install_recorder()
        .map_err(|e| Error::Observability {
            message: format!("Failed to install Prometheus exporter: {}", e),
        })
}
