use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::error::{Error, ErrorDetails};

/// Set up logs
pub fn setup_logs(debug: bool) {
    let default_level = if debug { "debug,warn" } else { "warn" };
    // Get the current log level from the environment variable `RUST_LOG`
    let log_level = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        format!("warn,gateway={default_level},tensorzero_internal={default_level}").into()
    });

    tracing_subscriber::registry()
        .with(log_level)
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_current_span(false),
        )
        .init();
}

/// Set up Prometheus metrics exporter
pub fn setup_metrics() -> Result<PrometheusHandle, Error> {
    PrometheusBuilder::new().install_recorder().map_err(|e| {
        Error::new(ErrorDetails::Observability {
            message: format!("Failed to install Prometheus exporter: {}", e),
        })
    })
}
