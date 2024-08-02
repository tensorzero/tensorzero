use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::error::Error;

/// Set up observability
pub fn setup() -> Result<(), Error> {
    setup_logs();
    setup_metrics()?;
    Ok(())
}

/// Set up logs
fn setup_logs() {
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
fn setup_metrics() -> Result<(), Error> {
    // TODO: make this configurable
    let prometheus_listener_addr = SocketAddr::from(([0, 0, 0, 0], 9090));

    PrometheusBuilder::new()
        .with_http_listener(prometheus_listener_addr)
        .install()
        .map_err(|e| Error::Observability {
            message: format!("Failed to install Prometheus exporter: {}", e),
        })?;

    Ok(())
}
