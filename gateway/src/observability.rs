use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::config_parser::Config;
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
pub fn setup_metrics(config: &Config) -> Result<(), Error> {
    // Get the Prometheus listener address from the config, or default to 0.0.0.0:9090
    let prometheus_listener_addr = config
        .gateway
        .as_ref()
        .and_then(|gateway_config| gateway_config.prometheus_address)
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 9090)));

    PrometheusBuilder::new()
        .with_http_listener(prometheus_listener_addr)
        .install()
        .map_err(|e| Error::Observability {
            message: format!("Failed to install Prometheus exporter: {}", e),
        })?;

    Ok(())
}
