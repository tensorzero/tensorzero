use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Set up observability
pub fn setup() {
    setup_logs();
    setup_metrics();
}

/// Set up logs
fn setup_logs() {
    // Get the current log level from the environment variable `RUST_LOG`
    let log_level = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "api=debug,warn".into());

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
fn setup_metrics() {
    // TODO: make this configurable
    let prometheus_listener_addr = "0.0.0.0:9090"
        .parse::<SocketAddr>()
        .expect("Failed to parse address");

    PrometheusBuilder::new()
        .with_http_listener(prometheus_listener_addr)
        .install()
        .expect("Failed to install Prometheus exporter");
}
