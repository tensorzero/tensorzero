use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;

/// Set up observability
pub fn setup() {
    setup_metrics();
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
