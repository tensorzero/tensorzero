#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]

mod common;

use common::start_gateway_on_random_port;
use std::time::Duration;

/// Test the gateway does not log '/health' requests when RUST_LOG and [gateway.debug] are not set
#[tokio::test]
async fn test_logging_no_rust_log_default_debug() {
    let mut child_data = start_gateway_on_random_port("", None).await;
    let health_response = child_data.call_health_endpoint().await;
    assert!(health_response.status().is_success());
    let _err: tokio::time::error::Elapsed =
        tokio::time::timeout(Duration::from_secs(1), child_data.stdout.next_line())
            .await
            .expect_err("Gateway wrote to stdout after /health endpoint in non-debug mode");
}

/// Test that the gateway logs '/health' requests when [gateway.debug] is set
#[tokio::test]
async fn test_logging_no_rust_log_debug_on() {
    let mut child_data = start_gateway_on_random_port(r"debug = true", None).await;
    let health_response = child_data.call_health_endpoint().await;
    assert!(health_response.status().is_success());

    let gateway_log_line =
        tokio::time::timeout(Duration::from_secs(1), child_data.stdout.next_line())
            .await
            .expect("Gateway didn't write to stdout after /health endpoint in debug mode")
            .expect("Error reading gateway log line")
            .expect("Gateway stdout was closed");
    println!("gateway log line: {gateway_log_line}");
    assert!(
        gateway_log_line.contains("/health"),
        "Unexpected log line: {gateway_log_line}",
    );
    assert!(
        gateway_log_line.contains("tower_http::trace"),
        "Missing tower_http::trace in log line: {gateway_log_line}",
    );
}

/// Test that the gateway does logs '/health' requests when `RUST_LOG` blocks it
#[tokio::test]
async fn test_logging_rust_log_debug_on() {
    let mut child_data = start_gateway_on_random_port(r"debug = true", Some("gateway=debug")).await;
    let health_response = child_data.call_health_endpoint().await;
    assert!(health_response.status().is_success());

    let _err: tokio::time::error::Elapsed =
        tokio::time::timeout(Duration::from_secs(1), child_data.stdout.next_line())
            .await
            .expect_err("Gateway wrote to stdout after /health endpoint in non-debug mode");
}
