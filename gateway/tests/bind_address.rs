#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]

mod common;

use common::{start_gateway_expect_failure, start_gateway_with_cli_bind_address};

/// Test that the gateway uses the --bind-address CLI flag when no config bind_address is set.
#[tokio::test]
async fn test_bind_address_cli_flag() {
    let child_data = start_gateway_with_cli_bind_address(
        None,          // no config bind_address
        "127.0.0.1:0", // CLI bind_address
        "",            // no extra config
    )
    .await;

    // Verify the gateway bound to 127.0.0.1 (not 0.0.0.0)
    assert!(
        child_data.addr.ip().is_loopback(),
        "Expected loopback address, got {}",
        child_data.addr.ip()
    );

    // Verify the log line shows the correct address
    let listening_logged = child_data
        .output
        .iter()
        .any(|line| line.contains("listening on 127.0.0.1:"));
    assert!(
        listening_logged,
        "Expected log to show 'listening on 127.0.0.1:', output: {:?}",
        child_data.output
    );

    // Verify the health endpoint responds
    let health_response = child_data.call_health_endpoint().await;
    assert!(
        health_response.status().is_success(),
        "Health endpoint failed with status {}",
        health_response.status()
    );
}

/// Test that specifying both --bind-address CLI flag and config file bind_address causes an error.
#[tokio::test]
async fn test_bind_address_cli_and_config_errors() {
    let output = start_gateway_expect_failure(
        Some("0.0.0.0:0"), // config bind_address
        "127.0.0.1:0",     // CLI bind_address
        "",                // no extra config
    )
    .await;

    // Verify the error message
    let error_logged = output.iter().any(|line| {
        line.contains("must not specify both `--bind-address` and `[gateway].bind_address`")
    });
    assert!(
        error_logged,
        "Expected error about specifying both CLI and config bind_address, output: {output:?}",
    );
}
