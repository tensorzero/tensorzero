#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]

mod common;

use common::start_gateway_on_random_port;
use futures::StreamExt;
use http::StatusCode;
use reqwest_eventsource::{Event, RequestBuilderExt};
use std::time::Duration;
use tokio::time::error::Elapsed;

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
        gateway_log_line.contains("tensorzero_core::observability"),
        "Missing tensorzero_core::observability in log line: {gateway_log_line}",
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

async fn test_log_early_drop_streaming(model_name: &str, expect_finish: bool) {
    let mut child_data = start_gateway_on_random_port(
        r"debug = true",
        Some("gateway=debug,tensorzero_core::observability=debug,warn"),
    )
    .await;

    let client = reqwest::Client::new();

    let mut stream = client
        .post(format!("http://{}/inference", child_data.addr))
        .json(&serde_json::json!({
            "model_name": model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, world!"
                    }
                ]
            },
            "stream": true,
        }))
        .eventsource()
        .unwrap();

    println!("Started stream");

    // Cancel the request early, and verify that the gateway logs a warning.
    let _elapsed = tokio::time::timeout(Duration::from_millis(500), async move {
        while let Some(event) = stream.next().await {
            let event = event.unwrap();
            println!("Event: {event:?}");
            if let Event::Message(event) = event {
                if event.data == "[DONE]" {
                    break;
                }
            }
        }
    })
    .await
    .unwrap_err();
    drop(client);

    let start_line = child_data
        .stdout
        .next_line()
        .await
        .unwrap()
        .expect("Didn't find a log line after cancelling the request");
    assert!(
        start_line.contains("started processing request"),
        "Log line missing start: {start_line}"
    );
    println!("Got start line");

    let next_line = child_data
        .stdout
        .next_line()
        .await
        .unwrap()
        .expect("Didn't find a log line after cancelling the request");
    println!("Got next line: {next_line}");
    assert!(
        next_line.contains("Client closed the connection before the response was sent"),
        "Unexpected log line: {next_line}"
    );
    assert!(
        next_line.contains("WARN"),
        "Log line missing WARN: {next_line}"
    );

    if expect_finish {
        // We should get a 'finished processing request' line after the 'Client closed the connection before the response was sent' line,
        // (so that users can see the request status code for dropped SSE streams)
        // when request processing got far enough to produce a status code
        let finish_line = child_data
            .stdout
            .next_line()
            .await
            .unwrap()
            .expect("Didn't find a log line after cancelling the request");
        assert!(
            finish_line.contains("finished processing request"),
            "Unexpected log line: {finish_line}"
        );
    }
}

/// Test that the gateway logs a warning when a client connection is closed early.
#[tokio::test(flavor = "multi_thread")]
async fn test_log_early_drop_streaming_dummy_slow_initial_chunk() {
    test_log_early_drop_streaming("dummy::slow", false).await;
}

/// Test that the gateway logs a warning when a client connection is closed early.
#[tokio::test]
async fn test_log_early_drop_streaming_dummy_slow_delay_second_chunk() {
    test_log_early_drop_streaming("dummy::slow_second_chunk", true).await;
}

/// Test that the gateway logs a warning when a client connection is closed early.
#[tokio::test]
async fn test_log_early_drop_non_streaming() {
    let mut child_data = start_gateway_on_random_port("", Some("gateway=debug,warn")).await;
    let response_fut = reqwest::Client::new()
        .post(format!("http://{}/inference", child_data.addr))
        .json(&serde_json::json!({
            "model_name": "dummy::slow",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, world!"
                    }
                ]
            },
        }))
        .send();

    // Cancel the request early, and verify that the gateway logs a warning,
    // even though we aren't logging the request start/stop
    let _elapsed = tokio::time::timeout(Duration::from_millis(500), response_fut)
        .await
        .unwrap_err();

    let next_line = child_data
        .stdout
        .next_line()
        .await
        .unwrap()
        .expect("Didn't find a log line after cancelling the request");
    assert!(
        next_line.contains("Client closed the connection before the response was sent"),
        "Unexpected log line: {next_line}"
    );
    assert!(
        next_line.contains(
            r#""method":"POST","uri":"/inference","version":"HTTP/1.1","name":"request""#
        ),
        "Log line missing request information: {next_line}"
    );
    assert!(
        next_line.contains("WARN"),
        "Log line missing WARN: {next_line}"
    );
}

/// Test that the gateway does not log a warning for HEAD requests..
#[tokio::test]
async fn test_no_early_drop_warning_on_head() {
    let mut child_data = start_gateway_on_random_port(
        r"debug = true",
        Some("gateway=debug,tensorzero_core::observability=debug,warn"),
    )
    .await;
    let response = reqwest::Client::new()
        .head(format!("http://{}/health", child_data.addr))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let start_line = child_data
        .stdout
        .next_line()
        .await
        .unwrap()
        .expect("Didn't find a log line after sending HEAD request");
    assert!(
        start_line.contains("started processing request"),
        "Log line missing start: {start_line}"
    );

    println!("Getting finish line");

    let next_line = child_data
        .stdout
        .next_line()
        .await
        .unwrap()
        .expect("Didn't find a log line after HEAD request finished");
    assert!(
        next_line.contains("finished processing request"),
        "Unexpected log line: {next_line}"
    );

    println!("Got finish line");

    // We should not get any more lines
    let _: Elapsed =
        tokio::time::timeout(Duration::from_millis(100), child_data.stdout.next_line())
            .await
            .unwrap_err();
}
