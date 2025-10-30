#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]

mod common;

use common::start_gateway_on_random_port;
use futures::StreamExt;
use reqwest_eventsource::{Event, RequestBuilderExt};
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

async fn test_log_early_drop_streaming(model_name: &str) {
    let mut child_data = start_gateway_on_random_port(
        r"debug = true",
        Some("gateway=debug,tower_http::trace=debug,warn"),
    )
    .await;

    let mut stream = reqwest::Client::new()
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

    println!("Getting next line");

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

    // Tower will log a somewhat misleading 'finished processing request' line when our *route handler* finishes -
    // that is, when we return the stream body object to axum.
    // The actual processing will continue on indefinitely, since we still need to pull chunks from the remote
    // server, transform them, and send them to the client.
    // We expect to see this line, but then still see a 'Client closed the connection before the response was sent' line,
    // since our detector logic correctly detects that we didn't finish sending the entire stream to the client.
    // We may want to adjust the 'finished processing request' line in the case of streaming requests.
    if model_name == "dummy::slow_second_chunk" {
        let next_line = child_data
            .stdout
            .next_line()
            .await
            .unwrap()
            .expect("Didn't find a log line after cancelling the request");
        assert!(
            next_line.contains("finished processing request"),
            "Unexpected log line: {next_line}"
        );
    }

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
        next_line.contains("WARN"),
        "Log line missing WARN: {next_line}"
    );
}

/// Test that the gateway logs a warning when a client connection is closed early.
#[tokio::test]
async fn test_log_early_drop_streaming_dummy_slow_initial_chunk() {
    test_log_early_drop_streaming("dummy::slow").await;
}

/// Test that the gateway logs a warning when a client connection is closed early.
#[tokio::test]
async fn test_log_early_drop_streaming_dummy_slow_delay_second_chunk() {
    test_log_early_drop_streaming("dummy::slow_second_chunk").await;
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
