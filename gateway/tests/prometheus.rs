#![allow(clippy::print_stderr, clippy::print_stdout, clippy::unwrap_used)]
use std::time::{Duration, Instant};

use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use tensorzero::test_helpers::get_metrics;
use tokio::task::JoinSet;
use tokio_stream::StreamExt;

use crate::common::start_gateway_on_random_port;

mod common;

#[tokio::test]
async fn test_prometheus_metrics_overhead_inference_nonstreaming() {
    test_prometheus_metrics_inference_helper(false).await;
}

#[tokio::test]
async fn test_prometheus_metrics_overhead_inference_streaming() {
    test_prometheus_metrics_inference_helper(true).await;
}

#[tokio::test]
async fn test_prometheus_metrics_no_health_metric() {
    let child_data = start_gateway_on_random_port(r"observability.enabled = false", None).await;
    let client = Client::new();

    let response = client
        .get(format!("http://{}/health", child_data.addr))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());

    let metrics = get_metrics(&client, &format!("http://{}/metrics", child_data.addr)).await;
    for key in metrics.keys() {
        assert!(
            !key.contains("health"),
            "No prometheus metrics should be present for the /health endpoint, but found: {key}"
        );
    }
}

async fn test_prometheus_metrics_inference_helper(stream: bool) {
    let child_data = start_gateway_on_random_port(r"observability.enabled = false", None).await;
    let client = Client::new();

    let count = 1;

    let mut join_set = JoinSet::new();

    let start = Instant::now();

    for _ in 0..count {
        let client = client.clone();
        join_set.spawn(async move {
            // Run inference (standard)
            let inference_payload = serde_json::json!({
                "model_name": "dummy::slow",
                "input": {
                    "messages": [{"role": "user", "content": "Hello, world!"}]
                },
                "stream": stream,
            });

            let builder = client
                .post(format!("http://{}/inference", child_data.addr))
                .json(&inference_payload);

            if stream {
                let mut event_source = builder.eventsource().unwrap();
                while let Some(event) = event_source.next().await {
                    let event = event.unwrap();
                    if let Event::Message(event) = event
                        && event.data == "[DONE]"
                    {
                        break;
                    }
                }
            } else {
                let response = builder.send().await.unwrap();

                assert!(response.status().is_success());
            }
        });
    }
    join_set.join_all().await;

    // Make sure that the 'dummy::slow' model was actually used and caused a sleep
    let elapsed = start.elapsed();
    assert!(
        elapsed > Duration::from_secs(5),
        "Elapsed time should be greater than 5 seconds, but was {elapsed:?}"
    );

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let metrics = get_metrics(&client, &format!("http://{}/metrics", child_data.addr)).await;

    println!("Metrics: {metrics:#?}");

    assert_eq!(
        metrics[r#"tensorzero_inference_latency_overhead_seconds_count{function_name="tensorzero::default",variant_name="dummy::slow"}"#],
        count.to_string()
    );

    let pct_50 = metrics[r#"tensorzero_inference_latency_overhead_seconds{function_name="tensorzero::default",variant_name="dummy::slow",quantile="0.5"}"#]
        .parse::<f64>()
        .unwrap();
    assert!(
        pct_50 > 0.001,
        "50th percentile overhead should be greater than 1ms"
    );
    // We have observability disabled, so we expect the overhead to be low (even though this is a debug build)
    // Notably, it does *not* include the 5-second sleep in the 'dummy::slow' model
    // This test can be slow on CI, so we give a generous 200ms margin
    assert!(
        pct_50 < 0.2,
        "Unexpectedly high 50th percentile overhead: {pct_50}s"
    );
}

#[tokio::test]
async fn test_prometheus_metrics_custom_histogram_buckets() {
    // Test that custom histogram buckets are properly configured
    let child_data = start_gateway_on_random_port(
        r"
observability.enabled = false

[gateway.metrics]
inference_overhead_histogram_buckets = [0.0001, 1.0, 10]
",
        None,
    )
    .await;
    let client = Client::new();

    // Make an inference request to generate metrics
    let inference_payload = serde_json::json!({
        "model_name": "dummy::slow",
        "input": {
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    });

    let response = client
        .post(format!("http://{}/inference", child_data.addr))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let metrics = get_metrics(&client, &format!("http://{}/metrics", child_data.addr)).await;

    println!("Metrics: {metrics:#?}");

    // Verify that our custom buckets appear in the histogram
    let expected_buckets = ["0.0001", "1", "10", "+Inf"];

    for bucket in expected_buckets {
        let key = format!(
            r#"tensorzero_inference_latency_overhead_seconds_bucket{{function_name="tensorzero::default",variant_name="dummy::slow",le="{bucket}"}}"#
        );
        assert!(
            metrics.contains_key(&key),
            "Expected bucket with le=\"{bucket}\" not found in metrics. Available keys: {:#?}",
            metrics
                .keys()
                .filter(|k| k.contains("tensorzero_inference_latency_overhead_seconds_bucket"))
                .collect::<Vec<_>>()
        );

        // Parse the value to ensure it's a valid number
        let value = metrics[&key].parse::<f64>().unwrap();
        assert!(
            value >= 0.0,
            "Bucket count should be non-negative, got {value}"
        );
    }

    // Quantile metrics should not exist
    let quantile_keys = metrics
        .keys()
        .filter(|k| k.contains("quantile"))
        .collect::<Vec<_>>();
    assert!(
        quantile_keys.is_empty(),
        "Quantile metrics should not exist, but found: {quantile_keys:#?}"
    );

    // Verify that the count metric exists
    let count_key = r#"tensorzero_inference_latency_overhead_seconds_count{function_name="tensorzero::default",variant_name="dummy::slow"}"#;
    assert!(
        metrics.contains_key(count_key),
        "Expected count metric not found"
    );
    assert_eq!(metrics[count_key], "1");

    // Verify that the sum metric exists
    let sum_key = r#"tensorzero_inference_latency_overhead_seconds_sum{function_name="tensorzero::default",variant_name="dummy::slow"}"#;
    assert!(
        metrics.contains_key(sum_key),
        "Expected sum metric not found"
    );
    let sum = metrics[sum_key].parse::<f64>().unwrap();
    assert!(sum > 0.0, "Sum should be greater than 0");

    // The latency should be between 0.0001 and 1 second
    let target_bucket = metrics[r#"tensorzero_inference_latency_overhead_seconds_bucket{function_name="tensorzero::default",variant_name="dummy::slow",le="1"}"#]
    .parse::<f64>()
    .unwrap();
    assert_eq!(target_bucket, 1.0, "Target bucket should have one entry");
}

#[tokio::test]
async fn test_prometheus_metrics_multi_variant() {
    // Create a function that always errors on the first variant,
    // and then succeeds with the second variant
    let child_data = start_gateway_on_random_port(
        r#"
observability.enabled = false

[functions.multi_variant]
type = "chat"

[functions.multi_variant.variants.variant_a]
type = "chat_completion"
model = "dummy::error"
weight = 1

[functions.multi_variant.variants.variant_b]
type = "chat_completion"
model = "dummy::slow"
"#,
        None,
    )
    .await;
    let client = Client::new();

    let count = 1;

    let mut join_set = JoinSet::new();

    let start = Instant::now();

    for _ in 0..count {
        let client = client.clone();
        join_set.spawn(async move {
            // Run inference (standard)
            let inference_payload = serde_json::json!({
                "function_name": "multi_variant",
                "input": {
                    "messages": [{"role": "user", "content": "Hello, world!"}]
                },
            });

            let builder = client
                .post(format!("http://{}/inference", child_data.addr))
                .json(&inference_payload);

            let response = builder.send().await.unwrap();

            assert!(response.status().is_success());
        });
    }
    join_set.join_all().await;

    // Make sure that the 'dummy::slow' model was actually used and caused a sleep
    let elapsed = start.elapsed();
    assert!(
        elapsed > Duration::from_secs(5),
        "Elapsed time should be greater than 5 seconds, but was {elapsed:?}"
    );

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let metrics = get_metrics(&client, &format!("http://{}/metrics", child_data.addr)).await;

    println!("Metrics: {metrics:#?}");

    // The metrics should be reported without 'variant_name', since we ran multiple variants
    assert_eq!(
        metrics[r#"tensorzero_inference_latency_overhead_seconds_count{function_name="multi_variant"}"#],
        count.to_string()
    );

    let pct_50 = metrics[r#"tensorzero_inference_latency_overhead_seconds{function_name="multi_variant",quantile="0.5"}"#]
        .parse::<f64>()
        .unwrap();
    assert!(
        pct_50 > 0.001,
        "50th percentile overhead should be greater than 1ms"
    );
    // We have observability disabled, so we expect the overhead to be low (even though this is a debug build)
    // Notably, it does *not* include the 5-second sleep in the 'dummy::slow' model
    // This test can be slow on CI, so we give a generous 200ms margin
    assert!(
        pct_50 < 0.2,
        "Unexpectedly high 50th percentile overhead: {pct_50}s"
    );
}
