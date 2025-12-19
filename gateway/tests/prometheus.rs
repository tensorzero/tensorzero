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
async fn test_prometheus_metrics_overhead_health() {
    let child_data = start_gateway_on_random_port(r"observability.enabled = false", None).await;
    let client = Client::new();

    let response = client
        .get(format!("http://{}/health", child_data.addr))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());

    let metrics = get_metrics(&client, &format!("http://{}/metrics", child_data.addr)).await;
    println!("Metrics: {metrics:#?}");
    assert_eq!(
        metrics["tensorzero_overhead_count{kind=\"GET /health\"}"],
        "1"
    );
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
        metrics["tensorzero_overhead_count{kind=\"POST /inference\",model_name=\"dummy::slow\"}"],
        count.to_string()
    );

    let pct_50 = metrics["tensorzero_overhead{kind=\"POST /inference\",model_name=\"dummy::slow\",quantile=\"0.5\"}"]
        .parse::<f64>()
        .unwrap();
    assert!(
        pct_50 > 1.0,
        "50th percentile overhead should be greater than 1ms"
    );
    // We have observability disabled, so we expect the overhead to be low (even though this is a debug build)
    // Notably, it does *not* include the 5-second sleep in the 'dummy::slow' model
    // This test can be slow on CI, so we give a generous 200ms margin
    assert!(
        pct_50 < 200.0,
        "Unexpectedly high 50th percentile overhead: {pct_50}ms"
    );
}
