//! Tests for cost passthrough with relay.
//!
//! These tests validate that cost is correctly computed and passed through
//! when using the relay feature.

use crate::common::relay::start_relay_test_environment;
use futures::StreamExt;
use reqwest::Client;
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

#[tokio::test]
async fn test_relay_cost_non_streaming() {
    let downstream_config = r#"
[models.cost_test]
routing = ["good"]

[models.cost_test.providers.good]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 3.0 },
    { pointer = "/usage/completion_tokens", cost_per_million = 15.0 },
]
"#;
    let relay_config = r#"
[models.cost_test]
routing = ["error"]

[models.cost_test.providers.error]
type = "dummy"
model_name = "error"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "cost_test",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");
    let body: Value = serde_json::from_str(&body_text).unwrap();

    let usage = body.get("usage").unwrap().as_object().unwrap();
    let cost = usage
        .get("cost")
        .expect("Expected `cost` key in relay non-streaming usage")
        .as_f64()
        .expect("Expected `cost` to be a number");
    // dummy "good" non-streaming: prompt_tokens=10, completion_tokens=10
    assert!(
        (cost - 0.00018).abs() < 1e-10,
        "Expected cost ~0.00018 (10*$3/M + 10*$15/M), got {cost}"
    );
}

#[tokio::test]
async fn test_relay_cost_streaming() {
    let downstream_config = r#"
[models.cost_test]
routing = ["good"]

[models.cost_test.providers.good]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 3.0 },
    { pointer = "/usage/completion_tokens", cost_per_million = 15.0 },
]
"#;
    let relay_config = r#"
[models.cost_test]
routing = ["error"]

[models.cost_test.providers.error]
type = "dummy"
model_name = "error"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let mut event_source = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "cost_test",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": true
        }))
        .eventsource()
        .await
        .unwrap();

    let mut last_chunk = None;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                last_chunk = Some(message.data);
            }
        }
    }

    let last_chunk: Value = serde_json::from_str(&last_chunk.unwrap()).unwrap();
    let usage = last_chunk.get("usage").unwrap().as_object().unwrap();
    let cost = usage
        .get("cost")
        .expect("Expected `cost` key in relay streaming usage")
        .as_f64()
        .expect("Expected `cost` to be a number");
    // dummy "good" streaming: prompt_tokens=10, completion_tokens=16
    assert!(
        (cost - 0.00027).abs() < 1e-10,
        "Expected cost ~0.00027 (10*$3/M + 16*$15/M), got {cost}"
    );
}

#[tokio::test]
async fn test_relay_cost_embeddings() {
    let downstream_config = r#"
[embedding_models.cost_embedding]
routing = ["good"]

[embedding_models.cost_embedding.providers.good]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 3.0 },
]
"#;
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/openai/v1/embeddings", env.relay.addr))
        .json(&json!({
            "model": "tensorzero::embedding_model_name::cost_embedding",
            "input": "Hello, world!"
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");
    let body: Value = serde_json::from_str(&body_text).unwrap();

    let usage = body.get("usage").unwrap().as_object().unwrap();
    let cost = usage
        .get("tensorzero_cost")
        .expect("Expected `tensorzero_cost` key in relay embeddings usage")
        .as_f64()
        .expect("Expected `tensorzero_cost` to be a number");
    // dummy embedding: prompt_tokens=10 * $3/M = 0.00003
    assert!(
        (cost - 0.00003).abs() < 1e-10,
        "Expected cost ~0.00003 (10*$3/M), got {cost}"
    );
}
