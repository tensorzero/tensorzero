//! Tests for cost propagation with relay passthrough.
//!
//! These tests validate that `usage.cost` is correctly passed through
//! when using the relay feature. Cost is computed on the downstream gateway
//! (where the model provider's cost config is defined) and forwarded
//! unchanged through the relay.

use crate::common::relay::start_relay_test_environment;
use futures::StreamExt;
use reqwest::Client;
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

// The dummy provider's DUMMY_INFER_RESPONSE_RAW has:
//   "usage": { "prompt_tokens": 10, "completion_tokens": 10 }
//
// With cost_per_million of 1.50 for prompt and 2.00 for completion:
//   prompt cost  = 10 * 1.50 / 1_000_000 = 0.000015
//   completion   = 10 * 2.00 / 1_000_000 = 0.000020
//   total        = 0.000035

// ============================================================================
// Non-Streaming Tests
// ============================================================================

/// Test that cost is propagated through relay when downstream has cost config (non-streaming).
#[tokio::test]
async fn test_relay_cost_non_streaming() {
    // Downstream has cost config on the dummy provider
    let downstream_config = r#"
[models.cost_model]
routing = ["dummy_with_cost"]

[models.cost_model.providers.dummy_with_cost]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 1.50, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 2.00, required = true },
]
"#;

    // Relay has no cost config (cost is computed on downstream, not relay)
    let relay_config = r#"
[models.cost_model]
routing = ["relay_dummy"]

[models.cost_model.providers.relay_dummy]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "cost_model",
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

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    // Verify usage exists
    let usage = body
        .get("usage")
        .unwrap_or_else(|| panic!("Response should have usage field. Body: {body}"));

    // Verify cost is present and has the expected value
    let cost = usage
        .get("cost")
        .unwrap_or_else(|| {
            panic!("usage should have cost when downstream has cost config. Body: {body}")
        })
        .as_f64()
        .expect("cost should be a number");

    assert!(
        (cost - 0.000035).abs() < 1e-10,
        "Expected cost ~0.000035, got {cost}"
    );
}

/// Test that cost is NOT present when downstream has no cost config (non-streaming).
#[tokio::test]
async fn test_relay_no_cost_config_non_streaming() {
    // Neither downstream nor relay has cost config
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "dummy::good",
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

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    let usage = body
        .get("usage")
        .unwrap_or_else(|| panic!("Response should have usage field. Body: {body}"));

    // Cost should NOT be present (skip_serializing_if = "Option::is_none")
    assert!(
        usage.get("cost").is_none(),
        "cost should not be present when no cost config is defined. Usage: {usage}"
    );
}

// ============================================================================
// Streaming Tests
// ============================================================================

/// Test that cost is propagated through relay when downstream has cost config (streaming).
#[tokio::test]
async fn test_relay_cost_streaming() {
    // Downstream has cost config
    let downstream_config = r#"
[models.cost_model]
routing = ["dummy_with_cost"]

[models.cost_model.providers.dummy_with_cost]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 1.50, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 2.00, required = true },
]
"#;

    let relay_config = r#"
[models.cost_model]
routing = ["relay_dummy"]

[models.cost_model.providers.relay_dummy]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let mut stream = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "cost_model",
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

    let mut found_cost = false;

    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk: Value = serde_json::from_str(&message.data).unwrap();

        // Check the final chunk with usage for cost
        if let Some(usage) = chunk.get("usage") {
            if let Some(cost) = usage.get("cost") {
                found_cost = true;
                let cost = cost.as_f64().expect("cost should be a number");
                assert!(
                    (cost - 0.000035).abs() < 1e-10,
                    "Expected cost ~0.000035, got {cost}"
                );
            }
        }
    }

    assert!(
        found_cost,
        "Streaming relay response should include cost in usage of final chunk"
    );
}

/// Test that cost is NOT present in streaming when downstream has no cost config.
#[tokio::test]
async fn test_relay_no_cost_config_streaming() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let mut stream = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "dummy::good",
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

    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk: Value = serde_json::from_str(&message.data).unwrap();

        // No chunk should have cost when no cost config
        if let Some(usage) = chunk.get("usage") {
            assert!(
                usage.get("cost").is_none(),
                "cost should not be present in streaming chunks when no cost config. Usage: {usage}"
            );
        }
    }
}

// ============================================================================
// Function / Variant Tests
// ============================================================================

/// Test that cost is propagated through relay when using a function with variant (non-streaming).
#[tokio::test]
async fn test_relay_cost_with_function_non_streaming() {
    // Downstream has cost config on the model
    let downstream_config = r#"
[models.cost_model]
routing = ["dummy_with_cost"]

[models.cost_model.providers.dummy_with_cost]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 1.50, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 2.00, required = true },
]
"#;

    // Relay defines a function/variant that uses cost_model
    let relay_config = r#"
[functions.cost_function]
type = "chat"

[functions.cost_function.variants.cost_variant]
type = "chat_completion"
weight = 1
model = "cost_model"

[models.cost_model]
routing = ["relay_dummy"]

[models.cost_model.providers.relay_dummy]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "function_name": "cost_function",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello via function"
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

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    let usage = body
        .get("usage")
        .unwrap_or_else(|| panic!("Response should have usage field. Body: {body}"));

    let cost = usage
        .get("cost")
        .unwrap_or_else(|| {
            panic!("usage should have cost when downstream has cost config. Body: {body}")
        })
        .as_f64()
        .expect("cost should be a number");

    assert!(
        (cost - 0.000035).abs() < 1e-10,
        "Expected cost ~0.000035, got {cost}"
    );
}

/// Test that cost is propagated through relay when using a function with variant (streaming).
#[tokio::test]
async fn test_relay_cost_with_function_streaming() {
    let downstream_config = r#"
[models.cost_model]
routing = ["dummy_with_cost"]

[models.cost_model.providers.dummy_with_cost]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 1.50, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 2.00, required = true },
]
"#;

    let relay_config = r#"
[functions.cost_function]
type = "chat"

[functions.cost_function.variants.cost_variant]
type = "chat_completion"
weight = 1
model = "cost_model"

[models.cost_model]
routing = ["relay_dummy"]

[models.cost_model.providers.relay_dummy]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let mut stream = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "function_name": "cost_function",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello via function"
                    }
                ]
            },
            "stream": true
        }))
        .eventsource()
        .await
        .unwrap();

    let mut found_cost = false;

    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk: Value = serde_json::from_str(&message.data).unwrap();

        if let Some(usage) = chunk.get("usage") {
            if let Some(cost) = usage.get("cost") {
                found_cost = true;
                let cost = cost.as_f64().expect("cost should be a number");
                assert!(
                    (cost - 0.000035).abs() < 1e-10,
                    "Expected cost ~0.000035, got {cost}"
                );
            }
        }
    }

    assert!(
        found_cost,
        "Streaming relay response with function should include cost in usage"
    );
}

// ============================================================================
// Cost Value Verification Tests
// ============================================================================

/// Test that cost values are correct with different cost_per_million rates.
#[tokio::test]
async fn test_relay_cost_different_rates() {
    // Use different rates: 3.00 for prompt, 6.00 for completion
    // Expected: 10 * 3.00 / 1_000_000 + 10 * 6.00 / 1_000_000 = 0.000090
    let downstream_config = r#"
[models.expensive_model]
routing = ["dummy_expensive"]

[models.expensive_model.providers.dummy_expensive]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 3.00, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 6.00, required = true },
]
"#;

    let relay_config = r#"
[models.expensive_model]
routing = ["relay_dummy"]

[models.expensive_model.providers.relay_dummy]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "expensive_model",
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

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    let cost = body["usage"]["cost"]
        .as_f64()
        .expect("cost should be a number");

    assert!(
        (cost - 0.000090).abs() < 1e-10,
        "Expected cost ~0.000090 with higher rates, got {cost}"
    );
}

/// Test cost with usage that also includes input/output tokens.
#[tokio::test]
async fn test_relay_cost_alongside_token_counts() {
    let downstream_config = r#"
[models.cost_model]
routing = ["dummy_with_cost"]

[models.cost_model.providers.dummy_with_cost]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 1.50, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 2.00, required = true },
]
"#;

    let relay_config = r#"
[models.cost_model]
routing = ["relay_dummy"]

[models.cost_model.providers.relay_dummy]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "cost_model",
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

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    let usage = body.get("usage").expect("Response should have usage field");

    // Token counts should be present alongside cost
    assert!(
        usage.get("input_tokens").is_some(),
        "usage should have input_tokens"
    );
    assert!(
        usage.get("output_tokens").is_some(),
        "usage should have output_tokens"
    );
    assert!(
        usage.get("cost").is_some(),
        "usage should have cost when cost config is defined"
    );

    // Verify the values make sense together
    let input_tokens = usage["input_tokens"].as_u64().unwrap();
    let output_tokens = usage["output_tokens"].as_u64().unwrap();
    let cost = usage["cost"].as_f64().unwrap();

    assert_eq!(input_tokens, 10, "dummy provider returns 10 input tokens");
    assert!(output_tokens > 0, "output_tokens should be positive");
    assert!(cost > 0.0, "cost should be positive");
}

// ============================================================================
// Cost Config Location Tests
// ============================================================================

/// Test that cost is NOT present when cost config is on the relay but not on downstream.
/// The relay forwards requests to downstream, so only the downstream's cost config matters.
/// The relay's local provider config (including cost) is ignored for relayed requests.
#[tokio::test]
async fn test_relay_cost_config_on_relay_only_non_streaming() {
    // Downstream has NO cost config
    let downstream_config = r#"
[models.cost_model]
routing = ["dummy_no_cost"]

[models.cost_model.providers.dummy_no_cost]
type = "dummy"
model_name = "good"
"#;

    // Relay HAS cost config, but it doesn't matter — relay forwards to downstream
    let relay_config = r#"
[models.cost_model]
routing = ["relay_dummy_with_cost"]

[models.cost_model.providers.relay_dummy_with_cost]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 1.50, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 2.00, required = true },
]
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "cost_model",
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

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    let usage = body
        .get("usage")
        .unwrap_or_else(|| panic!("Response should have usage field. Body: {body}"));

    // Cost should NOT be present — downstream doesn't have cost config,
    // and the relay's cost config is irrelevant for relayed requests
    assert!(
        usage.get("cost").is_none(),
        "cost should not be present when only the relay (not downstream) has cost config. Usage: {usage}"
    );
}

/// Same as above but streaming — cost config on relay only should not produce cost.
#[tokio::test]
async fn test_relay_cost_config_on_relay_only_streaming() {
    let downstream_config = r#"
[models.cost_model]
routing = ["dummy_no_cost"]

[models.cost_model.providers.dummy_no_cost]
type = "dummy"
model_name = "good"
"#;

    let relay_config = r#"
[models.cost_model]
routing = ["relay_dummy_with_cost"]

[models.cost_model.providers.relay_dummy_with_cost]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 1.50, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 2.00, required = true },
]
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let mut stream = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "cost_model",
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

    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk: Value = serde_json::from_str(&message.data).unwrap();

        if let Some(usage) = chunk.get("usage") {
            assert!(
                usage.get("cost").is_none(),
                "cost should not be present in streaming when only relay has cost config. Usage: {usage}"
            );
        }
    }
}

/// Test that when both relay and downstream have cost config with different rates,
/// the downstream's cost wins — the relay doesn't recompute or override.
#[tokio::test]
async fn test_relay_cost_downstream_wins_over_relay_non_streaming() {
    // Downstream has cost config with rates 1.50 / 2.00
    // Expected: 10 * 1.50 / 1_000_000 + 10 * 2.00 / 1_000_000 = 0.000035
    let downstream_config = r#"
[models.cost_model]
routing = ["dummy_downstream"]

[models.cost_model.providers.dummy_downstream]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 1.50, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 2.00, required = true },
]
"#;

    // Relay has DIFFERENT (much higher) cost config — should be ignored
    // If relay were computing cost, we'd get: 10 * 100 / 1M + 10 * 200 / 1M = 0.003
    let relay_config = r#"
[models.cost_model]
routing = ["relay_dummy"]

[models.cost_model.providers.relay_dummy]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 100.00, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 200.00, required = true },
]
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "cost_model",
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

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    let cost = body["usage"]["cost"]
        .as_f64()
        .expect("cost should be present from downstream");

    // Should match downstream's rates (0.000035), NOT relay's rates (0.003)
    assert!(
        (cost - 0.000035).abs() < 1e-10,
        "Cost should reflect downstream's rates (0.000035), not relay's rates (0.003). Got: {cost}"
    );
}

/// Same as above but streaming — downstream's cost should win over relay's.
#[tokio::test]
async fn test_relay_cost_downstream_wins_over_relay_streaming() {
    let downstream_config = r#"
[models.cost_model]
routing = ["dummy_downstream"]

[models.cost_model.providers.dummy_downstream]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 1.50, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 2.00, required = true },
]
"#;

    let relay_config = r#"
[models.cost_model]
routing = ["relay_dummy"]

[models.cost_model.providers.relay_dummy]
type = "dummy"
model_name = "good"
cost = [
    { pointer = "/usage/prompt_tokens", cost_per_million = 100.00, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 200.00, required = true },
]
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let mut stream = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "cost_model",
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

    let mut found_cost = false;

    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk: Value = serde_json::from_str(&message.data).unwrap();

        if let Some(usage) = chunk.get("usage") {
            if let Some(cost) = usage.get("cost") {
                found_cost = true;
                let cost = cost.as_f64().expect("cost should be a number");
                assert!(
                    (cost - 0.000035).abs() < 1e-10,
                    "Streaming cost should reflect downstream's rates (0.000035), not relay's (0.003). Got: {cost}"
                );
            }
        }
    }

    assert!(
        found_cost,
        "Streaming response should include cost from downstream"
    );
}
