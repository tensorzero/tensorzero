//! Tests for include_raw_usage with relay passthrough.
//!
//! These tests validate that raw_usage is correctly passed through
//! when using the relay feature.

use crate::common::relay::start_relay_test_environment;
use futures::StreamExt;
use reqwest::Client;
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

fn assert_openai_chat_usage_details(entry: &Value) {
    let data = entry.get("data").unwrap_or(&Value::Null);
    assert!(
        data.is_object(),
        "raw_usage entry should include data for chat completions"
    );
    assert!(
        data.get("total_tokens").is_some(),
        "raw_usage should include `total_tokens` for chat completions"
    );
    assert!(
        data.get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .is_some(),
        "raw_usage should include `prompt_tokens_details.cached_tokens` for chat completions"
    );
    assert!(
        data.get("completion_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .is_some(),
        "raw_usage should include `completion_tokens_details.reasoning_tokens` for chat completions"
    );
}

/// Test that relay passthrough works for include_raw_usage (non-streaming)
#[tokio::test]
async fn test_relay_raw_usage_non_streaming() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Make a non-streaming inference request with include_raw_usage
    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "openai::gpt-5-nano",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": false,
            "include_raw_usage": true,
            "params": {
                "chat_completion": {
                    "reasoning_effort": "minimal"
                }
            }
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    // Verify we got a successful inference response
    assert!(
        body.get("inference_id").is_some(),
        "Response should have inference_id. Body: {body}"
    );

    // Check that usage exists and raw_usage is at response level (sibling to usage)
    assert!(
        body.get("usage").is_some(),
        "Response should have usage field. Body: {body}"
    );

    let raw_usage = body
        .get("raw_usage")
        .unwrap_or_else(|| panic!("Response should have raw_usage when requested. Body: {body}"));
    assert!(
        raw_usage.is_array(),
        "raw_usage should be an array, got: {raw_usage}"
    );

    let raw_usage_array = raw_usage.as_array().expect("raw_usage should be an array");
    assert!(
        !raw_usage_array.is_empty(),
        "raw_usage should have entries from downstream gateway"
    );

    // Verify structure of entries
    for entry in raw_usage_array {
        assert!(
            entry.get("model_inference_id").is_some(),
            "raw_usage entry should have model_inference_id"
        );
        // Verify provider_type is from downstream (not "relay")
        let provider_type = entry
            .get("provider_type")
            .and_then(|v| v.as_str())
            .expect("raw_usage entry should have provider_type");
        assert_eq!(
            provider_type, "openai",
            "provider_type should be from downstream provider, not relay"
        );
        assert!(
            entry.get("api_type").is_some(),
            "raw_usage entry should have api_type"
        );
        assert_openai_chat_usage_details(entry);
    }
}

/// Test relay streaming with include_raw_usage
#[tokio::test]
async fn test_relay_raw_usage_streaming() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let mut stream = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "openai::gpt-5-nano",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": true,
            "include_raw_usage": true,
            "params": {
                "chat_completion": {
                    "reasoning_effort": "minimal"
                }
            }
        }))
        .eventsource()
        .await
        .unwrap();

    let mut found_raw_usage = false;

    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk: Value = serde_json::from_str(&message.data).unwrap();

        // Check if this chunk has raw_usage (sibling to usage at chunk level)
        if let Some(raw_usage) = chunk.get("raw_usage") {
            found_raw_usage = true;
            assert!(
                raw_usage.is_array(),
                "raw_usage should be an array, got: {raw_usage}"
            );

            let raw_usage_array = raw_usage.as_array().expect("raw_usage should be an array");
            assert!(
                !raw_usage_array.is_empty(),
                "raw_usage should have entries from downstream gateway"
            );

            // Verify structure of entries (same checks as non-streaming test)
            for entry in raw_usage_array {
                assert!(
                    entry.get("model_inference_id").is_some(),
                    "raw_usage entry should have model_inference_id"
                );
                // Verify provider_type is from downstream (not "relay")
                let provider_type = entry
                    .get("provider_type")
                    .and_then(|v| v.as_str())
                    .expect("raw_usage entry should have provider_type");
                assert_eq!(
                    provider_type, "openai",
                    "provider_type should be from downstream provider, not relay"
                );
                assert!(
                    entry.get("api_type").is_some(),
                    "raw_usage entry should have api_type"
                );
                assert_openai_chat_usage_details(entry);
            }
        }
    }

    assert!(
        found_raw_usage,
        "Streaming relay response should include raw_usage in final chunk"
    );
}

/// Test that relay does NOT return raw_usage when not requested
#[tokio::test]
async fn test_relay_raw_usage_not_requested() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Make request WITHOUT include_raw_usage
    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "openai::gpt-5-nano",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": false,
            "include_raw_usage": false,
            "params": {
                "chat_completion": {
                    "reasoning_effort": "minimal"
                }
            }
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    // raw_usage should NOT be present at response level when not requested
    assert!(
        body.get("usage").is_some(),
        "Response should have usage field"
    );
    assert!(
        body.get("raw_usage").is_none(),
        "raw_usage should NOT be present when not requested. Body: {body}"
    );
}

/// Test that relay streaming does NOT return raw_usage when not requested
#[tokio::test]
async fn test_relay_raw_usage_not_requested_streaming() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let mut stream = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "openai::gpt-5-nano",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": true,
            "include_raw_usage": false,
            "params": {
                "chat_completion": {
                    "reasoning_effort": "minimal"
                }
            }
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

        // raw_usage should NOT be present at chunk level when not requested
        assert!(
            chunk.get("raw_usage").is_none(),
            "raw_usage should NOT be present in streaming chunks when not requested"
        );
    }
}

// =============================================================================
// Advanced Variant Tests (Best-of-N)
// =============================================================================

/// Test relay raw_usage passthrough with best-of-n variant (non-streaming).
/// This tests that raw_usage entries from multiple model inferences (candidates + judge)
/// are correctly passed through the relay with correct provider_type values.
#[tokio::test]
async fn test_relay_raw_usage_best_of_n_non_streaming() {
    // Downstream uses default shorthand models
    let downstream_config = "";

    // Define the function and best-of-n variant on the relay gateway.
    // Model calls will be forwarded to the downstream via relay.
    // Using gpt-5-mini with reasoning disabled for speed.
    let relay_config = r#"
[functions.best_of_n_test]
type = "chat"

[functions.best_of_n_test.variants.candidate0]
type = "chat_completion"
weight = 0
model = "openai::gpt-5-nano"
reasoning_effort = "minimal"

[functions.best_of_n_test.variants.candidate1]
type = "chat_completion"
weight = 0
model = "openai::gpt-5-nano"
reasoning_effort = "minimal"

[functions.best_of_n_test.variants.best_of_n]
type = "experimental_best_of_n_sampling"
weight = 1
candidates = ["candidate0", "candidate1"]

[functions.best_of_n_test.variants.best_of_n.evaluator]
model = "openai::gpt-5-nano"
reasoning_effort = "minimal"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "function_name": "best_of_n_test",
            "variant_name": "best_of_n",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": false,
            "include_raw_usage": true
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    assert!(
        body.get("usage").is_some(),
        "Response should have usage field. Body: {body}"
    );

    let raw_usage = body
        .get("raw_usage")
        .unwrap_or_else(|| panic!("Response should have raw_usage when requested. Body: {body}"))
        .as_array()
        .expect("raw_usage should be an array");

    // Best-of-n should have at least 3 entries: 2 candidates + 1 judge
    assert!(
        raw_usage.len() >= 3,
        "Best-of-n relay should have at least 3 raw_usage entries (2 candidates + 1 judge), got {}. Body: {body}",
        raw_usage.len()
    );

    // All entries should have provider_type = "openai" (from downstream), not "relay"
    for entry in raw_usage {
        let provider_type = entry
            .get("provider_type")
            .and_then(|v| v.as_str())
            .expect("raw_usage entry should have provider_type");
        assert_eq!(
            provider_type, "openai",
            "All raw_usage entries should have provider_type 'openai' from downstream, not 'relay'"
        );

        // Verify each entry has required fields
        assert!(
            entry.get("model_inference_id").is_some(),
            "raw_usage entry should have model_inference_id"
        );
        assert!(
            entry.get("api_type").is_some(),
            "raw_usage entry should have api_type"
        );
    }
}

/// Test relay raw_usage passthrough with best-of-n variant (streaming).
/// This tests that streaming raw_usage entries from multiple model inferences
/// are correctly passed through the relay.
#[tokio::test]
async fn test_relay_raw_usage_best_of_n_streaming() {
    // Downstream uses default shorthand models
    let downstream_config = "";

    // Define the function and best-of-n variant on the relay gateway.
    // Model calls will be forwarded to the downstream via relay.
    // Using gpt-5-mini with reasoning disabled for speed.
    let relay_config = r#"
[functions.best_of_n_test]
type = "chat"

[functions.best_of_n_test.variants.candidate0]
type = "chat_completion"
weight = 0
model = "openai::gpt-5-nano"
reasoning_effort = "minimal"

[functions.best_of_n_test.variants.candidate1]
type = "chat_completion"
weight = 0
model = "openai::gpt-5-nano"
reasoning_effort = "minimal"

[functions.best_of_n_test.variants.best_of_n]
type = "experimental_best_of_n_sampling"
weight = 1
candidates = ["candidate0", "candidate1"]

[functions.best_of_n_test.variants.best_of_n.evaluator]
model = "openai::gpt-5-nano"
reasoning_effort = "minimal"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let mut stream = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "function_name": "best_of_n_test",
            "variant_name": "best_of_n",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": true,
            "include_raw_usage": true
        }))
        .eventsource()
        .await
        .unwrap();

    let mut raw_usage_entries: Vec<Value> = Vec::new();

    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk: Value = serde_json::from_str(&message.data).unwrap();

        // Check if this chunk has raw_usage (sibling to usage at chunk level)
        if let Some(raw_usage) = chunk.get("raw_usage")
            && let Some(arr) = raw_usage.as_array()
        {
            raw_usage_entries.extend(arr.clone());
        }
    }

    // Best-of-n streaming should have at least 3 entries: 2 candidates + 1 judge
    assert!(
        raw_usage_entries.len() >= 3,
        "Best-of-n relay streaming should have at least 3 raw_usage entries (2 candidates + 1 judge), got {} (accumulated across all chunks)",
        raw_usage_entries.len()
    );

    // All entries should have provider_type = "openai" (from downstream), not "relay"
    for entry in &raw_usage_entries {
        let provider_type = entry
            .get("provider_type")
            .and_then(|v| v.as_str())
            .expect("raw_usage entry should have provider_type");
        assert_eq!(
            provider_type, "openai",
            "All streaming raw_usage entries should have provider_type 'openai' from downstream, not 'relay'"
        );

        // Verify each entry has required fields
        assert!(
            entry.get("model_inference_id").is_some(),
            "raw_usage entry should have model_inference_id"
        );
        assert!(
            entry.get("api_type").is_some(),
            "raw_usage entry should have api_type"
        );
    }
}

// =============================================================================
// Entry Structure Tests
// =============================================================================

/// Test that raw_usage entries have correct structure from downstream
#[tokio::test]
async fn test_relay_raw_usage_entry_structure() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "openai::gpt-5-nano",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": false,
            "include_raw_usage": true,
            "params": {
                "chat_completion": {
                    "reasoning_effort": "minimal"
                }
            }
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    assert!(
        body.get("usage").is_some(),
        "Response should have usage field. Body: {body}"
    );

    let raw_usage = body
        .get("raw_usage")
        .unwrap_or_else(|| panic!("Response should have raw_usage when requested. Body: {body}"))
        .as_array()
        .expect("raw_usage should be an array");

    // Should have at least one entry
    assert!(
        !raw_usage.is_empty(),
        "raw_usage should have at least one entry"
    );

    let entry = &raw_usage[0];

    // Check model_inference_id is a valid UUID string
    let model_inference_id = entry.get("model_inference_id").unwrap().as_str().unwrap();
    assert!(
        uuid::Uuid::parse_str(model_inference_id).is_ok(),
        "model_inference_id should be a valid UUID"
    );

    // Check provider_type is a non-empty string
    let provider_type = entry.get("provider_type").unwrap().as_str().unwrap();
    assert!(
        !provider_type.is_empty(),
        "provider_type should be non-empty"
    );

    // Check api_type is present
    let api_type = entry.get("api_type").unwrap().as_str().unwrap();
    assert!(
        api_type == "chat_completions" || api_type == "responses" || api_type == "embeddings",
        "api_type should be a valid value, got: {api_type}"
    );
}
