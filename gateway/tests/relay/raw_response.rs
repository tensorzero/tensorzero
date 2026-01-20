//! Tests for include_raw_response with relay passthrough.
//!
//! These tests validate that raw_response is correctly passed through
//! when using the relay feature.

use crate::common::relay::start_relay_test_environment;
use futures::StreamExt;
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

fn assert_raw_response_entry_structure(entry: &Value) {
    assert!(
        entry.get("model_inference_id").is_some(),
        "raw_response entry should have model_inference_id"
    );
    assert!(
        entry.get("provider_type").is_some(),
        "raw_response entry should have provider_type"
    );
    assert!(
        entry.get("api_type").is_some(),
        "raw_response entry should have api_type"
    );

    // Verify api_type is valid
    let api_type = entry
        .get("api_type")
        .and_then(|v| v.as_str())
        .unwrap_or("missing");
    assert!(
        ["chat_completions", "responses", "embeddings"].contains(&api_type),
        "api_type should be 'chat_completions', 'responses', or 'embeddings', got: {api_type}"
    );

    // Verify data is a string (raw response)
    let data = entry.get("data").unwrap_or(&Value::Null);
    assert!(
        data.is_string(),
        "raw_response entry data should be a string, got: {data:?}"
    );
}

/// Test that relay passthrough works for include_raw_response (non-streaming)
#[tokio::test]
async fn test_relay_raw_response_non_streaming() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Make a non-streaming inference request with include_raw_response
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
            "include_raw_response": true,
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

    let raw_response = body.get("raw_response").unwrap_or_else(|| {
        panic!("Response should have raw_response when requested. Body: {body}")
    });
    assert!(
        raw_response.is_array(),
        "raw_response should be an array, got: {raw_response}"
    );

    let raw_response_array = raw_response
        .as_array()
        .expect("raw_response should be an array");
    assert!(
        !raw_response_array.is_empty(),
        "raw_response should have entries from downstream gateway"
    );

    // Verify structure of entries
    for entry in raw_response_array {
        assert_raw_response_entry_structure(entry);

        // Verify provider_type is from downstream (not "relay")
        let provider_type = entry
            .get("provider_type")
            .and_then(|v| v.as_str())
            .expect("raw_response entry should have provider_type");
        assert_eq!(
            provider_type, "openai",
            "provider_type should be from downstream provider, not relay"
        );
    }
}

/// Test relay streaming with include_raw_response
#[tokio::test]
async fn test_relay_raw_response_streaming() {
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
            "include_raw_response": true,
            "params": {
                "chat_completion": {
                    "reasoning_effort": "minimal"
                }
            }
        }))
        .eventsource()
        .unwrap();

    let mut found_raw_chunk = false;
    let mut content_chunks_count: usize = 0;
    let mut chunks_with_raw_chunk: usize = 0;

    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk: Value = serde_json::from_str(&message.data).unwrap();

        // Count content chunks (chunks with content delta)
        if chunk.get("content").is_some() {
            content_chunks_count += 1;
        }

        // Check for raw_chunk field in streaming
        if let Some(raw_chunk) = chunk.get("raw_chunk") {
            found_raw_chunk = true;
            chunks_with_raw_chunk += 1;
            assert!(
                raw_chunk.is_string(),
                "raw_chunk should be a string, got: {raw_chunk:?}"
            );
        }
    }

    assert!(
        found_raw_chunk,
        "Streaming relay response should include raw_chunk in at least one chunk"
    );

    // Most content chunks should have raw_chunk (allow first/last to not have it)
    assert!(
        chunks_with_raw_chunk >= content_chunks_count.saturating_sub(2),
        "Most content chunks should have raw_chunk: {chunks_with_raw_chunk} of {content_chunks_count}"
    );
}

/// Test that relay does NOT return raw_response when not requested
#[tokio::test]
async fn test_relay_raw_response_not_requested() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Make request WITHOUT include_raw_response
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
            "include_raw_response": false,
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

    // raw_response should NOT be present when not requested
    assert!(
        body.get("raw_response").is_none(),
        "raw_response should NOT be present when not requested. Body: {body}"
    );
}

/// Test that relay streaming does NOT return raw_response or raw_chunk when not requested
#[tokio::test]
async fn test_relay_raw_response_not_requested_streaming() {
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
            "include_raw_response": false,
            "params": {
                "chat_completion": {
                    "reasoning_effort": "minimal"
                }
            }
        }))
        .eventsource()
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

        // raw_response and raw_chunk should NOT be present when not requested
        assert!(
            chunk.get("raw_response").is_none(),
            "raw_response should NOT be present in streaming chunks when not requested"
        );
        assert!(
            chunk.get("raw_chunk").is_none(),
            "raw_chunk should NOT be present in streaming chunks when not requested"
        );
    }
}

// =============================================================================
// Advanced Variant Tests (Best-of-N)
// =============================================================================

/// Test relay raw_response passthrough with best-of-n variant (non-streaming).
/// This tests that raw_response entries from multiple model inferences (candidates + judge)
/// are correctly passed through the relay with correct provider_type values.
#[tokio::test]
async fn test_relay_raw_response_best_of_n_non_streaming() {
    // Downstream uses default shorthand models
    let downstream_config = "";

    // Define the function and best-of-n variant on the relay gateway.
    // Model calls will be forwarded to the downstream via relay.
    // Using gpt-5-nano with reasoning disabled for speed.
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
            "include_raw_response": true
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    let raw_response = body
        .get("raw_response")
        .unwrap_or_else(|| panic!("Response should have raw_response when requested. Body: {body}"))
        .as_array()
        .expect("raw_response should be an array");

    // Best-of-n should have at least 3 entries: 2 candidates + 1 judge
    assert!(
        raw_response.len() >= 3,
        "Best-of-n relay should have at least 3 raw_response entries (2 candidates + 1 judge), got {}. Body: {body}",
        raw_response.len()
    );

    // All entries should have provider_type = "openai" (from downstream), not "relay"
    for entry in raw_response {
        assert_raw_response_entry_structure(entry);

        let provider_type = entry
            .get("provider_type")
            .and_then(|v| v.as_str())
            .expect("raw_response entry should have provider_type");
        assert_eq!(
            provider_type, "openai",
            "All raw_response entries should have provider_type 'openai' from downstream, not 'relay'"
        );
    }
}

/// Test relay raw_response passthrough with best-of-n variant (streaming).
/// This tests that streaming raw_response entries from multiple model inferences
/// are correctly passed through the relay.
#[tokio::test]
async fn test_relay_raw_response_best_of_n_streaming() {
    // Downstream uses default shorthand models
    let downstream_config = "";

    // Define the function and best-of-n variant on the relay gateway.
    // Model calls will be forwarded to the downstream via relay.
    // Using gpt-5-nano with reasoning disabled for speed.
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
            "include_raw_response": true
        }))
        .eventsource()
        .unwrap();

    let mut raw_response_entries: Vec<Value> = Vec::new();
    let mut found_raw_chunk = false;

    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk: Value = serde_json::from_str(&message.data).unwrap();

        // Check if this chunk has raw_response (previous inferences for best-of-n)
        if let Some(raw_response) = chunk.get("raw_response")
            && let Some(arr) = raw_response.as_array()
        {
            raw_response_entries.extend(arr.clone());
        }

        // Check for raw_chunk field in streaming
        if chunk.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }
    }

    // Best-of-n streaming should have at least the 2 candidates in raw_response
    assert!(
        raw_response_entries.len() >= 2,
        "Best-of-n relay streaming should have at least 2 raw_response entries (2 candidates), got {} (accumulated across all chunks)",
        raw_response_entries.len()
    );

    // All entries should have provider_type = "openai" (from downstream), not "relay"
    for entry in &raw_response_entries {
        assert_raw_response_entry_structure(entry);

        let provider_type = entry
            .get("provider_type")
            .and_then(|v| v.as_str())
            .expect("raw_response entry should have provider_type");
        assert_eq!(
            provider_type, "openai",
            "All streaming raw_response entries should have provider_type 'openai' from downstream, not 'relay'"
        );
    }

    // Best-of-N uses fake streaming (non-streaming candidate converted to stream)
    // so raw_chunk should NOT be present (no actual streaming data)
    assert!(
        !found_raw_chunk,
        "Best-of-N streaming should NOT have raw_chunk (fake streaming has no chunk data)"
    );
}

// =============================================================================
// Entry Structure Tests
// =============================================================================

/// Test that raw_response entries have correct structure from downstream
#[tokio::test]
async fn test_relay_raw_response_entry_structure() {
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
            "include_raw_response": true,
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

    let raw_response = body
        .get("raw_response")
        .unwrap_or_else(|| panic!("Response should have raw_response when requested. Body: {body}"))
        .as_array()
        .expect("raw_response should be an array");

    // Should have at least one entry
    assert!(
        !raw_response.is_empty(),
        "raw_response should have at least one entry"
    );

    let entry = &raw_response[0];

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

    // Check api_type is present and valid
    let api_type = entry.get("api_type").unwrap().as_str().unwrap();
    assert!(
        api_type == "chat_completions" || api_type == "responses" || api_type == "embeddings",
        "api_type should be a valid value, got: {api_type}"
    );

    // Check data is a non-empty string
    let data = entry.get("data").unwrap().as_str().unwrap();
    assert!(!data.is_empty(), "data should be a non-empty string");
}

// =============================================================================
// Embeddings Relay Tests
// =============================================================================

/// Test relay passthrough works for embeddings include_raw_response
#[tokio::test]
async fn test_relay_raw_response_embeddings() {
    // Configure downstream with dummy embedding provider
    let downstream_config = r#"
[embedding_models.test-embedding]
routing = ["dummy"]

[embedding_models.test-embedding.providers.dummy]
type = "dummy"
model_name = "test-embeddings"
"#;
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/openai/v1/embeddings", env.relay.addr))
        .json(&json!({
            "input": "Hello, world!",
            "model": "tensorzero::embedding_model_name::test-embedding",
            "tensorzero::include_raw_response": true
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    // Verify standard embedding response fields
    assert_eq!(
        body.get("object").and_then(|v| v.as_str()),
        Some("list"),
        "object should be 'list'"
    );
    assert!(
        body.get("data")
            .and_then(|v| v.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false),
        "data should have embeddings"
    );

    // Check tensorzero::raw_response exists
    let raw_response = body.get("tensorzero::raw_response").unwrap_or_else(|| {
        panic!("Response should have tensorzero::raw_response when requested. Body: {body}")
    });
    assert!(
        raw_response.is_array(),
        "tensorzero::raw_response should be an array, got: {raw_response}"
    );

    let raw_response_array = raw_response
        .as_array()
        .expect("tensorzero::raw_response should be an array");
    assert!(
        !raw_response_array.is_empty(),
        "tensorzero::raw_response should have entries from downstream gateway"
    );

    // Verify structure of entries
    for entry in raw_response_array {
        assert!(
            entry.get("model_inference_id").is_some(),
            "raw_response entry should have model_inference_id"
        );
        assert!(
            entry.get("provider_type").is_some(),
            "raw_response entry should have provider_type"
        );
        assert!(
            entry.get("api_type").is_some(),
            "raw_response entry should have api_type"
        );

        // Verify api_type is "embeddings"
        let api_type = entry
            .get("api_type")
            .and_then(|v| v.as_str())
            .expect("raw_response entry should have api_type");
        assert_eq!(
            api_type, "embeddings",
            "api_type should be 'embeddings' for embeddings endpoint"
        );

        // Verify provider_type is from downstream (not "relay")
        let provider_type = entry
            .get("provider_type")
            .and_then(|v| v.as_str())
            .expect("raw_response entry should have provider_type");
        assert_eq!(
            provider_type, "dummy",
            "provider_type should be from downstream provider, not relay"
        );

        // Verify data is a string (raw response)
        let data = entry.get("data").unwrap_or(&Value::Null);
        assert!(
            data.is_string(),
            "raw_response entry data should be a string, got: {data:?}"
        );
    }
}

/// Test that relay does NOT return raw_response for embeddings when not requested
#[tokio::test]
async fn test_relay_raw_response_embeddings_not_requested() {
    // Configure downstream with dummy embedding provider
    let downstream_config = r#"
[embedding_models.test-embedding]
routing = ["dummy"]

[embedding_models.test-embedding.providers.dummy]
type = "dummy"
model_name = "test-embeddings"
"#;
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/openai/v1/embeddings", env.relay.addr))
        .json(&json!({
            "input": "Hello, world!",
            "model": "tensorzero::embedding_model_name::test-embedding"
            // tensorzero::include_raw_response is NOT set
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    // tensorzero::raw_response should NOT be present when not requested
    assert!(
        body.get("tensorzero::raw_response").is_none(),
        "tensorzero::raw_response should NOT be present when not requested. Body: {body}"
    );

    // Standard embedding response fields should still be present
    assert_eq!(body.get("object").and_then(|v| v.as_str()), Some("list"));
}

/// Test relay embeddings with batch input
#[tokio::test]
async fn test_relay_raw_response_embeddings_batch() {
    // Configure downstream with dummy embedding provider
    let downstream_config = r#"
[embedding_models.test-embedding]
routing = ["dummy"]

[embedding_models.test-embedding.providers.dummy]
type = "dummy"
model_name = "test-embeddings"
"#;
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/openai/v1/embeddings", env.relay.addr))
        .json(&json!({
            "input": ["Hello, world!", "How are you?", "This is a test."],
            "model": "tensorzero::embedding_model_name::test-embedding",
            "tensorzero::include_raw_response": true
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    // Verify batch embeddings
    let data = body
        .get("data")
        .and_then(|v| v.as_array())
        .expect("data should be an array");
    assert_eq!(
        data.len(),
        3,
        "Should have embedding for each input in batch"
    );

    // Check tensorzero::raw_response exists
    let raw_response = body
        .get("tensorzero::raw_response")
        .expect("Response should have tensorzero::raw_response");
    let raw_response_array = raw_response.as_array().unwrap();

    // Should have exactly one entry (one API call for batch)
    assert_eq!(
        raw_response_array.len(),
        1,
        "Batch embedding should have exactly 1 raw_response entry (single API call)"
    );

    // Verify the entry has correct api_type
    let api_type = raw_response_array[0]
        .get("api_type")
        .and_then(|v| v.as_str())
        .unwrap();
    assert_eq!(api_type, "embeddings");
}
