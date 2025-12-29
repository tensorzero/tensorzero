//! Tests for include_raw_usage with relay passthrough.
//!
//! These tests validate that raw_usage is correctly passed through
//! when using the relay feature.

use crate::common::relay::start_relay_test_environment;
use futures::StreamExt;
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

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

    // Verify we got a successful inference response
    assert!(
        body.get("inference_id").is_some(),
        "Response should have inference_id. Body: {body}"
    );

    // Check that usage exists and contains raw_usage (nested structure)
    let usage = body
        .get("usage")
        .unwrap_or_else(|| panic!("Response should have usage field. Body: {body}"));

    let raw_usage = usage
        .get("raw_usage")
        .unwrap_or_else(|| panic!("usage should have raw_usage when requested. Body: {body}"));
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
            provider_type, "dummy",
            "provider_type should be from downstream provider, not relay"
        );
        assert!(
            entry.get("api_type").is_some(),
            "raw_usage entry should have api_type"
        );
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
            "stream": true,
            "include_raw_usage": true
        }))
        .eventsource()
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

        // Check if this chunk has usage with raw_usage nested inside
        if let Some(usage) = chunk.get("usage")
            && let Some(raw_usage) = usage.get("raw_usage")
        {
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
                    provider_type, "dummy",
                    "provider_type should be from downstream provider, not relay"
                );
                assert!(
                    entry.get("api_type").is_some(),
                    "raw_usage entry should have api_type"
                );
            }
        }
    }

    assert!(
        found_raw_usage,
        "Streaming relay response should include raw_usage nested in usage in final chunk"
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
            "stream": false,
            "include_raw_usage": false
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");

    let body: Value = serde_json::from_str(&body_text).expect("Response should be valid JSON");

    // raw_usage should NOT be present in usage when not requested
    let usage = body.get("usage").expect("Response should have usage field");
    assert!(
        usage.get("raw_usage").is_none(),
        "raw_usage should NOT be present in usage when not requested. Body: {body}"
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
            "stream": true,
            "include_raw_usage": false
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

        // raw_usage should NOT be present in any chunk's usage when not requested
        if let Some(usage) = chunk.get("usage") {
            assert!(
                usage.get("raw_usage").is_none(),
                "raw_usage should NOT be present in streaming chunks when not requested"
            );
        }
    }
}

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

    let usage = body
        .get("usage")
        .unwrap_or_else(|| panic!("Response should have usage field. Body: {body}"));

    let raw_usage = usage
        .get("raw_usage")
        .unwrap_or_else(|| panic!("usage should have raw_usage when requested. Body: {body}"))
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
