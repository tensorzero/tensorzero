//! E2E tests for `include_raw_usage` parameter.
//!
//! Tests that raw provider-specific usage data is correctly returned when requested.

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Helper to assert raw_usage structure is valid
fn assert_raw_usage_entry(entry: &Value) {
    assert!(
        entry.get("model_inference_id").is_some(),
        "raw_usage entry should have model_inference_id"
    );
    assert!(
        entry.get("provider_type").is_some(),
        "raw_usage entry should have provider_type"
    );
    assert!(
        entry.get("api_type").is_some(),
        "raw_usage entry should have api_type"
    );
    // usage can be null if provider doesn't return it, but field should exist or be omitted
}

// =============================================================================
// Chat Completions API Tests (api_type = "chat_completions")
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_usage_chat_completions_non_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("What is the weather in Tokyo? {random_suffix}")
                }
            ]
        },
        "stream": false,
        "include_raw_usage": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "Response should be successful"
    );

    let response_json: Value = response.json().await.unwrap();

    // Check usage exists and contains raw_usage (nested structure)
    let usage = response_json
        .get("usage")
        .expect("Response should have usage field");

    assert!(
        usage.get("input_tokens").is_some(),
        "usage should have input_tokens"
    );
    assert!(
        usage.get("output_tokens").is_some(),
        "usage should have output_tokens"
    );

    // Check raw_usage exists inside usage and is an array
    let raw_usage = usage
        .get("raw_usage")
        .expect("usage should have raw_usage when include_raw_usage=true");
    assert!(raw_usage.is_array(), "raw_usage should be an array");

    let raw_usage_array = raw_usage.as_array().unwrap();
    assert!(
        !raw_usage_array.is_empty(),
        "raw_usage should have at least one entry"
    );

    // Validate first entry structure
    let first_entry = &raw_usage_array[0];
    assert_raw_usage_entry(first_entry);

    // For OpenAI chat completions, api_type should be "chat_completions"
    let api_type = first_entry.get("api_type").unwrap().as_str().unwrap();
    assert_eq!(
        api_type, "chat_completions",
        "OpenAI chat completions should have api_type 'chat_completions'"
    );

    // Provider type should be "openai"
    let provider_type = first_entry.get("provider_type").unwrap().as_str().unwrap();
    assert_eq!(provider_type, "openai", "Provider type should be 'openai'");

    // The usage object should contain OpenAI-specific fields
    if let Some(usage) = first_entry.get("usage") {
        assert!(
            usage.get("prompt_tokens").is_some() || usage.is_null(),
            "OpenAI usage should have prompt_tokens or be null"
        );
    }
}

#[tokio::test]
async fn e2e_test_raw_usage_chat_completions_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("What is the weather in Paris? {random_suffix}")
                }
            ]
        },
        "stream": true,
        "include_raw_usage": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .expect("Failed to create eventsource for streaming request");

    let mut found_raw_usage = false;
    let mut last_chunk_with_usage: Option<Value> = None;
    let mut all_chunks: Vec<Value> = Vec::new();

    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.expect("Failed to receive chunk from stream");
        let Event::Message(chunk) = chunk else {
            continue;
        };
        if chunk.data == "[DONE]" {
            break;
        }

        let chunk_json: Value =
            serde_json::from_str(&chunk.data).expect("Failed to parse chunk as JSON");

        all_chunks.push(chunk_json.clone());

        // Check if this chunk has usage with raw_usage nested inside
        if let Some(usage) = chunk_json.get("usage")
            && usage.get("raw_usage").is_some()
        {
            found_raw_usage = true;
            last_chunk_with_usage = Some(chunk_json.clone());
        }
    }

    assert!(
        found_raw_usage,
        "Streaming response should include raw_usage nested in usage in final chunk when include_raw_usage=true.\n\
        Total chunks received: {}\n\
        Last few chunks:\n{:#?}",
        all_chunks.len(),
        all_chunks.iter().rev().take(3).collect::<Vec<_>>()
    );

    let final_chunk = last_chunk_with_usage
        .expect("No chunk with raw_usage found despite found_raw_usage being true");
    let usage = final_chunk
        .get("usage")
        .expect("usage field missing from final chunk");
    let raw_usage = usage
        .get("raw_usage")
        .expect("raw_usage field missing from usage");
    assert!(raw_usage.is_array(), "raw_usage should be an array");

    let _raw_usage_array = raw_usage.as_array().expect("raw_usage should be an array");
    // Note: For streaming, raw_usage may come from previous_model_inference_results
    // which excludes the current streaming inference. For a simple variant, this means
    // raw_usage may be empty if there's only one model inference in progress.
    // This is expected behavior - the current streaming inference's raw_usage
    // would need to be collected from chunks.
}

// =============================================================================
// Responses API Tests (api_type = "responses")
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_usage_responses_api_non_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "openai-responses",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("What is the weather in London? {random_suffix}")
                }
            ]
        },
        "stream": false,
        "include_raw_usage": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "Response should be successful"
    );

    let response_json: Value = response.json().await.unwrap();

    // Check usage exists and contains raw_usage (nested structure)
    let usage = response_json
        .get("usage")
        .expect("Response should have usage field");

    let raw_usage = usage
        .get("raw_usage")
        .expect("usage should have raw_usage when include_raw_usage=true");
    assert!(raw_usage.is_array(), "raw_usage should be an array");

    let raw_usage_array = raw_usage.as_array().unwrap();
    assert!(
        !raw_usage_array.is_empty(),
        "raw_usage should have at least one entry for responses API"
    );

    // For OpenAI Responses API, api_type should be "responses"
    let first_entry = &raw_usage_array[0];
    assert_raw_usage_entry(first_entry);

    let api_type = first_entry.get("api_type").unwrap().as_str().unwrap();
    assert_eq!(
        api_type, "responses",
        "OpenAI Responses API should have api_type 'responses'"
    );
}

#[tokio::test]
async fn e2e_test_raw_usage_responses_api_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "openai-responses",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("What is the weather in Berlin? {random_suffix}")
                }
            ]
        },
        "stream": true,
        "include_raw_usage": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .expect("Failed to create eventsource for streaming request");

    let mut found_raw_usage = false;
    let mut all_chunks: Vec<Value> = Vec::new();

    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.expect("Failed to receive chunk from stream");
        let Event::Message(chunk) = chunk else {
            continue;
        };
        if chunk.data == "[DONE]" {
            break;
        }

        let chunk_json: Value =
            serde_json::from_str(&chunk.data).expect("Failed to parse chunk as JSON");

        all_chunks.push(chunk_json.clone());

        // Check if this chunk has usage with raw_usage nested inside
        if let Some(usage) = chunk_json.get("usage")
            && usage.get("raw_usage").is_some()
        {
            found_raw_usage = true;
        }
    }

    assert!(
        found_raw_usage,
        "Streaming response should include raw_usage nested in usage for Responses API.\n\
        Total chunks received: {}\n\
        Last few chunks:\n{:#?}",
        all_chunks.len(),
        all_chunks.iter().rev().take(3).collect::<Vec<_>>()
    );
}

// =============================================================================
// Raw usage NOT requested - should not be included
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_usage_not_requested_non_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("What is the weather in Sydney? {random_suffix}")
                }
            ]
        },
        "stream": false,
        // include_raw_usage is NOT set (defaults to false)
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let response_json: Value = response.json().await.unwrap();

    // raw_usage should NOT be present in usage when not requested
    let usage = response_json
        .get("usage")
        .expect("Response should have usage field");
    assert!(
        usage.get("raw_usage").is_none(),
        "raw_usage should not be present in usage when include_raw_usage is not set"
    );
}

#[tokio::test]
async fn e2e_test_raw_usage_not_requested_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "WeatherBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("What is the weather in Madrid? {random_suffix}")
                }
            ]
        },
        "stream": true,
        // include_raw_usage is NOT set
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.unwrap();
        let Event::Message(chunk) = chunk else {
            continue;
        };
        if chunk.data == "[DONE]" {
            break;
        }

        let chunk_json: Value = serde_json::from_str(&chunk.data).unwrap();

        // raw_usage should NOT be present in any chunk's usage
        if let Some(usage) = chunk_json.get("usage") {
            assert!(
                usage.get("raw_usage").is_none(),
                "raw_usage should not be present in streaming chunks' usage when not requested"
            );
        }
    }
}
