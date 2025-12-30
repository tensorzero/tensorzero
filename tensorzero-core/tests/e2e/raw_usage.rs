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

fn assert_openai_chat_usage_details(entry: &Value) {
    let usage = entry
        .get("usage")
        .expect("raw_usage entry should include usage for chat completions");
    assert!(
        usage.is_object(),
        "raw_usage entry usage should be an object for chat completions"
    );
    assert!(
        usage.get("total_tokens").is_some(),
        "raw_usage should include `total_tokens` for chat completions"
    );
    assert!(
        usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .is_some(),
        "raw_usage should include `prompt_tokens_details.cached_tokens` for chat completions"
    );
    assert!(
        usage
            .get("completion_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .is_some(),
        "raw_usage should include `completion_tokens_details.reasoning_tokens` for chat completions"
    );
}

fn assert_openai_responses_usage_details(entry: &Value) {
    let usage = entry
        .get("usage")
        .expect("raw_usage entry should include usage for responses");
    assert!(
        usage.is_object(),
        "raw_usage entry usage should be an object for responses"
    );
    assert!(
        usage.get("total_tokens").is_some(),
        "raw_usage should include `total_tokens` for responses"
    );
    assert!(
        usage
            .get("input_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .is_some(),
        "raw_usage should include `input_tokens_details.cached_tokens` for responses"
    );
    assert!(
        usage
            .get("output_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .is_some(),
        "raw_usage should include `output_tokens_details.reasoning_tokens` for responses"
    );
}

fn assert_openai_embeddings_usage_details(entry: &Value) {
    let usage = entry
        .get("usage")
        .expect("raw_usage entry should include usage for embeddings");
    assert!(
        usage.is_object(),
        "raw_usage entry usage should be an object for embeddings"
    );
    assert!(
        usage.get("prompt_tokens").is_some(),
        "raw_usage should include `prompt_tokens` for embeddings"
    );
    assert!(
        usage.get("total_tokens").is_some(),
        "raw_usage should include `total_tokens` for embeddings"
    );
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
    assert_openai_chat_usage_details(first_entry);

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

    let raw_usage_array = raw_usage.as_array().expect("raw_usage should be an array");
    assert!(
        !raw_usage_array.is_empty(),
        "Streaming chat completions should include at least one raw_usage entry"
    );
    assert_openai_chat_usage_details(&raw_usage_array[0]);
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
    assert_openai_responses_usage_details(first_entry);

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
        "Streaming response should include raw_usage nested in usage for Responses API.\n\
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
    let raw_usage_array = raw_usage.as_array().expect("raw_usage should be an array");
    assert!(
        !raw_usage_array.is_empty(),
        "Streaming responses should include at least one raw_usage entry"
    );
    assert_openai_responses_usage_details(&raw_usage_array[0]);
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

// =============================================================================
// Multi-Inference Variant Tests (Best-of-N)
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_usage_best_of_n_non_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "best_of_n",
        "variant_name": "best_of_n_variant_openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("Hello, what is your name? {random_suffix}")
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

    // Check usage exists and contains raw_usage
    let usage = response_json
        .get("usage")
        .expect("Response should have usage field");

    let raw_usage = usage
        .get("raw_usage")
        .expect("usage should have raw_usage when include_raw_usage=true");
    assert!(raw_usage.is_array(), "raw_usage should be an array");

    let raw_usage_array = raw_usage.as_array().unwrap();

    // Best-of-N should have multiple entries:
    // - 2 candidate inferences (openai_variant0 and openai_variant1)
    // - 1 evaluator/judge inference
    // Total: 3 model inferences
    assert!(
        raw_usage_array.len() >= 3,
        "Best-of-N should have at least 3 raw_usage entries (2 candidates + 1 judge), got {}",
        raw_usage_array.len()
    );

    // Validate each entry has required fields
    for entry in raw_usage_array {
        assert_raw_usage_entry(entry);
        assert_openai_chat_usage_details(entry);
    }

    // All entries should have api_type = "chat_completions" for this variant
    for entry in raw_usage_array {
        let api_type = entry.get("api_type").unwrap().as_str().unwrap();
        assert_eq!(
            api_type, "chat_completions",
            "All Best-of-N inferences should have api_type 'chat_completions'"
        );
    }
}

#[tokio::test]
async fn e2e_test_raw_usage_best_of_n_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "best_of_n",
        "variant_name": "best_of_n_variant_openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("What is your favorite color? {random_suffix}")
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
    let mut raw_usage_count = 0;

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

        // Check if this chunk has usage with raw_usage nested inside
        if let Some(usage) = chunk_json.get("usage")
            && let Some(raw_usage) = usage.get("raw_usage")
        {
            found_raw_usage = true;
            if let Some(arr) = raw_usage.as_array() {
                raw_usage_count = arr.len();

                // Validate each entry has required fields
                for entry in arr {
                    assert_raw_usage_entry(entry);
                    assert_openai_chat_usage_details(entry);
                }
            }
        }
    }

    assert!(
        found_raw_usage,
        "Streaming Best-of-N response should include raw_usage in final chunk"
    );

    // Best-of-N should have multiple entries:
    // - 2 candidate inferences (openai_variant0 and openai_variant1)
    // - 1 evaluator/judge inference
    // Total: 3 model inferences
    assert!(
        raw_usage_count >= 3,
        "Best-of-N streaming should have at least 3 raw_usage entries (2 candidates + 1 judge), got {raw_usage_count}"
    );
}

// =============================================================================
// DICL Tests (with embeddings api_type)
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_usage_dicl_non_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "dicl",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("What is the capital of France? {random_suffix}")
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

    // Check usage exists and contains raw_usage
    let usage = response_json
        .get("usage")
        .expect("Response should have usage field");

    let raw_usage = usage
        .get("raw_usage")
        .expect("usage should have raw_usage when include_raw_usage=true");
    assert!(raw_usage.is_array(), "raw_usage should be an array");

    let raw_usage_array = raw_usage.as_array().unwrap();

    // DICL should have at least 2 entries:
    // - 1 embedding call (api_type = "embeddings")
    // - 1 chat completion call (api_type = "chat_completions")
    assert!(
        raw_usage_array.len() >= 2,
        "DICL should have at least 2 raw_usage entries (embedding + chat), got {}",
        raw_usage_array.len()
    );

    // Validate each entry has required fields
    for entry in raw_usage_array {
        assert_raw_usage_entry(entry);
        match entry.get("api_type").and_then(|v| v.as_str()) {
            Some("embeddings") => assert_openai_embeddings_usage_details(entry),
            Some("chat_completions") => assert_openai_chat_usage_details(entry),
            _ => {}
        }
    }

    // Check that we have both api_types
    let api_types: Vec<&str> = raw_usage_array
        .iter()
        .map(|e| e.get("api_type").unwrap().as_str().unwrap())
        .collect();

    assert!(
        api_types.contains(&"embeddings"),
        "DICL should have an entry with api_type `embeddings`, got: {api_types:?}"
    );
    assert!(
        api_types.contains(&"chat_completions"),
        "DICL should have an entry with api_type `chat_completions`, got: {api_types:?}"
    );
}

#[tokio::test]
async fn e2e_test_raw_usage_dicl_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "dicl",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("What is the capital of Germany? {random_suffix}")
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
    let mut api_types: Vec<String> = Vec::new();

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

        // Check if this chunk has usage with raw_usage nested inside
        if let Some(usage) = chunk_json.get("usage")
            && let Some(raw_usage) = usage.get("raw_usage")
        {
            found_raw_usage = true;
            if let Some(arr) = raw_usage.as_array() {
                for entry in arr {
                    assert_raw_usage_entry(entry);
                    match entry.get("api_type").and_then(|v| v.as_str()) {
                        Some("embeddings") => assert_openai_embeddings_usage_details(entry),
                        Some("chat_completions") => assert_openai_chat_usage_details(entry),
                        _ => {}
                    }
                    if let Some(api_type) = entry.get("api_type").and_then(|v| v.as_str()) {
                        api_types.push(api_type.to_string());
                    }
                }
            }
        }
    }

    assert!(
        found_raw_usage,
        "Streaming DICL response should include raw_usage in final chunk"
    );

    // DICL should have both embeddings and chat_completions api_types
    assert!(
        api_types.iter().any(|t| t == "embeddings"),
        "DICL streaming should have an entry with api_type `embeddings`, got: {api_types:?}"
    );
    assert!(
        api_types.iter().any(|t| t == "chat_completions"),
        "DICL streaming should have an entry with api_type `chat_completions`, got: {api_types:?}"
    );
}

// =============================================================================
// Streaming Constraint Error Test
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_usage_streaming_requires_include_usage() {
    let episode_id = Uuid::now_v7();

    // OpenAI-compatible API: include_raw_usage without include_usage should error
    let payload = json!({
        "model": "tensorzero::function_name::basic_test",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "stream": true,
        "stream_options": {
            "include_usage": false  // Explicitly false
        },
        "tensorzero::episode_id": episode_id,
        "tensorzero::include_raw_usage": true  // This should error without include_usage
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Should return an error status
    assert!(
        response.status().is_client_error(),
        "Request with include_raw_usage but without include_usage should fail, got status: {}",
        response.status()
    );
}

// =============================================================================
// JSON Function Tests
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_usage_json_function_non_streaming() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "JsonBot"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
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

    // Check usage exists and contains raw_usage
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
        "raw_usage should have at least one entry for JSON function"
    );

    // Validate entry structure
    let first_entry = &raw_usage_array[0];
    assert_raw_usage_entry(first_entry);
    assert_openai_chat_usage_details(first_entry);

    // JSON functions should still have api_type = "chat_completions"
    let api_type = first_entry.get("api_type").unwrap().as_str().unwrap();
    assert_eq!(
        api_type, "chat_completions",
        "JSON function should have api_type 'chat_completions'"
    );
}

#[tokio::test]
async fn e2e_test_raw_usage_json_function_streaming() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "JsonBot"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "France"}}]
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
        "Streaming JSON function response should include raw_usage in final chunk"
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
    let raw_usage_array = raw_usage.as_array().expect("raw_usage should be an array");
    assert!(
        !raw_usage_array.is_empty(),
        "Streaming JSON function should include at least one raw_usage entry"
    );
    assert_openai_chat_usage_details(&raw_usage_array[0]);
}
