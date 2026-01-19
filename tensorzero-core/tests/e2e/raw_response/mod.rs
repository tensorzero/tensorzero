//! E2E tests for `include_raw_response` parameter.
//!
//! Tests that raw provider-specific response data is correctly returned when requested.

mod cache;
mod embeddings;
mod openai_compatible;

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Helper to assert raw_response entry structure is valid
fn assert_raw_response_entry(entry: &Value) {
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
    assert!(
        entry.get("data").is_some(),
        "raw_response entry should have data field"
    );

    // Verify api_type is a valid value
    let api_type = entry.get("api_type").unwrap().as_str().unwrap();
    assert!(
        ["chat_completions", "responses", "embeddings"].contains(&api_type),
        "api_type should be 'chat_completions', 'responses', or 'embeddings', got: {api_type}"
    );

    // Verify data is a string (raw response from provider)
    assert!(
        entry.get("data").unwrap().is_string(),
        "data should be a string (raw response from provider)"
    );
}

// =============================================================================
// Chat Completions API Tests (api_type = "chat_completions")
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_response_chat_completions_non_streaming() {
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
        "include_raw_response": true
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

    // Check raw_response exists at response level
    let raw_response = response_json
        .get("raw_response")
        .expect("Response should have raw_response when include_raw_response=true");
    assert!(raw_response.is_array(), "raw_response should be an array");

    let raw_response_array = raw_response.as_array().unwrap();
    assert!(
        !raw_response_array.is_empty(),
        "raw_response should have at least one entry"
    );

    // Validate first entry structure
    let first_entry = &raw_response_array[0];
    assert_raw_response_entry(first_entry);

    // For OpenAI chat completions, api_type should be "chat_completions"
    let api_type = first_entry.get("api_type").unwrap().as_str().unwrap();
    assert_eq!(
        api_type, "chat_completions",
        "OpenAI chat completions should have api_type 'chat_completions'"
    );

    // Provider type should be "openai"
    let provider_type = first_entry.get("provider_type").unwrap().as_str().unwrap();
    assert_eq!(provider_type, "openai", "Provider type should be 'openai'");

    // The data field should be a non-empty string (raw response from provider)
    let data = first_entry.get("data").unwrap().as_str().unwrap();
    assert!(!data.is_empty(), "data should not be empty");
}

#[tokio::test]
async fn e2e_test_raw_response_chat_completions_streaming() {
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
        "include_raw_response": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .expect("Failed to create eventsource for streaming request");

    let mut found_raw_chunk = false;
    let mut content_chunks_count: usize = 0;
    let mut chunks_with_raw_chunk: usize = 0;
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

        // Check if this is a content chunk (has content field with actual deltas)
        if chunk_json.get("content").is_some() {
            content_chunks_count += 1;
        }

        // Check if this chunk has raw_chunk
        if let Some(raw_chunk) = chunk_json.get("raw_chunk") {
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
        "Streaming response should include raw_chunk in at least one chunk.\n\
        Total chunks received: {}\n\
        Content chunks: {}\n\
        Last few chunks:\n{:#?}",
        all_chunks.len(),
        content_chunks_count,
        all_chunks.iter().rev().take(3).collect::<Vec<_>>()
    );

    // Most content chunks should have raw_chunk (allow first/last to not have it)
    assert!(
        chunks_with_raw_chunk >= content_chunks_count.saturating_sub(2),
        "Most content chunks should have raw_chunk: {chunks_with_raw_chunk} of {content_chunks_count}"
    );
}

// =============================================================================
// Responses API Tests (api_type = "responses")
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_response_responses_api_non_streaming() {
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
        "include_raw_response": true
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

    // Check raw_response exists at response level
    let raw_response = response_json
        .get("raw_response")
        .expect("Response should have raw_response when include_raw_response=true");
    assert!(raw_response.is_array(), "raw_response should be an array");

    let raw_response_array = raw_response.as_array().unwrap();
    assert!(
        !raw_response_array.is_empty(),
        "raw_response should have at least one entry for responses API"
    );

    // For OpenAI Responses API, api_type should be "responses"
    let first_entry = &raw_response_array[0];
    assert_raw_response_entry(first_entry);

    let api_type = first_entry.get("api_type").unwrap().as_str().unwrap();
    assert_eq!(
        api_type, "responses",
        "OpenAI Responses API should have api_type 'responses'"
    );
}

#[tokio::test]
async fn e2e_test_raw_response_responses_api_streaming() {
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
        "include_raw_response": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .expect("Failed to create eventsource for streaming request");

    let mut found_raw_chunk = false;
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

        // Check if this chunk has raw_chunk
        if chunk_json.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }
    }

    assert!(
        found_raw_chunk,
        "Streaming response should include raw_chunk for Responses API.\n\
        Total chunks received: {}\n\
        Last few chunks:\n{:#?}",
        all_chunks.len(),
        all_chunks.iter().rev().take(3).collect::<Vec<_>>()
    );
}

// =============================================================================
// Raw response NOT requested - should not be included
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_response_not_requested_non_streaming() {
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
        // include_raw_response is NOT set (defaults to false)
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let response_json: Value = response.json().await.unwrap();

    // raw_response should NOT be present at response level when not requested
    assert!(
        response_json.get("raw_response").is_none(),
        "raw_response should not be present when include_raw_response is not set"
    );
}

#[tokio::test]
async fn e2e_test_raw_response_not_requested_streaming() {
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
        // include_raw_response is NOT set
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

        // raw_chunk should NOT be present at chunk level when not requested
        assert!(
            chunk_json.get("raw_chunk").is_none(),
            "raw_chunk should not be present in streaming chunks when not requested"
        );
        // raw_response should also NOT be present
        assert!(
            chunk_json.get("raw_response").is_none(),
            "raw_response should not be present in streaming chunks when not requested"
        );
    }
}

// =============================================================================
// Multi-Inference Variant Tests (Best-of-N)
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_response_best_of_n_non_streaming() {
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
        "include_raw_response": true
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

    // Check raw_response exists at response level
    let raw_response = response_json
        .get("raw_response")
        .expect("Response should have raw_response when include_raw_response=true");
    assert!(raw_response.is_array(), "raw_response should be an array");

    let raw_response_array = raw_response.as_array().unwrap();

    // Best-of-N should have multiple entries:
    // - 2 candidate inferences (openai_variant0 and openai_variant1)
    // - 1 evaluator/judge inference
    // Total: 3 model inferences
    assert!(
        raw_response_array.len() >= 3,
        "Best-of-N should have at least 3 raw_response entries (2 candidates + 1 judge), got {}",
        raw_response_array.len()
    );

    // Validate each entry has required fields
    for entry in raw_response_array {
        assert_raw_response_entry(entry);
    }

    // All entries should have api_type = "chat_completions" for this variant
    for entry in raw_response_array {
        let api_type = entry.get("api_type").unwrap().as_str().unwrap();
        assert_eq!(
            api_type, "chat_completions",
            "All Best-of-N inferences should have api_type 'chat_completions'"
        );
    }
}

#[tokio::test]
async fn e2e_test_raw_response_best_of_n_streaming() {
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
        "include_raw_response": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .expect("Failed to create eventsource for streaming request");

    let mut found_raw_response = false;
    let mut found_raw_chunk = false;
    let mut raw_response_count = 0;

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

        // Check for raw_chunk (current streaming inference)
        if chunk_json.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }

        // Check for raw_response (previous model inferences - candidates + evaluator)
        if let Some(raw_response) = chunk_json.get("raw_response") {
            found_raw_response = true;
            if let Some(arr) = raw_response.as_array() {
                raw_response_count += arr.len();

                // Validate each entry has required fields
                for entry in arr {
                    assert_raw_response_entry(entry);
                }
            }
        }
    }

    assert!(
        found_raw_chunk,
        "Streaming Best-of-N response should include raw_chunk for the streaming inference"
    );

    assert!(
        found_raw_response,
        "Streaming Best-of-N response should include raw_response array for previous inferences"
    );

    // Best-of-N should have 2 candidates in raw_response (the evaluator's response is streamed)
    // So raw_response should have at least 2 entries
    assert!(
        raw_response_count >= 2,
        "Best-of-N streaming should have at least 2 raw_response entries (2 candidates), got {raw_response_count}"
    );
}

// =============================================================================
// Mixture-of-N Tests
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_response_mixture_of_n_non_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "mixture_of_n",
        "variant_name": "mixture_of_n_variant",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("Please write a short sentence. {random_suffix}")
                }
            ]
        },
        "stream": false,
        "include_raw_response": true
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

    // Check raw_response exists at response level
    let raw_response = response_json
        .get("raw_response")
        .expect("Response should have raw_response when include_raw_response=true");
    assert!(raw_response.is_array(), "raw_response should be an array");

    let raw_response_array = raw_response.as_array().unwrap();

    // Mixture-of-N should have multiple entries:
    // - 2 candidate inferences
    // - 1 fuser inference
    // Total: 3 model inferences
    assert!(
        raw_response_array.len() >= 3,
        "Mixture-of-N should have at least 3 raw_response entries (2 candidates + 1 fuser), got {}",
        raw_response_array.len()
    );

    // Validate each entry has required fields
    for entry in raw_response_array {
        assert_raw_response_entry(entry);
    }
}

#[tokio::test]
async fn e2e_test_raw_response_mixture_of_n_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "mixture_of_n",
        "variant_name": "mixture_of_n_variant",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("Tell me a fun fact. {random_suffix}")
                }
            ]
        },
        "stream": true,
        "include_raw_response": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .expect("Failed to create eventsource for streaming request");

    let mut found_raw_response = false;
    let mut found_raw_chunk = false;
    let mut raw_response_count = 0;

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

        // Check for raw_chunk (current streaming inference)
        if chunk_json.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }

        // Check for raw_response (previous model inferences - candidates)
        if let Some(raw_response) = chunk_json.get("raw_response") {
            found_raw_response = true;
            if let Some(arr) = raw_response.as_array() {
                raw_response_count += arr.len();

                // Validate each entry has required fields
                for entry in arr {
                    assert_raw_response_entry(entry);
                }
            }
        }
    }

    assert!(
        found_raw_chunk,
        "Streaming Mixture-of-N response should include raw_chunk for the fuser inference"
    );

    assert!(
        found_raw_response,
        "Streaming Mixture-of-N response should include raw_response array for candidate inferences"
    );

    // Mixture-of-N should have 2 candidates in raw_response (the fuser's response is streamed)
    assert!(
        raw_response_count >= 2,
        "Mixture-of-N streaming should have at least 2 raw_response entries (2 candidates), got {raw_response_count}"
    );
}

// =============================================================================
// DICL Tests (api_type includes "embeddings")
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_response_dicl_non_streaming() {
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
        "include_raw_response": true
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

    // Check raw_response exists at response level
    let raw_response = response_json
        .get("raw_response")
        .expect("Response should have raw_response when include_raw_response=true");
    assert!(raw_response.is_array(), "raw_response should be an array");

    let raw_response_array = raw_response.as_array().unwrap();

    // DICL should have exactly 2 entries:
    // - 1 embedding call (api_type = "embeddings")
    // - 1 chat completion call (api_type = "chat_completions")
    assert_eq!(
        raw_response_array.len(),
        2,
        "DICL should have exactly 2 raw_response entries (1 embedding + 1 chat), got {}. raw_response:\n{:#?}",
        raw_response_array.len(),
        raw_response_array
    );

    // Validate each entry has required fields
    for entry in raw_response_array {
        assert_raw_response_entry(entry);
    }

    // Check that we have exactly one of each api_type
    let embedding_count = raw_response_array
        .iter()
        .filter(|e| e.get("api_type").and_then(|v| v.as_str()) == Some("embeddings"))
        .count();
    let chat_completions_count = raw_response_array
        .iter()
        .filter(|e| e.get("api_type").and_then(|v| v.as_str()) == Some("chat_completions"))
        .count();

    assert_eq!(
        embedding_count, 1,
        "DICL should have exactly 1 embedding entry, got {embedding_count}"
    );
    assert_eq!(
        chat_completions_count, 1,
        "DICL should have exactly 1 chat_completions entry, got {chat_completions_count}"
    );
}

#[tokio::test]
async fn e2e_test_raw_response_dicl_streaming() {
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
        "include_raw_response": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .expect("Failed to create eventsource for streaming request");

    let mut found_raw_response = false;
    let mut found_raw_chunk = false;
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

        // Check for raw_chunk
        if chunk_json.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }

        // Check for raw_response (previous model inferences - embedding)
        if let Some(raw_response) = chunk_json.get("raw_response") {
            found_raw_response = true;
            if let Some(arr) = raw_response.as_array() {
                for entry in arr {
                    assert_raw_response_entry(entry);
                    if let Some(api_type) = entry.get("api_type").and_then(|v| v.as_str()) {
                        api_types.push(api_type.to_string());
                    }
                }
            }
        }
    }

    assert!(
        found_raw_chunk,
        "Streaming DICL response should include raw_chunk for the chat inference"
    );

    assert!(
        found_raw_response,
        "Streaming DICL response should include raw_response for the embedding inference"
    );

    // DICL streaming should have embeddings in raw_response (chat is streamed via raw_chunk)
    assert!(
        api_types.iter().any(|t| t == "embeddings"),
        "DICL streaming should have an entry with api_type `embeddings` in raw_response, got: {api_types:?}"
    );
}

// =============================================================================
// JSON Function Tests
// =============================================================================

#[tokio::test]
async fn e2e_test_raw_response_json_function_non_streaming() {
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
        "include_raw_response": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body = response.text().await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "Response should be successful. Body: {body}"
    );
    let response_json: Value = serde_json::from_str(&body).unwrap();

    // Check raw_response exists at response level
    let raw_response = response_json
        .get("raw_response")
        .expect("Response should have raw_response when include_raw_response=true");
    assert!(raw_response.is_array(), "raw_response should be an array");

    let raw_response_array = raw_response.as_array().unwrap();
    assert!(
        !raw_response_array.is_empty(),
        "raw_response should have at least one entry for JSON function"
    );

    // Validate entry structure
    let first_entry = &raw_response_array[0];
    assert_raw_response_entry(first_entry);
}

#[tokio::test]
async fn e2e_test_raw_response_json_function_streaming() {
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
        "include_raw_response": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .expect("Failed to create eventsource for streaming request");

    let mut found_raw_chunk = false;

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

        // Check for raw_chunk
        if chunk_json.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }
    }

    assert!(
        found_raw_chunk,
        "Streaming JSON function response should include raw_chunk"
    );
}
