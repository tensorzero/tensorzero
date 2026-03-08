//! E2E tests for `raw_response` interaction with caching in error/fallback scenarios.
//!
//! Part A: Cache hit/miss cycle tests — first request (miss) has populated `raw_response`;
//! second request (hit) has reduced `raw_response` because cached entries are excluded.
//!
//! Part B: Error + cache regression tests — provider failures with `cache_options` enabled
//! still return proper `raw_response` error entries (errors are never cached).

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use super::assert_error_raw_response_entry;
use crate::common::get_gateway_endpoint;
use crate::raw_response::assert_raw_response_entry;

// =============================================================================
// Part A: Cache Hit/Miss — Chat Fallback (non-streaming + streaming)
// =============================================================================

/// Cache hit/miss cycle: model fallback (error provider -> good provider), non-streaming.
///
/// First request (cache miss): 2 raw_response entries (1 error + 1 success).
/// Second request (cache hit): 1 raw_response entry (error only — success provider is cached).
#[tokio::test]
async fn test_cache_hit_miss_chat_fallback_non_streaming() {
    let unique_input = format!("cache test: chat fallback non-streaming {}", Uuid::now_v7());
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "model_fallback_with_raw_response",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": unique_input}]
        },
        "stream": false,
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60}
    });

    // --- First request (cache miss) ---
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "First request should succeed (fallback to good provider)"
    );

    let response_json: Value = response.json().await.unwrap();
    let raw_response_array = response_json["raw_response"]
        .as_array()
        .expect("raw_response should be an array");
    assert_eq!(
        raw_response_array.len(),
        2,
        "Cache miss: expected 2 entries (1 error + 1 success)"
    );
    assert_error_raw_response_entry(&raw_response_array[0]);
    assert_raw_response_entry(&raw_response_array[1]);

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // --- Second request (cache hit for good provider) ---
    let response2 = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "Second request should succeed (cache hit)"
    );

    let response2_json: Value = response2.json().await.unwrap();
    let raw_response_array2 = response2_json["raw_response"]
        .as_array()
        .expect("raw_response should be an array on cache hit");
    assert_eq!(
        raw_response_array2.len(),
        1,
        "Cache hit: expected 1 entry (error only — success provider cached)"
    );
    assert_error_raw_response_entry(&raw_response_array2[0]);
}

/// Cache hit/miss cycle: model fallback (error provider -> good provider), streaming.
///
/// First request (cache miss): streaming chunks include error raw_response + raw_chunk from good provider.
/// Second request (cache hit): streaming chunks include error raw_response but raw_chunk is suppressed
/// (consistent with non-streaming cache hits excluding the success entry).
#[tokio::test]
async fn test_cache_hit_miss_chat_fallback_streaming() {
    let unique_input = format!("cache test: chat fallback streaming {}", Uuid::now_v7());
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "model_fallback_with_raw_response",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": unique_input}]
        },
        "stream": true,
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60}
    });

    // --- First request (cache miss) ---
    {
        let mut chunks = Client::new()
            .post(get_gateway_endpoint("/inference"))
            .json(&payload)
            .eventsource()
            .await
            .expect("Failed to create eventsource for first request");

        let mut found_error_raw_response = false;
        let mut found_raw_chunk = false;

        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.expect("Failed to receive chunk");
            let Event::Message(chunk) = chunk else {
                continue;
            };
            if chunk.data == "[DONE]" {
                break;
            }

            let chunk_json: Value =
                serde_json::from_str(&chunk.data).expect("Failed to parse chunk JSON");

            if let Some(raw_response) = chunk_json.get("raw_response")
                && let Some(arr) = raw_response.as_array()
            {
                for entry in arr {
                    assert_error_raw_response_entry(entry);
                }
                if !arr.is_empty() {
                    found_error_raw_response = true;
                }
            }

            if chunk_json.get("raw_chunk").is_some() {
                found_raw_chunk = true;
            }
        }

        assert!(
            found_error_raw_response,
            "Cache miss: should have error raw_response entries from failed provider"
        );
        assert!(
            found_raw_chunk,
            "Cache miss: should have raw_chunk from successful provider streaming"
        );
    }

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // --- Second request (cache hit for good provider) ---
    {
        let mut chunks = Client::new()
            .post(get_gateway_endpoint("/inference"))
            .json(&payload)
            .eventsource()
            .await
            .expect("Failed to create eventsource for second request");

        let mut found_error_raw_response = false;
        let mut found_raw_chunk = false;

        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.expect("Failed to receive chunk");
            let Event::Message(chunk) = chunk else {
                continue;
            };
            if chunk.data == "[DONE]" {
                break;
            }

            let chunk_json: Value =
                serde_json::from_str(&chunk.data).expect("Failed to parse chunk JSON");

            if let Some(raw_response) = chunk_json.get("raw_response")
                && let Some(arr) = raw_response.as_array()
            {
                for entry in arr {
                    assert_error_raw_response_entry(entry);
                }
                if !arr.is_empty() {
                    found_error_raw_response = true;
                }
            }

            if chunk_json.get("raw_chunk").is_some() {
                found_raw_chunk = true;
            }
        }

        assert!(
            found_error_raw_response,
            "Cache hit: should still have error raw_response entries (errors are never cached)"
        );
        // Streaming cache hits suppress raw_chunk (consistent with non-streaming excluding success entries).
        assert!(
            !found_raw_chunk,
            "Cache hit: raw_chunk should be suppressed (cached response has no raw provider data)"
        );
    }
}

// =============================================================================
// Part A: Cache Hit/Miss — DICL Partial Fail
// =============================================================================

/// Cache hit/miss cycle: DICL with embedding partial failure, non-streaming.
///
/// First request (cache miss): 3 entries (1 embedding error + 1 embedding success + 1 LLM success).
/// Second request (cache hit): 1 entry (1 embedding error only — embedding success and LLM are cached).
#[tokio::test(flavor = "multi_thread")]
async fn test_cache_hit_miss_dicl_partial_fail() {
    let unique_input = format!("cache test: dicl partial fail {}", Uuid::now_v7());
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "dicl_embedding_partial_fail",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [{"role": "user", "content": unique_input}]
        },
        "stream": false,
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60}
    });

    // --- First request (cache miss) ---
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "First request should succeed (embedding fallback + LLM succeed)"
    );

    let response_json: Value = response.json().await.unwrap();
    let raw_response_array = response_json["raw_response"]
        .as_array()
        .expect("raw_response should be an array");
    assert_eq!(
        raw_response_array.len(),
        3,
        "Cache miss: expected 3 entries (1 embedding error + 1 embedding success + 1 LLM success)"
    );
    assert_error_raw_response_entry(&raw_response_array[0]);
    assert_raw_response_entry(&raw_response_array[1]);
    assert_raw_response_entry(&raw_response_array[2]);

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // --- Second request (cache hit for embedding success + LLM) ---
    let response2 = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "Second request should succeed (cache hit)"
    );

    let response2_json: Value = response2.json().await.unwrap();
    let raw_response_array2 = response2_json["raw_response"]
        .as_array()
        .expect("raw_response should be an array on cache hit");
    // On cache hit: both the embedding success and LLM results are cached,
    // so only the embedding error entry remains.
    assert_eq!(
        raw_response_array2.len(),
        1,
        "Cache hit: expected 1 entry (embedding error only — embedding success and LLM cached)"
    );
    assert_error_raw_response_entry(&raw_response_array2[0]);
}

// =============================================================================
// Part A: Cache Hit/Miss — BestOfN Partial Fail
// =============================================================================

/// Cache hit/miss cycle: Best-of-N with one candidate failing, non-streaming.
///
/// First request (cache miss): 2 entries (1 error + 1 success).
/// Second request (cache hit): 1 entry (error only — success candidate cached).
#[tokio::test]
async fn test_cache_hit_miss_best_of_n_partial_fail() {
    let unique_input = format!("cache test: best_of_n partial fail {}", Uuid::now_v7());
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "best_of_n_raw_response_partial_fail",
        "variant_name": "best_of_n",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": unique_input}]
        },
        "stream": false,
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60}
    });

    // --- First request (cache miss) ---
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "First request should succeed (one candidate survived)"
    );

    let response_json: Value = response.json().await.unwrap();
    let raw_response_array = response_json["raw_response"]
        .as_array()
        .expect("raw_response should be an array");
    assert_eq!(
        raw_response_array.len(),
        2,
        "Cache miss: expected 2 entries (1 error candidate + 1 success candidate)"
    );
    assert_error_raw_response_entry(&raw_response_array[0]);
    assert_raw_response_entry(&raw_response_array[1]);

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // --- Second request (cache hit for success candidate) ---
    let response2 = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "Second request should succeed (cache hit)"
    );

    let response2_json: Value = response2.json().await.unwrap();
    let raw_response_array2 = response2_json["raw_response"]
        .as_array()
        .expect("raw_response should be an array on cache hit");
    assert_eq!(
        raw_response_array2.len(),
        1,
        "Cache hit: expected 1 entry (error only — success candidate cached)"
    );
    assert_error_raw_response_entry(&raw_response_array2[0]);
}

// =============================================================================
// Part A: Cache Hit/Miss — MixtureOfN Partial Fail
// =============================================================================

/// Cache hit/miss cycle: Mixture-of-N with one candidate failing, non-streaming.
///
/// First request (cache miss): 2 entries (1 error + 1 success).
/// Second request (cache hit): 1 entry (error only — success candidate cached).
#[tokio::test]
async fn test_cache_hit_miss_mixture_of_n_partial_fail() {
    let unique_input = format!("cache test: mixture_of_n partial fail {}", Uuid::now_v7());
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "mixture_of_n_raw_response_partial_fail",
        "variant_name": "mixture_of_n",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": unique_input}]
        },
        "stream": false,
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60}
    });

    // --- First request (cache miss) ---
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "First request should succeed (one candidate survived)"
    );

    let response_json: Value = response.json().await.unwrap();
    let raw_response_array = response_json["raw_response"]
        .as_array()
        .expect("raw_response should be an array");
    assert_eq!(
        raw_response_array.len(),
        2,
        "Cache miss: expected 2 entries (1 error candidate + 1 success candidate)"
    );
    assert_error_raw_response_entry(&raw_response_array[0]);
    assert_raw_response_entry(&raw_response_array[1]);

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // --- Second request (cache hit for success candidate) ---
    let response2 = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "Second request should succeed (cache hit)"
    );

    let response2_json: Value = response2.json().await.unwrap();
    let raw_response_array2 = response2_json["raw_response"]
        .as_array()
        .expect("raw_response should be an array on cache hit");
    assert_eq!(
        raw_response_array2.len(),
        1,
        "Cache hit: expected 1 entry (error only — success candidate cached)"
    );
    assert_error_raw_response_entry(&raw_response_array2[0]);
}

// =============================================================================
// Part A: Cache Hit/Miss — Retries with Fallback
// =============================================================================

/// Cache hit/miss cycle: retries with provider fallback (error provider -> good provider), non-streaming.
///
/// First request (cache miss): 2 entries (1 error + 1 success).
/// Second request (cache hit): 1 entry (error only — success provider cached).
#[tokio::test]
async fn test_cache_hit_miss_retries_with_fallback() {
    let unique_input = format!("cache test: retries with fallback {}", Uuid::now_v7());
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "raw_response_retries_with_provider_fallback",
        "variant_name": "test",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": unique_input}]
        },
        "stream": false,
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60}
    });

    // --- First request (cache miss) ---
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "First request should succeed (provider fallback succeeds)"
    );

    let response_json: Value = response.json().await.unwrap();
    let raw_response_array = response_json["raw_response"]
        .as_array()
        .expect("raw_response should be an array");
    assert_eq!(
        raw_response_array.len(),
        2,
        "Cache miss: expected 2 entries (1 error + 1 success)"
    );
    assert_error_raw_response_entry(&raw_response_array[0]);
    assert_raw_response_entry(&raw_response_array[1]);

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // --- Second request (cache hit for good provider) ---
    let response2 = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "Second request should succeed (cache hit)"
    );

    let response2_json: Value = response2.json().await.unwrap();
    let raw_response_array2 = response2_json["raw_response"]
        .as_array()
        .expect("raw_response should be an array on cache hit");
    assert_eq!(
        raw_response_array2.len(),
        1,
        "Cache hit: expected 1 entry (error only — success provider cached)"
    );
    assert_error_raw_response_entry(&raw_response_array2[0]);
}

// =============================================================================
// Part A: Cache Hit/Miss — Embeddings Fallback
// =============================================================================

/// Cache hit/miss cycle: embeddings endpoint with fallback (error provider -> good provider).
///
/// First request (cache miss): 2 `tensorzero::raw_response` entries (1 error + 1 success).
/// Second request (cache hit): 1 entry (error only — success provider is cached).
#[tokio::test(flavor = "multi_thread")]
async fn test_cache_hit_miss_embeddings_fallback() {
    let unique_input = format!("cache test: embeddings fallback {}", Uuid::now_v7());

    let payload = json!({
        "input": unique_input,
        "model": "tensorzero::embedding_model_name::embedding_fallback_with_raw_response",
        "tensorzero::include_raw_response": true,
        "tensorzero::cache_options": {"enabled": "on", "max_age_s": 60}
    });

    // --- First request (cache miss) ---
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "First request should succeed (fallback to good provider)"
    );

    let response_json: Value = response.json().await.unwrap();
    let raw_response_array = response_json["tensorzero::raw_response"]
        .as_array()
        .expect("tensorzero::raw_response should be an array");
    assert_eq!(
        raw_response_array.len(),
        2,
        "Cache miss: expected 2 entries (1 error + 1 success)"
    );
    assert_error_raw_response_entry(&raw_response_array[0]);
    // Success entry should have model_inference_id
    assert!(
        raw_response_array[1].get("model_inference_id").is_some(),
        "Success entry should have model_inference_id"
    );

    // Wait for cache write
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // --- Second request (cache hit for good provider) ---
    let response2 = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "Second request should succeed (cache hit)"
    );

    let response2_json: Value = response2.json().await.unwrap();
    let raw_response_array2 = response2_json["tensorzero::raw_response"]
        .as_array()
        .expect("tensorzero::raw_response should be an array on cache hit");
    // The embeddings endpoint includes failed_raw_response entries on cache hit
    // (errors are never cached, so they are always fresh).
    assert_eq!(
        raw_response_array2.len(),
        1,
        "Cache hit: expected 1 entry (error only — success provider cached)"
    );
    assert_error_raw_response_entry(&raw_response_array2[0]);
}

// =============================================================================
// Part B: Error + Cache — Simple Chat Error
// =============================================================================

/// Error + cache regression: shorthand model with extra_body error.
/// Errors are never cached, so `cache_options` should not affect the error raw_response.
#[tokio::test(flavor = "multi_thread")]
async fn test_cache_error_simple_chat() {
    let unique_input = format!("cache test: simple chat error {}", Uuid::now_v7());

    let payload = json!({
        "model_name": "openai::gpt-5-mini",
        "input": {
            "messages": [{"role": "user", "content": [{"type": "text", "text": unique_input}]}]
        },
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60},
        "extra_body": [{"pointer": "/nonsense_field", "value": "garbage_12345"}]
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error, got status {}",
        response.status()
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response_array = response_json["raw_response"]
        .as_array()
        .expect("Error response should have raw_response array");
    assert_eq!(
        raw_response_array.len(),
        1,
        "Expected 1 error entry from single provider"
    );
    assert_error_raw_response_entry(&raw_response_array[0]);
    assert_eq!(
        raw_response_array[0]["provider_type"].as_str().unwrap(),
        "openai",
        "Provider type should be `openai`"
    );
    assert_eq!(
        raw_response_array[0]["api_type"].as_str().unwrap(),
        "chat_completions",
        "API type should be `chat_completions`"
    );
}

// =============================================================================
// Part B: Error + Cache — DICL Embedding All Fail
// =============================================================================

/// Error + cache regression: DICL where all embedding providers fail.
/// Errors are never cached, so `cache_options` should not affect the error raw_response.
#[tokio::test(flavor = "multi_thread")]
async fn test_cache_error_dicl_embedding_all_fail() {
    let unique_input = format!("cache test: dicl embedding all fail {}", Uuid::now_v7());
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "dicl_embedding_all_fail",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [{"role": "user", "content": unique_input}]
        },
        "stream": false,
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60}
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (all embedding providers failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response_array = response_json["raw_response"]
        .as_array()
        .expect("Error response should have raw_response array");
    assert_eq!(
        raw_response_array.len(),
        2,
        "Expected 2 error entries from 2 failed embedding providers"
    );
    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry["api_type"].as_str().unwrap(),
            "embeddings",
            "Failed entries should have api_type `embeddings`"
        );
    }
}

// =============================================================================
// Part B: Error + Cache — DICL LLM Fail
// =============================================================================

/// Error + cache regression: DICL where embedding succeeds but LLM fails.
/// Errors are never cached, so `cache_options` should not affect the error raw_response.
#[tokio::test(flavor = "multi_thread")]
async fn test_cache_error_dicl_llm_fail() {
    let unique_input = format!("cache test: dicl llm fail {}", Uuid::now_v7());
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "dicl_llm_fail",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [{"role": "user", "content": unique_input}]
        },
        "stream": false,
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60}
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (LLM provider failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response_array = response_json["raw_response"]
        .as_array()
        .expect("Error response should have raw_response array");
    assert_eq!(
        raw_response_array.len(),
        1,
        "Expected 1 error entry from failed LLM provider"
    );
    assert_error_raw_response_entry(&raw_response_array[0]);
    assert_eq!(
        raw_response_array[0]["api_type"].as_str().unwrap(),
        "chat_completions",
        "Failed LLM entry should have api_type `chat_completions`"
    );
}

// =============================================================================
// Part B: Error + Cache — BestOfN All Fail
// =============================================================================

/// Error + cache regression: Best-of-N where all candidates fail.
/// Errors are never cached, so `cache_options` should not affect the error raw_response.
#[tokio::test]
async fn test_cache_error_best_of_n_all_fail() {
    let unique_input = format!("cache test: best_of_n all fail {}", Uuid::now_v7());
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "best_of_n_raw_response_all_fail",
        "variant_name": "best_of_n",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": unique_input}]
        },
        "stream": false,
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60}
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (all candidates failed), got status {}",
        response.status()
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response_array = response_json["raw_response"]
        .as_array()
        .expect("Error response should have raw_response array");
    assert_eq!(
        raw_response_array.len(),
        2,
        "Expected 2 error entries from 2 failed candidates"
    );
    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
    }
}

// =============================================================================
// Part B: Error + Cache — MixtureOfN All Fail
// =============================================================================

/// Error + cache regression: Mixture-of-N where all candidates fail.
/// Errors are never cached, so `cache_options` should not affect the error raw_response.
#[tokio::test]
async fn test_cache_error_mixture_of_n_all_fail() {
    let unique_input = format!("cache test: mixture_of_n all fail {}", Uuid::now_v7());
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "mixture_of_n_raw_response_all_fail",
        "variant_name": "mixture_of_n",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": unique_input}]
        },
        "stream": false,
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60}
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (all candidates failed), got status {}",
        response.status()
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response_array = response_json["raw_response"]
        .as_array()
        .expect("Error response should have raw_response array");
    assert_eq!(
        raw_response_array.len(),
        2,
        "Expected 2 error entries from 2 failed candidates"
    );
    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
    }
}

// =============================================================================
// Part B: Error + Cache — Fallback Both Fail
// =============================================================================

/// Error + cache regression: model fallback where both providers fail.
/// Errors are never cached, so `cache_options` should not affect the error raw_response.
#[tokio::test]
async fn test_cache_error_fallback_both_fail() {
    let unique_input = format!("cache test: fallback both fail {}", Uuid::now_v7());
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "model_both_error_with_raw_response",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": unique_input}]
        },
        "stream": false,
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60}
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (all providers failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response_array = response_json["raw_response"]
        .as_array()
        .expect("Error response should have raw_response array");
    assert_eq!(
        raw_response_array.len(),
        2,
        "Expected 2 error entries from 2 failed providers"
    );
    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry["provider_type"].as_str().unwrap(),
            "dummy",
            "Failed provider_type should be `dummy`"
        );
    }
}

// =============================================================================
// Part B: Error + Cache — Retries All Fail
// =============================================================================

/// Error + cache regression: all retry attempts fail.
/// With 2 retries = 3 total attempts. Errors are never cached.
#[tokio::test]
async fn test_cache_error_retries_all_fail() {
    let unique_input = format!("cache test: retries all fail {}", Uuid::now_v7());
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "raw_response_retries_all_fail",
        "variant_name": "test",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": unique_input}]
        },
        "stream": false,
        "include_raw_response": true,
        "cache_options": {"enabled": "on", "max_age_s": 60}
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (all retries failed)"
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response_array = response_json["raw_response"]
        .as_array()
        .expect("Error response should have raw_response array");
    assert_eq!(
        raw_response_array.len(),
        3,
        "Expected 3 error entries from 3 failed attempts (1 initial + 2 retries)"
    );
    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry["provider_type"].as_str().unwrap(),
            "dummy",
            "Failed attempt provider_type should be `dummy`"
        );
    }
}

// =============================================================================
// Part B: Error + Cache — Embeddings All Fail
// =============================================================================

/// Error + cache regression: embeddings endpoint where all providers fail.
/// Errors are never cached, so `cache_options` should not affect the error raw_response.
#[tokio::test(flavor = "multi_thread")]
async fn test_cache_error_embeddings_all_fail() {
    let unique_input = format!("cache test: embeddings all fail {}", Uuid::now_v7());

    let payload = json!({
        "input": unique_input,
        "model": "tensorzero::embedding_model_name::embedding_both_error_with_raw_response",
        "tensorzero::include_raw_response": true,
        "tensorzero::cache_options": {"enabled": "on", "max_age_s": 60}
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        !response.status().is_success(),
        "Response should be an error (all embedding providers failed), got status {}",
        response.status()
    );

    let response_json: Value = response.json().await.unwrap();

    let raw_response_array = response_json["tensorzero_raw_response"]
        .as_array()
        .expect("Error response should have tensorzero_raw_response array");
    assert_eq!(
        raw_response_array.len(),
        2,
        "Expected 2 error entries from 2 failed embedding providers"
    );
    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry["api_type"].as_str().unwrap(),
            "embeddings",
            "Failed entries should have api_type `embeddings`"
        );
    }
}
