//! E2E tests for `raw_response` in best-of-n candidate/evaluator failure scenarios.
//!
//! Scenarios:
//! - Partial candidate failure: one candidate errors, one succeeds → 200 OK with error+success entries
//! - All candidates fail: both candidates error → error response with raw_response entries
//! - Evaluator failure: both candidates succeed, evaluator errors → 200 OK with random selection + error entries

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use crate::raw_response::assert_raw_response_entry;

use super::assert_error_raw_response_entry;

// =============================================================================
// Partial candidate failure (1 error + 1 good → single-candidate fast path)
// =============================================================================

/// Non-streaming: one candidate fails with raw_response, one succeeds.
/// With only 1 surviving candidate, the single-candidate fast path returns without
/// running the evaluator. Expects: 1 error entry (failed candidate) + 1 success entry.
#[tokio::test]
async fn test_best_of_n_partial_fail_non_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "best_of_n_raw_response_partial_fail",
        "variant_name": "best_of_n",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
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
        "Response should be successful (one candidate survived)"
    );

    let response_text = response.text().await.unwrap();
    let response_json: Value =
        serde_json::from_str(&response_text).expect("Response should be valid JSON");

    let raw_response = response_json.get("raw_response").unwrap_or_else(|| {
        panic!("Response should have `raw_response` when include_raw_response=true. Response: {response_text}")
    });
    let raw_response_array = raw_response.as_array().unwrap();

    // 1 error entry (failed candidate) + 1 success entry (good candidate)
    assert_eq!(
        raw_response_array.len(),
        2,
        "Expected 2 raw_response entries (1 failed candidate + 1 good candidate), got {}. Response: {response_text}",
        raw_response_array.len()
    );

    // First entry: error from failed candidate (no model_inference_id)
    let error_entry = &raw_response_array[0];
    assert_error_raw_response_entry(error_entry);
    assert_eq!(
        error_entry.get("provider_type").unwrap().as_str().unwrap(),
        "dummy",
        "Failed candidate provider_type should be `dummy`"
    );
    assert_eq!(
        error_entry.get("data").unwrap().as_str().unwrap(),
        "dummy error raw response",
        "Failed candidate data should be `dummy error raw response`"
    );

    // Second entry: success from good candidate (has model_inference_id)
    let success_entry = &raw_response_array[1];
    assert_raw_response_entry(success_entry);
    assert_eq!(
        success_entry
            .get("provider_type")
            .unwrap()
            .as_str()
            .unwrap(),
        "dummy",
        "Successful candidate provider_type should be `dummy`"
    );
}

/// Streaming: one candidate fails, one succeeds.
/// Best-of-N uses fake streaming. The surviving candidate IS the stream, so its
/// raw_response is not in a `raw_response` chunk — only the failed candidate's
/// error entry appears there.
#[tokio::test]
async fn test_best_of_n_partial_fail_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "best_of_n_raw_response_partial_fail",
        "variant_name": "best_of_n",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": true,
        "include_raw_response": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .expect("Failed to create eventsource");

    let mut error_count = 0;
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

        if chunk_json.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }

        if let Some(raw_response) = chunk_json.get("raw_response")
            && let Some(arr) = raw_response.as_array()
        {
            for entry in arr {
                assert_error_raw_response_entry(entry);
                error_count += 1;
            }
        }
    }

    assert!(
        !found_raw_chunk,
        "Best-of-N streaming should NOT have raw_chunk (fake streaming)"
    );
    // In streaming, the selected candidate IS the stream — its raw_response is not
    // emitted as a `raw_response` chunk. Only the failed candidate's error entry appears.
    assert_eq!(
        error_count, 1,
        "Expected 1 error entry (failed candidate) in streaming raw_response chunks"
    );
}

// =============================================================================
// All candidates fail
// =============================================================================

/// Non-streaming: both candidates fail. Expects an error response with
/// raw_response entries from both failed candidates.
#[tokio::test]
async fn test_best_of_n_all_fail_non_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "best_of_n_raw_response_all_fail",
        "variant_name": "best_of_n",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
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

    assert!(
        !response.status().is_success(),
        "Response should be an error (all candidates failed), got status {}",
        response.status()
    );

    let response_text = response.text().await.unwrap();
    let response_json: Value =
        serde_json::from_str(&response_text).expect("Response should be valid JSON");

    let error_field = response_json.get("error").unwrap_or_else(|| {
        panic!("Error response should have `error` field. Response: {response_text}")
    });
    assert!(
        error_field.is_string(),
        "T0 native error field should be a string, got: {error_field:?}"
    );

    let raw_response = response_json.get("raw_response").unwrap_or_else(|| {
        panic!("Error response should have `raw_response` when include_raw_response=true. Response: {response_text}")
    });
    let raw_response_array = raw_response.as_array().unwrap();

    // 2 error entries (both candidates failed)
    assert_eq!(
        raw_response_array.len(),
        2,
        "Expected 2 raw_response entries (2 failed candidates), got {}. Response: {response_text}",
        raw_response_array.len()
    );

    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry.get("provider_type").unwrap().as_str().unwrap(),
            "dummy",
            "Failed candidate provider_type should be `dummy`"
        );
        assert_eq!(
            entry.get("data").unwrap().as_str().unwrap(),
            "dummy error raw response",
            "Failed candidate data should be `dummy error raw response`"
        );
    }
}

/// Streaming: both candidates fail. Terminal error before any SSE chunk,
/// so the response is a regular HTTP error (not SSE).
#[tokio::test]
async fn test_best_of_n_all_fail_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "best_of_n_raw_response_all_fail",
        "variant_name": "best_of_n",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": true,
        "include_raw_response": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Terminal error: non-200 JSON response, not SSE
    assert!(
        !response.status().is_success(),
        "Response should be an error (all candidates failed), got status {}",
        response.status()
    );

    let response_text = response.text().await.unwrap();
    let response_json: Value =
        serde_json::from_str(&response_text).expect("Response should be valid JSON");

    let error_field = response_json.get("error").unwrap_or_else(|| {
        panic!("Error response should have `error` field. Response: {response_text}")
    });
    assert!(
        error_field.is_string(),
        "T0 native error field should be a string, got: {error_field:?}"
    );

    let raw_response = response_json.get("raw_response").unwrap_or_else(|| {
        panic!("Error response should have `raw_response` when include_raw_response=true. Response: {response_text}")
    });
    let raw_response_array = raw_response.as_array().unwrap();

    assert_eq!(
        raw_response_array.len(),
        2,
        "Expected 2 raw_response entries (2 failed candidates), got {}. Response: {response_text}",
        raw_response_array.len()
    );

    for entry in raw_response_array {
        assert_error_raw_response_entry(entry);
        assert_eq!(
            entry.get("provider_type").unwrap().as_str().unwrap(),
            "dummy",
            "Failed candidate provider_type should be `dummy`"
        );
    }
}

// =============================================================================
// Evaluator failure (both candidates succeed, evaluator errors)
// =============================================================================

/// Non-streaming: both candidates succeed, evaluator fails with raw_response.
/// Expects 200 OK (random selection) with 2 success entries + 1 error entry.
#[tokio::test]
async fn test_best_of_n_evaluator_fail_non_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "best_of_n_raw_response_evaluator_fail",
        "variant_name": "best_of_n",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
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
        "Response should be successful (evaluator failed, random candidate selected)"
    );

    let response_text = response.text().await.unwrap();
    let response_json: Value =
        serde_json::from_str(&response_text).expect("Response should be valid JSON");

    let raw_response = response_json.get("raw_response").unwrap_or_else(|| {
        panic!("Response should have `raw_response` when include_raw_response=true. Response: {response_text}")
    });
    let raw_response_array = raw_response.as_array().unwrap();

    // 2 success entries (both candidates) + 1 error entry (evaluator)
    assert_eq!(
        raw_response_array.len(),
        3,
        "Expected 3 raw_response entries (2 candidates + 1 failed evaluator), got {}. Response: {response_text}",
        raw_response_array.len()
    );

    let mut success_count = 0;
    let mut error_count = 0;
    for entry in raw_response_array {
        if entry.get("model_inference_id").is_some() {
            success_count += 1;
            assert_raw_response_entry(entry);
        } else {
            error_count += 1;
            assert_error_raw_response_entry(entry);
            assert_eq!(
                entry.get("provider_type").unwrap().as_str().unwrap(),
                "dummy",
                "Failed evaluator provider_type should be `dummy`"
            );
            assert_eq!(
                entry.get("data").unwrap().as_str().unwrap(),
                "dummy error raw response",
                "Failed evaluator data should be `dummy error raw response`"
            );
        }
    }
    assert_eq!(
        success_count, 2,
        "Expected 2 success entries (both candidates)"
    );
    assert_eq!(error_count, 1, "Expected 1 error entry (failed evaluator)");
}

/// Streaming: both candidates succeed, evaluator fails.
/// Best-of-N uses fake streaming. The selected candidate IS the stream, so its
/// raw_response is not in a `raw_response` chunk. Only the non-selected candidate
/// and the failed evaluator appear there.
#[tokio::test]
async fn test_best_of_n_evaluator_fail_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "best_of_n_raw_response_evaluator_fail",
        "variant_name": "best_of_n",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": true,
        "include_raw_response": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .expect("Failed to create eventsource");

    let mut found_raw_chunk = false;
    let mut success_count = 0;
    let mut error_count = 0;

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

        if chunk_json.get("raw_chunk").is_some() {
            found_raw_chunk = true;
        }

        if let Some(raw_response) = chunk_json.get("raw_response")
            && let Some(arr) = raw_response.as_array()
        {
            for entry in arr {
                if entry.get("model_inference_id").is_some() {
                    success_count += 1;
                } else {
                    error_count += 1;
                    assert_error_raw_response_entry(entry);
                }
            }
        }
    }

    assert!(
        !found_raw_chunk,
        "Best-of-N streaming should NOT have raw_chunk (fake streaming)"
    );
    // In streaming, the selected candidate IS the stream — only the non-selected
    // candidate appears as a raw_response entry.
    assert_eq!(
        success_count, 1,
        "Expected 1 success entry (non-selected candidate; selected candidate is the stream)"
    );
    assert_eq!(error_count, 1, "Expected 1 error entry (failed evaluator)");
}
